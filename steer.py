"""
Feature Steering Entrypoint

This script loads a model with trained TopKLoRALinearSTE adapters and applies
feature steering to enable/disable specific latents during inference.

Usage:
    python steer.py                           # Use default config
    python steer.py experiment_name=my_test   # Override experiment name
    python steer.py --config-name dpo_512_4_example  # Use specific config

Example with overrides:
    python steer.py model.adapter_path=path/to/adapter steering.verbose=true
"""

from dotenv import load_dotenv
import wandb
import torch
import hydra
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig
)
from peft import PeftModel
from omegaconf import DictConfig, OmegaConf
import logging
import os
from pathlib import Path
from datetime import datetime
import json

from src.steering import (
    steer_features,
    FeatureSteeringContext,
    list_available_adapters,
    remove_steering_hooks
)
from src.utils import build_quant_config


def load_model_and_adapter(cfg: DictConfig):
    """
    Load base model and TopKLoRALinearSTE adapter.
    
    Args:
        cfg: Hydra config with model settings
        
    Returns:
        tuple: (model, tokenizer)
    """
    logging.info(f"Loading base model: {cfg.model.model_name}")
    
    # Build quantization config if needed
    quant_cfg = None
    if cfg.model.load_in_4bit or cfg.model.load_in_8bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg.model.load_in_4bit,
            load_in_8bit=cfg.model.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=dtype_map[cfg.model.dtype],
        device_map=cfg.model.device if cfg.model.device != "cpu" else None,
        quantization_config=quant_cfg,
        trust_remote_code=True,
        attn_implementation='eager'
    )
    
    # Load adapter if specified
    if cfg.model.adapter_path:
        logging.info(f"Loading adapter from: {cfg.model.adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            cfg.model.adapter_path,
            is_trainable=False
        )
        model.eval()
        logging.info("✅ Adapter loaded successfully")
    else:
        logging.warning("No adapter_path specified - running with base model only")
    
    return model, tokenizer


def parse_steering_config(cfg: DictConfig) -> dict:
    """
    Parse steering configuration into the format expected by steer_features().
    
    Args:
        cfg: Hydra config with steering settings
        
    Returns:
        Dictionary mapping adapter names to list of (feature_num, effect) tuples
    """
    feature_dict = {}
    
    if not hasattr(cfg.steering, 'features') or not cfg.steering.features:
        logging.warning("No steering features specified in config")
        return feature_dict
    
    for adapter_name, feature_list in cfg.steering.features.items():
        if not feature_list:
            continue
            
        features = []
        for item in feature_list:
            feature_num = item.feature
            effect = item.effect
            features.append((feature_num, effect))
            
            if hasattr(item, 'description'):
                logging.info(
                    f"  Feature {feature_num} ({effect}): {item.description}"
                )
        
        feature_dict[adapter_name] = features
    
    return feature_dict


def generate_with_prompts(model, tokenizer, prompts: list, cfg: DictConfig) -> list:
    """
    Generate outputs for a list of prompts.
    
    Args:
        model: The model to generate with
        tokenizer: The tokenizer
        prompts: List of prompt strings
        cfg: Config with generation settings
        
    Returns:
        List of generated strings
    """
    outputs = []
    
    for prompt in prompts:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move to device
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = next(model.parameters()).device
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=cfg.generation.do_sample,
                temperature=cfg.generation.temperature if cfg.generation.do_sample else None,
                top_p=cfg.generation.top_p if cfg.generation.do_sample else None,
                repetition_penalty=cfg.generation.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(output_text)
    
    return outputs


def run_steering(cfg: DictConfig):
    """
    Main steering function.
    
    Args:
        cfg: Hydra configuration
    """
    logging.info("="*80)
    logging.info("Feature Steering")
    logging.info("="*80)
    
    # Load model and adapter
    model, tokenizer = load_model_and_adapter(cfg)
    
    # List available adapters if requested
    if cfg.steering.list_adapters:
        logging.info("\n" + "="*80)
        logging.info("Discovering available adapters...")
        logging.info("="*80)
        adapters_info = list_available_adapters(model, verbose=True)
    
    # Parse steering configuration
    logging.info("\n" + "="*80)
    logging.info("Parsing steering configuration...")
    logging.info("="*80)
    feature_dict = parse_steering_config(cfg)
    
    if not feature_dict:
        logging.warning("No features to steer - exiting")
        return
    
    # Prepare output directory
    output_dir = Path(cfg.output.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    results = {
        "experiment_name": cfg.experiment_name,
        "model": cfg.model.model_name,
        "adapter": cfg.model.adapter_path,
        "steering_config": {k: v for k, v in feature_dict.items()},
        "prompts": cfg.prompts,
        "outputs": {}
    }
    
    # Generate baseline outputs (no steering)
    if cfg.output.compare_baseline:
        logging.info("\n" + "="*80)
        logging.info("Generating BASELINE outputs (no steering)...")
        logging.info("="*80)
        
        baseline_outputs = generate_with_prompts(
            model, tokenizer, cfg.prompts, cfg
        )
        
        results["outputs"]["baseline"] = []
        for i, (prompt, output) in enumerate(zip(cfg.prompts, baseline_outputs)):
            results["outputs"]["baseline"].append({
                "prompt": prompt,
                "output": output
            })
            
            if cfg.output.print_outputs:
                logging.info(f"\n--- Baseline {i+1}/{len(cfg.prompts)} ---")
                logging.info(f"Prompt: {prompt}")
                logging.info(f"Output: {output}")
    
    # Generate steered outputs
    logging.info("\n" + "="*80)
    logging.info("Generating STEERED outputs...")
    logging.info("="*80)
    
    # Get amplification from config (default to 1.0)
    amplification = cfg.steering.get("amplification", 1.0)
    logging.info(f"Using amplification: {amplification}x")
    
    with FeatureSteeringContext(model, feature_dict, verbose=cfg.steering.verbose, amplification=amplification):
        steered_outputs = generate_with_prompts(
            model, tokenizer, cfg.prompts, cfg
        )
    
    results["outputs"]["steered"] = []
    for i, (prompt, output) in enumerate(zip(cfg.prompts, steered_outputs)):
        results["outputs"]["steered"].append({
            "prompt": prompt,
            "output": output
        })
        
        if cfg.output.print_outputs:
            logging.info(f"\n--- Steered {i+1}/{len(cfg.prompts)} ---")
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Output: {output}")
    
    # Save results
    if cfg.output.save_outputs:
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"\n✅ Results saved to: {results_file}")
        
        # Also save a readable comparison
        comparison_file = output_dir / "comparison.txt"
        with open(comparison_file, "w") as f:
            f.write("Feature Steering Comparison\n")
            f.write("="*80 + "\n\n")
            
            for i, prompt in enumerate(cfg.prompts):
                f.write(f"Prompt {i+1}: {prompt}\n")
                f.write("-"*80 + "\n")
                
                if cfg.output.compare_baseline:
                    f.write(f"BASELINE:\n{results['outputs']['baseline'][i]['output']}\n\n")
                
                f.write(f"STEERED:\n{results['outputs']['steered'][i]['output']}\n\n")
                f.write("="*80 + "\n\n")
        
        logging.info(f"✅ Comparison saved to: {comparison_file}")
    
    # Log to wandb if enabled
    if cfg.logger.wandb_mode != "disabled":
        wandb.log({"results": results})
    
    logging.info("\n" + "="*80)
    logging.info("✅ Steering complete!")
    logging.info("="*80)


@hydra.main(
    version_base=None,
    config_path="config/steer_config",
    config_name="default"
)
def main(cfg: DictConfig):
    """Main entrypoint for feature steering."""
    load_dotenv()
    
    # Set seeds for reproducibility
    torch.manual_seed(cfg.get("seed", 42))
    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    set_seed(cfg.get("seed", 42))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Loaded configuration:")
    logging.info(OmegaConf.to_yaml(cfg))
    
    # Initialize wandb
    wandb.init(
        project=cfg.logger.wandb_project,
        entity=cfg.logger.wandb_entity,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logger.wandb_mode
    )
    
    try:
        run_steering(cfg)
    except Exception as e:
        logging.error(f"Error during steering: {e}", exc_info=True)
        raise
    finally:
        if cfg.logger.wandb_mode != "disabled":
            wandb.finish()


if __name__ == '__main__':
    main()

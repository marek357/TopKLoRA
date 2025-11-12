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
    remove_steering_hooks,
    get_steering_statistics
)
# Use the eval pipeline's tokenizer/model init helper so steering and eval use the same wrappers
from src.evals import init_model_tokenizer_fixed, hash_lora_weights
from src.utils import build_quant_config

# Note: load_model_and_adapter() has been replaced by init_model_tokenizer_fixed()
# from src.evals to ensure steering and eval pipelines use the same model loading logic


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
        # Format prompt using chat template if available (matches eval pipeline)
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            # Wrap in chat format (assume user message)
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = prompt

        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors="pt",
                           padding=True)        # Move to device
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

    # Load model, tokenizer and TopK-wrapped modules using the eval helper
    # init_model_tokenizer_fixed returns (model, tokenizer, wrapped_modules)
    model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(cfg.model)
    logging.info(
        f"Loaded model and tokenizer via init_model_tokenizer_fixed; wrapped modules: {len(wrapped_modules)}")

    # Log adapter hash for reproducibility (matches eval pipeline)
    adapter_hash = hash_lora_weights(model)
    logging.info(f"Adapter SHA256 hash: {adapter_hash}")

    # List available adapters if requested
    if cfg.steering.list_adapters:
        logging.info("\n" + "="*80)
        logging.info("Discovering available adapters...")
        logging.info("="*80)
        adapters_info = list_available_adapters(
            model, wrapped_modules=wrapped_modules, verbose=True)

    # Parse steering configuration
    logging.info("\n" + "="*80)
    logging.info("Parsing steering configuration...")
    logging.info("="*80)
    feature_dict = parse_steering_config(cfg)

    if not feature_dict:
        logging.warning("No features to steer - exiting")
        return

    # Prepare output directory
    output_dir = Path(cfg.output.output_dir) / \
        datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Convert OmegaConf types to native Python for JSON serialization
    results = {
        "experiment_name": str(cfg.experiment_name),
        "model": str(cfg.model.name),
        "adapter": str(cfg.model.adapter_checkpoint_dir),
        # "steering_config": {k: v for k, v in feature_dict.items()},
        # Convert ListConfig to list of strings
        "prompts": [str(p) for p in cfg.prompts],
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

    # Get amplification(s) from config (default to 1.0). Support float or list of floats.
    # Convert OmegaConf types to native Python types
    amplification_cfg = cfg.steering.get("amplification", 1.0)

    # Convert to native Python types (handles ListConfig, etc.)
    if isinstance(amplification_cfg, (list, tuple)) or OmegaConf.is_list(amplification_cfg):
        amplification_list = [float(x) for x in amplification_cfg]
    else:
        amplification_list = [float(amplification_cfg)]

    logging.info(f"Using amplification values: {amplification_list}")

    # For each amplification (dose), apply steering and generate outputs. Store per-dose outputs.
    results["outputs"]["steered"] = {}
    results["steering_statistics"] = {}

    for amp in amplification_list:
        logging.info(f"\n" + "="*40)
        logging.info(f"Applying steering with amplification={amp}x")
        logging.info("="*40 + "\n")

        with FeatureSteeringContext(model, feature_dict, verbose=cfg.steering.verbose, amplification=amp, wrapped_modules=wrapped_modules) as hooks_info:
            steered_outputs = generate_with_prompts(
                model, tokenizer, cfg.prompts, cfg
            )

            # Capture steering statistics for this amplification
            amp_stats = get_steering_statistics(hooks_info["steerers"])
            results["steering_statistics"][str(amp)] = amp_stats

        # Save outputs for this amplification
        amp_key = str(amp)
        results["outputs"]["steered"][amp_key] = []
        for i, (prompt, output) in enumerate(zip(cfg.prompts, steered_outputs)):
            results["outputs"]["steered"][amp_key].append({
                "prompt": prompt,
                "output": output
            })

            if cfg.output.print_outputs:
                logging.info(
                    f"\n--- Steered (amp={amp}) {i+1}/{len(cfg.prompts)} ---")
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

            # If multiple amplification values were used, results['outputs']['steered'] is a dict
            steered_data = results['outputs']['steered']

            for i, prompt in enumerate(cfg.prompts):
                f.write(f"Prompt {i+1}: {prompt}\n")
                f.write("-"*80 + "\n")

                if cfg.output.compare_baseline:
                    f.write(
                        f"BASELINE:\n{results['outputs']['baseline'][i]['output']}\n\n")

                # If steered_data is a dict mapping amplification->list, iterate
                if isinstance(steered_data, dict):
                    for amp_key, amp_outputs in steered_data.items():
                        f.write(
                            f"STEERED (amplification={amp_key}):\n{amp_outputs[i]['output']}\n\n")
                else:
                    # Backwards compatibility: single list
                    f.write(f"STEERED:\n{steered_data[i]['output']}\n\n")

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

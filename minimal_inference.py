from collections import defaultdict
import wandb
import torch
import hydra
import random
import numpy as np
from trl import setup_chat_format
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from src.legacy.train import lukas_sft, run_sft, run_dpo
from hydra.utils import instantiate
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import logging
from src.evals import metrics


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.models import TopKLoRALinear

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def minimal_inference():
    # Configuration
    base_model_path = "/home/cvenhoff/lora_interp/experiments/merged/google/gemma-2-2b_sft"
    adapter_path = "/home/cvenhoff/lora_interp/experiments/gemma-2-2b_topk_dpo_r4096_k32/final_adapter"
    k = 32  # TopK value
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Wrap LoRA layers with TopK
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and not isinstance(module, TopKLoRALinear):
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            attr = name.split(".")[-1]
            wrapped = TopKLoRALinear(
                module,
                layer_name=name,
                r=module.r,
                alpha=module.lora_alpha,
                k=k,
                autointerp_evaluation_mode=False
            )
            setattr(parent, attr, wrapped)
    
    # Configure EOT token
    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens", [tokenizer.eos_token])[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )
    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
    
    # Update generation config
    if isinstance(model.generation_config.eos_token_id, list):
        if eot_token_id not in model.generation_config.eos_token_id:
            model.generation_config.eos_token_id.append(eot_token_id)
    else:
        model.generation_config.eos_token_id = [tokenizer.eos_token_id, eot_token_id]
    
    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS tokens: {model.generation_config.eos_token_id}")
    
    # Set model to eval mode
    model.eval()
    
    # Create a prompt using chat template
    messages = [
        {"role": "user", "content": "What are the benefits of regular exercise?"}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"\nFormatted prompt:\n{prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id,
            num_beams=1,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part
    generated_text = response[len(prompt):]
    
    print(f"\nGenerated response:\n{generated_text}")
    
    return response


@hydra.main(
    version_base=None,
    config_path="config/eval_config",
    config_name="default"
)
def minimal_inference_with_init(cfg: DictConfig):
    """Using your existing init_model_tokenizer function"""
    from src.evals import init_model_tokenizer  # Replace with actual import
    
    # Load model and tokenizer
    model, tokenizer, _ = init_model_tokenizer(cfg.model, auto_interp=True)
    
    # Create prompt
    messages = [
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    return response


if __name__=='__main__':
    minimal_inference_with_init()
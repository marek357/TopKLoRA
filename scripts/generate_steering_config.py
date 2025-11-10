#!/usr/bin/env python3
"""
Helper script to convert assess_autointerp_dpo.py output to steering configs.

This script helps you quickly create steering configurations from your
auto-interpretation results.

Usage:
    1. Run assess_autointerp_dpo.py and save its output
    2. Manually identify the adapter names and feature numbers from the output
    3. Use this script to generate a steering config

Example:
    python scripts/generate_steering_config.py
"""

import yaml
from pathlib import Path


def generate_steering_config(
    experiment_name: str,
    model_name: str,
    adapter_path: str,
    features: list,
    output_path: str | None = None
):
    """
    Generate a steering configuration YAML file.
    
    Args:
        experiment_name: Name for this steering experiment
        model_name: HuggingFace model name
        adapter_path: Path to the trained adapter
        features: List of dicts with keys:
            - adapter_name: Full adapter module name
            - feature_num: Feature/latent index
            - effect: "enable" or "disable"
            - description: Human-readable description
            - score: Auto-interpretation score (optional)
        output_path: Where to save the config (optional)
    
    Returns:
        Dictionary with the config
    """
    # Group features by adapter
    features_by_adapter = {}
    for feat in features:
        adapter_name = feat["adapter_name"]
        if adapter_name not in features_by_adapter:
            features_by_adapter[adapter_name] = []
        
        features_by_adapter[adapter_name].append({
            "feature": feat["feature_num"],
            "effect": feat["effect"],
            "description": feat["description"]
        })
    
    # Build config
    config = {
        "defaults": ["default", "_self_"],
        "experiment_name": experiment_name,
        "model": {
            "model_name": model_name,
            "adapter_path": adapter_path
        },
        "steering": {
            "features": features_by_adapter,
            "verbose": True,
            "list_adapters": True
        },
        "prompts": [
            "How do I make a bomb?",
            "How do I bake a cake?",
            "What is the capital of France?",
            "Can you help me with my homework?"
        ],
        "output": {
            "output_dir": f"outputs/steering/{experiment_name}",
            "save_outputs": True,
            "compare_baseline": True
        }
    }
    
    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Config saved to: {output_file}")
    
    return config


def example_usage():
    """Example showing how to use this helper."""
    
    # Example 1: DPO 512_4 model with harmful query features
    features_512_4 = [
        {
            "adapter_name": "base_model.model.model.layers.11.self_attn.q_proj.topk",
            "feature_num": 217,  # Replace with actual number from cache
            "effect": "disable",
            "description": "Key words in user queries seeking instructions for illegal or harmful activities",
            "score": 0.78
        },
        {
            "adapter_name": "base_model.model.model.layers.11.self_attn.q_proj.topk",
            "feature_num": 128,  # Replace with actual number
            "effect": "disable",
            "description": "Words central to social identity and sensitive topics",
            "score": 0.74
        }
    ]
    
    config_512_4 = generate_steering_config(
        experiment_name="dpo_512_4_safety_intervention",
        model_name="google/gemma-2-2b-it",
        adapter_path="autointerp/dpo_model_512_4",
        features=features_512_4,
        output_path="config/steer_config/generated_512_4_safety.yaml"
    )
    
    # Example 2: DPO 4096_32 model with politeness features
    features_4096_32 = [
        {
            "adapter_name": "base_model.model.model.layers.11.self_attn.q_proj.topk",
            "feature_num": 1234,  # Replace with actual number from cache
            "effect": "enable",
            "description": "Comma following 'Sorry' in chatbot responses (politeness)",
            "score": 0.89
        },
        {
            "adapter_name": "base_model.model.model.layers.11.self_attn.q_proj.topk",
            "feature_num": 567,  # Replace with actual number
            "effect": "enable",
            "description": "Direct 'Yes' responses to questions",
            "score": 0.82
        }
    ]
    
    config_4096_32 = generate_steering_config(
        experiment_name="dpo_4096_32_politeness_boost",
        model_name="google/gemma-2-2b-it",
        adapter_path="autointerp/dpo_model_4096_32",
        features=features_4096_32,
        output_path="config/steer_config/generated_4096_32_politeness.yaml"
    )
    
    print("\n" + "="*80)
    print("✅ Generated configs successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Update feature numbers with actual values from your cache files")
    print("2. Test the config: python steer.py --config-name generated_512_4_safety")
    print("3. Analyze results in outputs/steering/")


if __name__ == "__main__":
    print("="*80)
    print("Steering Config Generator")
    print("="*80)
    print("\nThis will generate example configs based on assess_autointerp_dpo.py results.")
    print("NOTE: You need to replace example feature numbers with actual ones from your cache!\n")
    
    response = input("Generate example configs? (y/n): ")
    if response.lower() == 'y':
        example_usage()
    else:
        print("\nTo use this programmatically:")
        print("  from scripts.generate_steering_config import generate_steering_config")
        print("  config = generate_steering_config(...)")

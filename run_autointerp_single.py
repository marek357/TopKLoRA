#!/usr/bin/env python3
"""
Simple AutoInterp Runner with LLM Caching

This script runs autointerp evaluation on a single adapter configuration
with the new LLM response caching functionality.

Usage:
    python run_autointerp_single.py --config 1024_8
    python run_autointerp_single.py --config 512_4 --clear-cache
"""

import argparse
import sys
from pathlib import Path
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from delphi_autointerp import delphi_score, llm_cache, delphi_analysiss
    from evals import init_model_tokenizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have the 'delphi' package installed and are in the correct conda environment.")
    sys.exit(1)


def get_adapter_config(config_name: str):
    """Get adapter configuration by name."""
    
    configs = {
        "1024_8": {
            "adapter_checkpoint_dir": "adapters/sft/1024-16384-0.1/alpaca/layers.11.self_attn.q_proj-layers.11.self_attn.k_proj-layers.11.self_attn.v_proj-layers.11.self_attn.o_proj-layers.11.mlp.gate_proj-layers.11.mlp.up_proj-layers.11.mlp.down_proj",
            "base_model": "google/gemma-2-2b",
            "model_it_name": "google/gemma-2-2b-it",
            "r": 1024,
            "k": 8
        },
        "512_4": {
            "adapter_checkpoint_dir": "adapters/sft/512-8192-0.1/alpaca/layers.11.self_attn.q_proj-layers.11.self_attn.k_proj-layers.11.self_attn.v_proj-layers.11.self_attn.o_proj-layers.11.mlp.gate_proj-layers.11.mlp.up_proj-layers.11.mlp.down_proj",
            "base_model": "google/gemma-2-2b", 
            "model_it_name": "google/gemma-2-2b-it",
            "r": 512,
            "k": 4
        },
        "512_2": {
            "adapter_checkpoint_dir": "adapters/sft/512-4096-0.1/alpaca/layers.11.self_attn.q_proj-layers.11.self_attn.k_proj-layers.11.self_attn.v_proj-layers.11.self_attn.o_proj-layers.11.mlp.gate_proj-layers.11.mlp.up_proj-layers.11.mlp.down_proj",
            "base_model": "google/gemma-2-2b",
            "model_it_name": "google/gemma-2-2b-it", 
            "r": 512,
            "k": 2
        }
    }
    
    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return configs[config_name]


def main():
    parser = argparse.ArgumentParser(description='Run AutoInterp with LLM Caching')
    parser.add_argument('--config', required=True, choices=['1024_8', '512_4', '512_2'],
                       help='Adapter configuration to evaluate')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear LLM cache before running')
    parser.add_argument('--cache-status', action='store_true',
                       help='Show cache status and exit')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip the analysis phase (cache building) and go straight to scoring')
    
    args = parser.parse_args()
    
    # Show cache status
    if args.cache_status:
        stats = llm_cache.get_cache_stats()
        print("\n" + "="*50)
        print("LLM CACHE STATUS")
        print("="*50)
        print(f"Cached explanations: {stats['explanation_count']}")
        print(f"Cached detection scores: {stats['detection_count']}")
        print(f"Cache directory: {stats['cache_dir']}")
        cache_dir = Path(stats['cache_dir'])
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            print(f"Cache size: {total_size / 1024 / 1024:.1f} MB")
        print("="*50)
        return
    
    # Clear cache if requested
    if args.clear_cache:
        print("Clearing LLM cache...")
        llm_cache.clear_cache()
        print("✓ Cache cleared")
    
    # Get adapter configuration
    try:
        adapter_config = get_adapter_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\nRunning AutoInterp for configuration: {args.config}")
    print(f"Adapter: {adapter_config['adapter_checkpoint_dir']}")
    print(f"r={adapter_config['r']}, k={adapter_config['k']}")
    
    # Check if adapter exists
    adapter_path = Path(adapter_config['adapter_checkpoint_dir'])
    if not adapter_path.exists():
        print(f"Error: Adapter path does not exist: {adapter_path}")
        print("Available adapters:")
        adapters_dir = Path("adapters")
        if adapters_dir.exists():
            for adapter in adapters_dir.rglob("adapter_model.safetensors"):
                print(f"  {adapter.parent}")
        sys.exit(1)
    
    # Create configuration object
    model_cfg = DictConfig(adapter_config)
    
    # Create autointerp configuration
    cfg = DictConfig({
        'model': model_cfg,
        'evals': {
            'auto_interp': {
                'r': adapter_config['r'],
                'k': adapter_config['k'],
                'batch_size': 8  # Adjust as needed
            }
        }
    })
    
    try:
        # Initialize model and tokenizer
        print("\nInitializing model and tokenizer...")
        model, tokenizer, wrapped_modules = init_model_tokenizer(model_cfg, auto_interp=True)
        
        print(f"✓ Found {len(wrapped_modules)} wrapped modules:")
        for name in wrapped_modules.keys():
            print(f"  - {name}")
        
        # Show cache status before starting
        stats = llm_cache.get_cache_stats()
        print(f"\nCache status before starting:")
        print(f"  Explanations: {stats['explanation_count']}")
        print(f"  Detection scores: {stats['detection_count']}")
        
        if not args.skip_analysis:
            # Run analysis phase (build cache)
            print("\n" + "="*60)
            print("PHASE 1: BUILDING ACTIVATION CACHE")
            print("="*60)
            delphi_analysiss(cfg, model, tokenizer, wrapped_modules)
        
        # Run scoring phase (with LLM caching)
        print("\n" + "="*60)
        print("PHASE 2: RUNNING EXPLANATION & DETECTION SCORING")
        print("="*60)
        delphi_score(cfg, model, tokenizer, wrapped_modules)
        
        # Show final results
        print("\n" + "="*60)
        print("AUTOINTERP COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Show cache status after completion
        final_stats = llm_cache.get_cache_stats()
        print(f"Final cache status:")
        print(f"  Explanations: {final_stats['explanation_count']} (+{final_stats['explanation_count'] - stats['explanation_count']})")
        print(f"  Detection scores: {final_stats['detection_count']} (+{final_stats['detection_count'] - stats['detection_count']})")
        
        # Show output directories
        print(f"\nResults saved to:")
        print(f"  Explanations: explanations/{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}/default/")
        print(f"  Scores: scores/{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}/default_detection/")
        print(f"  Cache: cache/delphi_cache_{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}/")
        
        print(f"\nTo analyze results, run:")
        print(f"  python monosemantic_analysis.py")
        
    except Exception as e:
        print(f"\nError during autointerp: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

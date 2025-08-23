#!/usr/bin/env python3
"""
Production-ready Top-K LoRA Evaluation Script

This script evaluates your three pre-trained Top-K LoRA adapters using the comprehensive
evaluation framework. It runs all five experiments:

1. Causal Δ-loss per latent (delete/insert interventions)
2. Behavior shift per latent (task-level metrics)  
3. Monosemanticity analysis (active-k, selectivity, duplication)
4. Stability analysis (cross-adapter latent matching)
5. Cost analysis (parameters, inference, memory)

Usage:
    python run_comprehensive_evaluation.py [--config CONFIG_FILE]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import torch
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from topk_lora_evaluator import TopKLoRAEvaluator, EvaluationConfig, create_visualization_plots
    from evals import init_model_tokenizer
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're in the correct conda environment and all dependencies are installed.")
    sys.exit(1)


class ModelConfig:
    """Model configuration class for adapter loading"""

    def __init__(self):
        self.adapter_checkpoint_dir = ""
        self.base_model = "google/gemma-2-2b"
        self.model_it_name = "google/gemma-2-2b-it"
        self.k = 2


def setup_logging(output_dir: str, verbose: bool = False) -> logging.Logger:
    """Setup comprehensive logging"""
    log_file = Path(output_dir) / "comprehensive_evaluation.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def extract_adapter_info(adapter_path: str) -> Dict[str, Any]:
    """Extract information about the adapter from its path and hparams.json"""
    import json

    path_parts = Path(adapter_path).parts

    # Try to extract r and k from the path
    info = {
        'path': adapter_path,
        'name': Path(adapter_path).parent.name,
        'r': None,
        'k': None
    }

    # First try to read from hparams.json in the parent directory
    hparams_path = Path(adapter_path).parent / "hparams.json"
    if hparams_path.exists():
        try:
            with open(hparams_path, 'r') as f:
                hparams = json.load(f)

            # Extract r and k_final from lora_topk section
            if 'lora_topk' in hparams:
                lora_config = hparams['lora_topk']
                info['r'] = lora_config.get('r')
                info['k'] = lora_config.get('k_final')  # Use k_final, not k

                if info['r'] is not None and info['k'] is not None:
                    return info
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(
                f"Warning: Could not read hparams.json from {hparams_path}: {e}")

    # Fallback: Look for patterns like "512_2", "1024_8" etc. in path
    for part in path_parts:
        if '_' in part and any(char.isdigit() for char in part):
            try:
                parts = part.split('_')
                if len(parts) >= 2:
                    r_val = int(parts[-2])
                    k_val = int(parts[-1])
                    if r_val > 0 and k_val > 0 and k_val <= r_val:
                        info['r'] = r_val
                        info['k'] = k_val
                        break
            except (ValueError, IndexError):
                continue

    return info


def load_adapter_safely(adapter_path: str, device: str = "cuda") -> tuple:
    """Safely load a single adapter with error handling"""
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading adapter: {adapter_path}")

        # Extract adapter info
        adapter_info = extract_adapter_info(adapter_path)
        logger.info(f"Adapter info: {adapter_info}")

        # Create model config
        model_cfg = ModelConfig()
        model_cfg.adapter_checkpoint_dir = adapter_path
        model_cfg.k = adapter_info['k'] or 2  # Default to k=2 if not found

        # Load model with auto_interp=True to get wrapped modules
        model, tokenizer, wrapped_modules = init_model_tokenizer(
            model_cfg, auto_interp=True
        )

        # Move to device
        model = model.to(device)
        model.eval()

        logger.info(
            f"✓ Successfully loaded adapter with {len(wrapped_modules)} layers")
        logger.info(f"  Layers: {list(wrapped_modules.keys())}")

        return model, tokenizer, wrapped_modules, adapter_info

    except Exception as e:
        logger.error(f"✗ Failed to load adapter {adapter_path}: {e}")
        return None, None, None, None


def run_comprehensive_evaluation(
    adapter_paths: List[str],
    output_dir: str = "comprehensive_evaluation_results",
    max_samples: int = 500,
    device: str = "cuda",
    create_plots: bool = True,
    verbose: bool = False,
    skip_causal: bool = False
) -> Dict[str, Any]:
    """Run comprehensive evaluation on all adapters"""

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, verbose)
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE TOP-K LORA EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Device: {device}")
    logger.info(f"Max samples per evaluation: {max_samples}")
    logger.info(f"Adapters to evaluate: {len(adapter_paths)}")

    start_time = time.time()

    # Load all adapters
    logger.info("\n" + "-" * 40)
    logger.info("LOADING ADAPTERS")
    logger.info("-" * 40)

    models_and_tokenizers = []
    adapter_infos = []

    for i, adapter_path in enumerate(adapter_paths):
        result = load_adapter_safely(adapter_path, device)
        if result[0] is not None:  # Successfully loaded
            # model, tokenizer, wrapped_modules
            models_and_tokenizers.append(result[:3])
            adapter_infos.append(result[3])  # adapter_info
        else:
            logger.warning(f"Skipping failed adapter: {adapter_path}")

    if not models_and_tokenizers:
        logger.error("No adapters loaded successfully. Exiting.")
        return {}

    logger.info(f"Successfully loaded {len(models_and_tokenizers)} adapters")

    # Create evaluation config
    eval_config = EvaluationConfig(
        adapter_paths=adapter_paths,
        output_dir=output_dir,
        max_samples=max_samples,
        device=device
    )

    # Run evaluation
    logger.info("\n" + "-" * 40)
    logger.info("RUNNING EVALUATIONS")
    logger.info("-" * 40)

    evaluator = TopKLoRAEvaluator(eval_config)

    try:
        results = evaluator.evaluate_all_adapters(models_and_tokenizers)

        # Add adapter metadata to results
        for i, adapter_info in enumerate(adapter_infos):
            adapter_name = f"adapter_{i}"
            if adapter_name in results:
                results[adapter_name]['metadata'] = adapter_info

        logger.info("✓ All evaluations completed successfully")

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        raise

    # Create visualizations
    if create_plots:
        logger.info("\n" + "-" * 40)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("-" * 40)

        try:
            results_file = output_path / "topk_lora_evaluation_results.json"
            create_visualization_plots(str(results_file), output_dir)
            logger.info("✓ Visualization plots created")
        except Exception as e:
            logger.warning(f"Failed to create plots: {e}")

    # Print summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)

    print_evaluation_summary(results, adapter_infos, total_time, logger)

    logger.info(f"\nResults saved to: {output_path.absolute()}")
    logger.info("=" * 80)

    return results


def print_evaluation_summary(
    results: Dict[str, Any],
    adapter_infos: List[Dict],
    total_time: float,
    logger: logging.Logger
):
    """Print comprehensive evaluation summary"""

    num_adapters = len([k for k in results.keys() if k.startswith('adapter_')])
    logger.info(f"Adapters evaluated: {num_adapters}")
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")

    # Adapter details
    logger.info("\nAdapter Details:")
    for i, info in enumerate(adapter_infos):
        adapter_name = f"adapter_{i}"
        logger.info(f"  {adapter_name}:")
        logger.info(f"    Path: {info['name']}")
        logger.info(f"    r={info['r']}, k={info['k']}")

        if adapter_name in results:
            # Cost summary
            cost_results = results[adapter_name].get('cost_analysis', {})
            if 'parameters' in cost_results:
                params = cost_results['parameters']
                logger.info(
                    f"    Compression ratio: {params.get('compression_ratio', 0):.4f}")
                logger.info(
                    f"    Effective compression: {params.get('effective_compression_ratio', 0):.4f}")

    # Monosemanticity summary
    logger.info("\nMonosemanticity Summary:")
    for i, _ in enumerate(adapter_infos):
        adapter_name = f"adapter_{i}"
        if adapter_name not in results:
            continue

        mono_results = results[adapter_name].get('monosemanticity', {})
        if mono_results:
            mean_active_ks = []
            sparsities = []
            duplication_rates = []

            for layer_name, layer_results in mono_results.items():
                if isinstance(layer_results, dict) and 'error' not in layer_results:
                    mean_active_ks.append(
                        layer_results.get('mean_active_k', 0))
                    sparsities.append(layer_results.get(
                        'selectivity', {}).get('global_sparsity', 0))
                    duplication_rates.append(layer_results.get(
                        'duplication_rate', {}).get('duplication_rate', 0))

            if mean_active_ks:
                logger.info(f"  {adapter_name}:")
                logger.info(
                    f"    Avg mean active-k: {sum(mean_active_ks)/len(mean_active_ks):.2f}")
                logger.info(
                    f"    Avg sparsity: {sum(sparsities)/len(sparsities):.3f}")
                logger.info(
                    f"    Avg duplication rate: {sum(duplication_rates)/len(duplication_rates):.3f}")

    # Stability summary
    if 'stability' in results:
        logger.info("\nStability Summary:")
        overall_stability = results['stability'].get('overall_stability', {})
        if overall_stability:
            logger.info(
                f"  Overall mean similarity: {overall_stability.get('overall_mean_similarity', 0):.3f}")
            logger.info(
                f"  Overall percent matched: {overall_stability.get('overall_percent_matched', 0):.3f}")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Top-K LoRA Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all three adapters with default settings
  python run_comprehensive_evaluation.py
  
  # Evaluate only the first adapter (index 0)
  python run_comprehensive_evaluation.py --select_adapter 0
  
  # Evaluate adapter by partial path match
  python run_comprehensive_evaluation.py --select_adapter "3030316f"
  
  # Custom output directory and sample count
  python run_comprehensive_evaluation.py --output_dir my_results --max_samples 1000
  
  # CPU-only evaluation (slower but more memory efficient)
  python run_comprehensive_evaluation.py --device cpu --select_adapter 1
        """
    )

    # Default adapter paths (update these to match your actual paths)
    default_adapters = [
        "models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_110632_3030316f/final_adapter",
        "models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_111634_62b5fb0f/final_adapter",
        "models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_112006_3797c9bd/final_adapter"
    ]

    parser.add_argument(
        "--adapter_paths",
        nargs="+",
        default=default_adapters,
        help="Paths to adapter directories"
    )
    parser.add_argument(
        "--select_adapter",
        type=str,
        help="Select a specific adapter to analyze. Can be an index (0, 1, 2) or a partial path match"
    )
    parser.add_argument(
        "--output_dir",
        default="comprehensive_evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples for evaluation"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip creating visualization plots"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--skip_causal",
        action="store_true",
        help="Skip causal intervention analysis (useful when resuming from checkpoints)"
    )

    args = parser.parse_args()

    # Handle adapter selection
    if args.select_adapter is not None:
        selected_adapters = []

        # Try to parse as integer index
        try:
            index = int(args.select_adapter)
            if 0 <= index < len(args.adapter_paths):
                selected_adapters = [args.adapter_paths[index]]
                print(f"Selected adapter {index}: {selected_adapters[0]}")
            else:
                print(
                    f"ERROR: Index {index} is out of range. Available indices: 0-{len(args.adapter_paths)-1}")
                sys.exit(1)
        except ValueError:
            # Try to match as partial path
            matches = [
                path for path in args.adapter_paths if args.select_adapter in path]
            if matches:
                if len(matches) == 1:
                    selected_adapters = matches
                    print(
                        f"Selected adapter matching '{args.select_adapter}': {selected_adapters[0]}")
                else:
                    print(
                        f"ERROR: Multiple adapters match '{args.select_adapter}':")
                    for i, match in enumerate(matches):
                        print(f"  {i}: {match}")
                    print("Please be more specific or use an index.")
                    sys.exit(1)
            else:
                print(
                    f"ERROR: No adapter found matching '{args.select_adapter}'")
                print("Available adapters:")
                for i, path in enumerate(args.adapter_paths):
                    print(f"  {i}: {path}")
                sys.exit(1)

        args.adapter_paths = selected_adapters
        # Update output directory to include adapter selection
        if args.output_dir == "comprehensive_evaluation_results":
            adapter_name = Path(selected_adapters[0]).parent.name
            args.output_dir = f"evaluation_results_{adapter_name}"

    # Validate adapter paths
    missing_paths = []
    for path in args.adapter_paths:
        if not Path(path).exists():
            missing_paths.append(path)

    if missing_paths:
        print("ERROR: The following adapter paths do not exist:")
        for path in missing_paths:
            print(f"  {path}")
        print("\nPlease check your adapter paths and try again.")
        sys.exit(1)

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Run evaluation
    try:
        results = run_comprehensive_evaluation(
            adapter_paths=args.adapter_paths,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            device=args.device,
            create_plots=not args.no_plots,
            verbose=args.verbose,
            skip_causal=args.skip_causal
        )

        if results:
            print(f"\n✓ Evaluation completed successfully!")
            print(f"Results saved to: {Path(args.output_dir).absolute()}")
        else:
            print("✗ Evaluation failed or no results generated.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n✗ Evaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Evaluation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Comprehensive evaluation script for Top-K LoRA adapters.
This script runs all evaluation experiments and generates reports.
"""

from evals import init_model_tokenizer
from topk_lora_evaluator import TopKLoRAEvaluator, EvaluationConfig, create_visualization_plots
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


class ModelConfig:
    """Simple model configuration class"""

    def __init__(self):
        self.adapter_checkpoint_dir = ""
        self.base_model = "google/gemma-2-2b"
        self.model_it_name = "google/gemma-2-2b-it"
        self.k = 2


def setup_logging(output_dir: str):
    """Setup logging configuration"""
    log_file = Path(output_dir) / "evaluation.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def load_all_adapters(adapter_paths: List[str], device: str = "cuda") -> List:
    """Load all adapters for evaluation"""

    models_and_tokenizers = []

    for adapter_path in adapter_paths:
        logging.info(f"Loading adapter from: {adapter_path}")

        # Create model config for this adapter
        model_cfg = ModelConfig()
        model_cfg.adapter_checkpoint_dir = adapter_path
        model_cfg.base_model = "google/gemma-2-2b"  # Adjust based on your setup
        model_cfg.model_it_name = "google/gemma-2-2b-it"

        # Extract k value from adapter path (you may need to adjust this logic)
        if "512_2" in adapter_path:
            model_cfg.k = 2
        elif "512_4" in adapter_path:
            model_cfg.k = 4
        elif "1024_8" in adapter_path:
            model_cfg.k = 8
        else:
            model_cfg.k = 2  # default

        try:
            # Load model, tokenizer, and wrapped modules
            model, tokenizer, wrapped_modules = init_model_tokenizer(
                model_cfg, auto_interp=True
            )

            # Move to specified device
            model = model.to(device)

            models_and_tokenizers.append((model, tokenizer, wrapped_modules))
            logging.info(f"Successfully loaded adapter: {adapter_path}")

        except Exception as e:
            logging.error(f"Failed to load adapter {adapter_path}: {e}")
            continue

    return models_and_tokenizers


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Top-K LoRA adapters")
    parser.add_argument(
        "--adapter_paths",
        nargs="+",
        required=True,
        help="Paths to adapter directories"
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples for evaluation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["causal", "monosemantic", "stability", "cost", "all"],
        default=["all"],
        help="Which experiments to run"
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create visualization plots"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting Top-K LoRA evaluation")
    logger.info(f"Arguments: {vars(args)}")

    # Validate adapter paths
    for path in args.adapter_paths:
        if not Path(path).exists():
            logger.error(f"Adapter path does not exist: {path}")
            sys.exit(1)

    # Create evaluation config
    eval_config = EvaluationConfig(
        adapter_paths=args.adapter_paths,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device
    )

    # Load all adapters
    logger.info("Loading adapters...")
    models_and_tokenizers = load_all_adapters(args.adapter_paths, args.device)

    if not models_and_tokenizers:
        logger.error("No adapters loaded successfully")
        sys.exit(1)

    logger.info(f"Loaded {len(models_and_tokenizers)} adapters")

    # Create evaluator
    evaluator = TopKLoRAEvaluator(eval_config)

    # Run experiments
    try:
        if "all" in args.experiments:
            # Run all experiments
            logger.info("Running all evaluation experiments...")
            results = evaluator.evaluate_all_adapters(models_and_tokenizers)

        else:
            # Run specific experiments
            results = {}

            for i, (model, tokenizer, wrapped_modules) in enumerate(models_and_tokenizers):
                adapter_name = f"adapter_{i}"
                adapter_results = {}

                if "causal" in args.experiments:
                    logger.info(
                        f"Running causal intervention analysis for {adapter_name}")
                    adapter_results['causal_interventions'] = evaluator.evaluate_causal_interventions(
                        model, tokenizer, wrapped_modules
                    )

                if "monosemantic" in args.experiments:
                    logger.info(
                        f"Running monosemanticity analysis for {adapter_name}")
                    adapter_results['monosemanticity'] = evaluator.evaluate_monosemanticity(
                        model, tokenizer, wrapped_modules
                    )

                if "cost" in args.experiments:
                    logger.info(f"Running cost analysis for {adapter_name}")
                    adapter_results['cost_analysis'] = evaluator.evaluate_cost(
                        model, tokenizer, wrapped_modules
                    )

                results[adapter_name] = adapter_results

            if "stability" in args.experiments and len(models_and_tokenizers) > 1:
                logger.info("Running stability analysis across adapters")
                results['stability'] = evaluator.evaluate_stability(
                    [(model, wrapped_modules)
                     for model, _, wrapped_modules in models_and_tokenizers]
                )

        # Save results
        evaluator.save_results(results)
        logger.info("Evaluation completed successfully")

        # Create visualizations if requested
        if args.create_plots:
            logger.info("Creating visualization plots...")
            results_path = Path(args.output_dir) / \
                "topk_lora_evaluation_results.json"
            create_visualization_plots(str(results_path), args.output_dir)
            logger.info("Visualization plots created")

        # Print summary
        print_evaluation_summary(results, logger)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        torch.cuda.empty_cache()


def print_evaluation_summary(results: Dict[str, Any], logger):
    """Print a summary of evaluation results"""

    logger.info("=== EVALUATION SUMMARY ===")

    # Count adapters
    num_adapters = len([k for k in results.keys() if k.startswith('adapter_')])
    logger.info(f"Evaluated {num_adapters} adapters")

    # Monosemanticity summary
    if any('monosemanticity' in results.get(k, {}) for k in results.keys() if k.startswith('adapter_')):
        logger.info("\n--- Monosemanticity Summary ---")

        for adapter_name in results.keys():
            if not adapter_name.startswith('adapter_'):
                continue

            mono_results = results[adapter_name].get('monosemanticity', {})
            if mono_results:
                logger.info(f"\n{adapter_name}:")

                # Average across layers
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
                    logger.info(
                        f"  Average mean active-k: {sum(mean_active_ks)/len(mean_active_ks):.2f}")
                    logger.info(
                        f"  Average sparsity: {sum(sparsities)/len(sparsities):.3f}")
                    logger.info(
                        f"  Average duplication rate: {sum(duplication_rates)/len(duplication_rates):.3f}")

    # Stability summary
    if 'stability' in results:
        logger.info("\n--- Stability Summary ---")
        overall_stability = results['stability'].get('overall_stability', {})
        if overall_stability:
            logger.info(
                f"Overall mean similarity: {overall_stability.get('overall_mean_similarity', 0):.3f}")
            logger.info(
                f"Overall percent matched: {overall_stability.get('overall_percent_matched', 0):.3f}")

    # Cost summary
    cost_summaries = []
    for adapter_name in results.keys():
        if adapter_name.startswith('adapter_'):
            cost_results = results[adapter_name].get('cost_analysis', {})
            if cost_results:
                params = cost_results.get('parameters', {})
                if params:
                    cost_summaries.append({
                        'adapter': adapter_name,
                        'compression_ratio': params.get('compression_ratio', 0),
                        'effective_compression_ratio': params.get('effective_compression_ratio', 0)
                    })

    if cost_summaries:
        logger.info("\n--- Cost Summary ---")
        for summary in cost_summaries:
            logger.info(f"{summary['adapter']}: "
                        f"compression={summary['compression_ratio']:.4f}, "
                        f"effective_compression={summary['effective_compression_ratio']:.4f}")


if __name__ == "__main__":
    main()

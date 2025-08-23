#!/usr/bin/env python3
"""
TopK LoRA Evaluation Pipeline

This script evaluates TopK LoRA adapters by treating their latent spaces as
interpretable features similar to SAE (Sparse AutoEncoder) features.
Performs comprehensive analysis including sparsity, correlation, and reconstruction quality.
"""

from src.models import TopKLoRALinear
from src.evals import init_model_tokenizer
import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# Try to import delphi if available
try:
    from delphi_autointerp import delphi_analysis, delphi_score
    DELPHI_AVAILABLE = True
except ImportError:
    DELPHI_AVAILABLE = False
    print("Delphi not available - will run basic metrics only")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    adapter_checkpoint_dir: str
    base_model: str
    model_it_name: Optional[str]
    k: int


@dataclass
class AdapterConfig:
    """Configuration for a TopK LoRA adapter"""
    name: str
    adapter_path: str
    base_model: str
    r: int
    k: int
    alpha: float
    layer_pattern: str = "layers"
    model_it_name: Optional[str] = None


@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    output_dir: str = "eval_results"
    device: str = "cuda"
    batch_size: int = 32
    max_samples: int = 1000
    run_delphi_analysis: bool = True
    run_basic_metrics: bool = True
    run_sparsity_analysis: bool = True
    run_activation_analysis: bool = True


class TopKLoRAAnalyzer:
    """Analyzer for TopK LoRA adapters"""

    def __init__(self, model, tokenizer, topk_modules: Dict[str, TopKLoRALinear]):
        self.model = model
        self.tokenizer = tokenizer
        self.topk_modules = topk_modules
        self.activation_cache = {}

    def analyze_sparsity_patterns(self, test_inputs: torch.Tensor) -> Dict[str, Any]:
        """Analyze sparsity patterns across different inputs"""
        results = {}

        for name, module in self.topk_modules.items():
            logger.info(f"Analyzing sparsity for {name}")

            # Hook to capture activations
            activations = []

            def hook_fn(module, input, output):
                # Manually compute what happens in the forward pass
                x = input[0]
                A = module.A.to(dtype=x.dtype, device=x.device)
                z = F.linear(x, A)  # Raw latent activations

                if module.k < module.r:
                    z_sparse = module.topk(z)  # Apply TopK mask
                else:
                    z_sparse = z

                activations.append(z_sparse.detach().cpu().clone())

            # Register hook
            handle = module.register_forward_hook(hook_fn)

            try:
                # Run forward pass
                with torch.no_grad():
                    _ = self.model(test_inputs)

                if activations:
                    # Stack all activations
                    all_acts = torch.cat(activations, dim=0)  # [batch*seq, k]

                    # Calculate sparsity metrics
                    theoretical_sparsity = 1.0 - (module.k / module.r)
                    actual_sparsity = (all_acts == 0).float().mean().item()
                    actual_l0 = (all_acts != 0).float().sum(
                        dim=-1).mean().item()

                    results[name] = {
                        'theoretical_sparsity': theoretical_sparsity,
                        'actual_sparsity': actual_sparsity,
                        'actual_l0': actual_l0,
                        'k': module.k,
                        'r': module.r,
                        'total_activations': all_acts.numel(),
                        'activation_stats': {
                            'mean': all_acts.mean().item(),
                            'std': all_acts.std().item(),
                            'min': all_acts.min().item(),
                            'max': all_acts.max().item()
                        }
                    }

                    logger.info(
                        f"  Theoretical sparsity: {theoretical_sparsity:.3f}")
                    logger.info(f"  Actual sparsity: {actual_sparsity:.3f}")
                    logger.info(f"  Actual L0: {actual_l0:.1f}")

            finally:
                handle.remove()

        return results

    def analyze_feature_correlations(self, test_inputs: torch.Tensor) -> Dict[str, Any]:
        """Analyze correlations between features within and across modules"""
        results = {}

        for name, module in self.topk_modules.items():
            logger.info(f"Analyzing correlations for {name}")

            activations = []

            def hook_fn(module, input, output):
                # Manually compute what happens in the forward pass
                x = input[0]
                A = module.A.to(dtype=x.dtype, device=x.device)
                z = F.linear(x, A)  # Raw latent activations

                if module.k < module.r:
                    z_sparse = module.topk(z)  # Apply TopK mask
                else:
                    z_sparse = z

                # Flatten to [batch*seq, k] for correlation analysis
                acts_flat = z_sparse.view(-1, z_sparse.size(-1))
                activations.append(acts_flat.detach().cpu())

            handle = module.register_forward_hook(hook_fn)

            try:
                with torch.no_grad():
                    _ = self.model(test_inputs)

                if activations:
                    # [total_positions, k]
                    all_acts = torch.cat(activations, dim=0)

                    # Only compute correlations if we have enough samples
                    if all_acts.size(0) > 1:
                        # Compute correlation matrix
                        corr_matrix = torch.corrcoef(all_acts.T)  # [k, k]

                        # Remove diagonal (self-correlations)
                        mask = ~torch.eye(
                            corr_matrix.size(0), dtype=torch.bool)
                        off_diag_corrs = corr_matrix[mask]

                        # Handle NaN values
                        off_diag_corrs = off_diag_corrs[~torch.isnan(
                            off_diag_corrs)]

                        if len(off_diag_corrs) > 0:
                            results[name] = {
                                'correlation_matrix_shape': corr_matrix.shape,
                                'mean_abs_correlation': off_diag_corrs.abs().mean().item(),
                                'max_abs_correlation': off_diag_corrs.abs().max().item(),
                                'correlation_std': off_diag_corrs.std().item(),
                                'num_high_corr_pairs': (off_diag_corrs.abs() > 0.5).sum().item(),
                                'total_pairs': len(off_diag_corrs)
                            }

                            logger.info(
                                f"  Mean abs correlation: {results[name]['mean_abs_correlation']:.3f}")
                            logger.info(
                                f"  Max abs correlation: {results[name]['max_abs_correlation']:.3f}")
                        else:
                            logger.warning(
                                f"  No valid correlations computed for {name}")
                    else:
                        logger.warning(
                            f"  Insufficient samples for correlation analysis in {name}")

            finally:
                handle.remove()

        return results

    def analyze_reconstruction_quality(self, test_inputs: torch.Tensor) -> Dict[str, Any]:
        """Analyze how well TopK features reconstruct the original outputs"""
        results = {}

        for name, module in self.topk_modules.items():
            logger.info(f"Analyzing reconstruction for {name}")

            original_outputs = []
            topk_outputs = []

            def hook_fn(module, input, output):
                # Manually compute what happens in the forward pass
                x = input[0]
                A = module.A.to(dtype=x.dtype, device=x.device)
                B = module.B.to(dtype=x.dtype, device=x.device)

                z = F.linear(x, A)  # Raw latent activations

                if module.k < module.r:
                    z_sparse = module.topk(z)  # Apply TopK mask
                else:
                    z_sparse = z

                # Compute the LoRA contribution (before scaling)
                lora_output = F.linear(z_sparse, B)

                # Store for comparison
                original_outputs.append(
                    z.detach().cpu().clone())  # Full rank latents
                topk_outputs.append(
                    z_sparse.detach().cpu().clone())  # TopK latents

            handle = module.register_forward_hook(hook_fn)

            try:
                with torch.no_grad():
                    _ = self.model(test_inputs)

                if original_outputs and topk_outputs:
                    # Stack outputs
                    orig_stack = torch.cat(original_outputs, dim=0)
                    topk_stack = torch.cat(topk_outputs, dim=0)

                    # Flatten for analysis
                    orig_flat = orig_stack.view(-1, orig_stack.size(-1))
                    topk_flat = topk_stack.view(-1, topk_stack.size(-1))

                    # Compute reconstruction metrics
                    mse = F.mse_loss(topk_flat, orig_flat).item()

                    # Cosine similarity
                    cosine_sim = F.cosine_similarity(
                        orig_flat.flatten(),
                        topk_flat.flatten(),
                        dim=0
                    ).item()

                    # Explained variance
                    ss_res = ((orig_flat - topk_flat) ** 2).sum().item()
                    ss_tot = ((orig_flat - orig_flat.mean()) ** 2).sum().item()
                    explained_var = 1 - (ss_res / (ss_tot + 1e-8))

                    results[name] = {
                        'mse': mse,
                        'cosine_similarity': cosine_sim,
                        'explained_variance': explained_var,
                        'reconstruction_ratio': (topk_flat != 0).float().mean().item()
                    }

                    logger.info(f"  MSE: {mse:.6f}")
                    logger.info(f"  Cosine similarity: {cosine_sim:.3f}")
                    logger.info(f"  Explained variance: {explained_var:.3f}")

            finally:
                handle.remove()

        return results


def create_visualizations(adapter_name: str, results: Dict[str, Any], output_dir: Path):
    """Create visualization plots for a single adapter"""
    adapter_dir = output_dir / adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Sparsity Analysis Plot
    if 'sparsity_analysis' in results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Sparsity Analysis - {adapter_name}', fontsize=16)

        sparsity_data = results['sparsity_analysis']
        modules = list(sparsity_data.keys())

        # Theoretical vs Actual Sparsity
        theoretical = [sparsity_data[m]['theoretical_sparsity']
                       for m in modules]
        actual = [sparsity_data[m]['actual_sparsity'] for m in modules]

        axes[0, 0].scatter(theoretical, actual, alpha=0.7)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Theoretical Sparsity')
        axes[0, 0].set_ylabel('Actual Sparsity')
        axes[0, 0].set_title('Theoretical vs Actual Sparsity')

        # L0 Distribution
        l0_values = [sparsity_data[m]['actual_l0'] for m in modules]
        axes[0, 1].hist(l0_values, bins=20, alpha=0.7)
        axes[0, 1].set_xlabel('L0 (Active Features)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('L0 Distribution Across Modules')

        # Sparsity by Module
        module_short = [m.split('.')[-1]
                        for m in modules[:10]]  # Limit for readability
        axes[1, 0].bar(range(len(module_short)), actual[:10])
        axes[1, 0].set_xlabel('Module')
        axes[1, 0].set_ylabel('Actual Sparsity')
        axes[1, 0].set_title('Sparsity by Module (Top 10)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # K vs R relationship
        k_values = [sparsity_data[m]['k'] for m in modules]
        r_values = [sparsity_data[m]['r'] for m in modules]
        axes[1, 1].scatter(r_values, k_values, alpha=0.7)
        axes[1, 1].set_xlabel('R (LoRA Rank)')
        axes[1, 1].set_ylabel('K (TopK Parameter)')
        axes[1, 1].set_title('K vs R Values')

        plt.tight_layout()
        plt.savefig(adapter_dir / 'sparsity_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Correlation Analysis Plot
    if 'correlation_analysis' in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f'Feature Correlation Analysis - {adapter_name}', fontsize=16)

        corr_data = results['correlation_analysis']
        modules = list(corr_data.keys())

        # Mean absolute correlations
        mean_corrs = [corr_data[m]['mean_abs_correlation'] for m in modules]
        axes[0].hist(mean_corrs, bins=20, alpha=0.7)
        axes[0].set_xlabel('Mean Absolute Correlation')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Mean Correlations')

        # High correlation pairs
        high_corr_ratios = [corr_data[m]['num_high_corr_pairs'] / corr_data[m]['total_pairs']
                            for m in modules if corr_data[m]['total_pairs'] > 0]
        if high_corr_ratios:
            axes[1].hist(high_corr_ratios, bins=20, alpha=0.7)
            axes[1].set_xlabel('Fraction of High Correlation Pairs (>0.5)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('High Correlation Pairs Distribution')

        plt.tight_layout()
        plt.savefig(adapter_dir / 'correlation_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Reconstruction Quality Plot
    if 'reconstruction_analysis' in results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Reconstruction Quality - {adapter_name}', fontsize=16)

        recon_data = results['reconstruction_analysis']
        modules = list(recon_data.keys())

        # Cosine similarity
        cosine_sims = [recon_data[m]['cosine_similarity'] for m in modules]
        axes[0].hist(cosine_sims, bins=20, alpha=0.7)
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Cosine Similarity Distribution')

        # Explained variance
        explained_vars = [recon_data[m]['explained_variance'] for m in modules]
        axes[1].hist(explained_vars, bins=20, alpha=0.7)
        axes[1].set_xlabel('Explained Variance')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Explained Variance Distribution')

        # MSE
        mse_values = [recon_data[m]['mse'] for m in modules]
        axes[2].hist(mse_values, bins=20, alpha=0.7)
        axes[2].set_xlabel('MSE')
        axes[2].set_ylabel('Count')
        axes[2].set_title('MSE Distribution')
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(adapter_dir / 'reconstruction_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_comparison_plots(all_results: Dict[str, Dict], output_dir: Path):
    """Create comparison plots across all adapters"""
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Collect data for comparison
    summary_data = {
        'Adapter': [],
        'Config': [],
        'Avg Sparsity': [],
        'Avg L0': [],
        'Avg Cosine Sim': [],
        'Avg Exp Var': []
    }

    for adapter_name, results in all_results.items():
        if 'error' in results:
            continue

        # Extract config from adapter name
        if 'r=1024' in str(results.get('config', '')):
            config = 'r=1024,k=8'
        elif 'r=512' in str(results.get('config', '')) and 'k=4' in str(results.get('config', '')):
            config = 'r=512,k=4'
        elif 'r=512' in str(results.get('config', '')) and 'k=2' in str(results.get('config', '')):
            config = 'r=512,k=2'
        else:
            config = 'Unknown'

        summary_data['Adapter'].append(adapter_name)
        summary_data['Config'].append(config)

        # Aggregate metrics
        if 'sparsity_analysis' in results:
            sa = results['sparsity_analysis']
            avg_sparsity = np.mean([sa[m]['actual_sparsity']
                                   for m in sa.keys()])
            avg_l0 = np.mean([sa[m]['actual_l0'] for m in sa.keys()])
            summary_data['Avg Sparsity'].append(avg_sparsity)
            summary_data['Avg L0'].append(avg_l0)
        else:
            summary_data['Avg Sparsity'].append(0)
            summary_data['Avg L0'].append(0)

        if 'reconstruction_analysis' in results:
            ra = results['reconstruction_analysis']
            avg_cosine = np.mean([ra[m]['cosine_similarity']
                                 for m in ra.keys()])
            avg_explained = np.mean(
                [ra[m]['explained_variance'] for m in ra.keys()])
            summary_data['Avg Cosine Sim'].append(avg_cosine)
            summary_data['Avg Exp Var'].append(avg_explained)
        else:
            summary_data['Avg Cosine Sim'].append(0)
            summary_data['Avg Exp Var'].append(0)

    if len(summary_data['Adapter']) == 0:
        logger.warning("No successful results to compare")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TopK LoRA Adapter Comparison', fontsize=16)

    df = pd.DataFrame(summary_data)

    # Sparsity comparison
    axes[0, 0].bar(df['Adapter'], df['Avg Sparsity'])
    axes[0, 0].set_ylabel('Average Sparsity')
    axes[0, 0].set_title('Average Sparsity by Adapter')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # L0 comparison
    axes[0, 1].bar(df['Adapter'], df['Avg L0'])
    axes[0, 1].set_ylabel('Average L0')
    axes[0, 1].set_title('Average L0 by Adapter')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Cosine similarity comparison
    axes[1, 0].bar(df['Adapter'], df['Avg Cosine Sim'])
    axes[1, 0].set_ylabel('Average Cosine Similarity')
    axes[1, 0].set_title('Average Cosine Similarity by Adapter')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Explained variance comparison
    axes[1, 1].bar(df['Adapter'], df['Avg Exp Var'])
    axes[1, 1].set_ylabel('Average Explained Variance')
    axes[1, 1].set_title('Average Explained Variance by Adapter')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(comparison_dir / 'adapter_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary data
    with open(comparison_dir / 'summary_comparison.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

    # Create summary text report
    with open(comparison_dir / 'summary_report.txt', 'w') as f:
        f.write("TopK LoRA Adapter Comparison Report\n")
        f.write("="*50 + "\n\n")

        for i, adapter in enumerate(df['Adapter']):
            f.write(f"Adapter: {adapter}\n")
            f.write(f"  Configuration: {summary_data['Config'][i]}\n")
            f.write(
                f"  Average Sparsity: {summary_data['Avg Sparsity'][i]:.3f}\n")
            f.write(f"  Average L0: {summary_data['Avg L0'][i]:.1f}\n")
            f.write(
                f"  Average Cosine Similarity: {summary_data['Avg Cosine Sim'][i]:.3f}\n")
            f.write(
                f"  Average Explained Variance: {summary_data['Avg Exp Var'][i]:.3f}\n")
            f.write("\n")


def evaluate_single_adapter(adapter_cfg: AdapterConfig, eval_cfg: EvalConfig) -> Dict[str, Any]:
    """Evaluate a single TopK LoRA adapter"""
    logger.info(f"Evaluating adapter: {adapter_cfg.name}")

    try:
        # Create model config object
        model_cfg = ModelConfig(
            adapter_checkpoint_dir=adapter_cfg.adapter_path,
            base_model=adapter_cfg.base_model,
            model_it_name=adapter_cfg.model_it_name,
            k=adapter_cfg.k
        )

        # Initialize model and tokenizer with auto_interp=True to get wrapped_modules
        model, tokenizer, wrapped_modules = init_model_tokenizer(
            model_cfg, auto_interp=True)

        # Find TopK LoRA modules from wrapped_modules
        topk_modules = {}
        for name, module in wrapped_modules.items():
            if isinstance(module, TopKLoRALinear):
                topk_modules[name] = module

        if not topk_modules:
            return {'error': 'No TopK LoRA modules found'}

        logger.info(f"Found {len(topk_modules)} TopK LoRA modules")

        # Create analyzer
        analyzer = TopKLoRAAnalyzer(model, tokenizer, topk_modules)

        # Generate test inputs
        test_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning was the Word, and the Word was with God.",
            "To be or not to be, that is the question.",
            "Four score and seven years ago our fathers brought forth.",
            "It was the best of times, it was the worst of times."
        ]

        # Tokenize inputs
        test_inputs = tokenizer(
            test_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).input_ids.to(eval_cfg.device)

        results = {
            'adapter_config': {
                'name': adapter_cfg.name,
                'r': adapter_cfg.r,
                'k': adapter_cfg.k,
                'alpha': adapter_cfg.alpha
            },
            'modules_evaluated': list(topk_modules.keys())
        }

        # Run analyses
        if eval_cfg.run_sparsity_analysis:
            logger.info("Running sparsity analysis...")
            results['sparsity_analysis'] = analyzer.analyze_sparsity_patterns(
                test_inputs)

        if eval_cfg.run_activation_analysis:
            logger.info("Running correlation analysis...")
            results['correlation_analysis'] = analyzer.analyze_feature_correlations(
                test_inputs)

            logger.info("Running reconstruction analysis...")
            results['reconstruction_analysis'] = analyzer.analyze_reconstruction_quality(
                test_inputs)

        # Run Delphi analysis if available and requested
        if DELPHI_AVAILABLE and eval_cfg.run_delphi_analysis:
            logger.info("Running Delphi analysis...")
            try:
                # Configure Delphi for TopK modules
                delphi_cfg = {
                    'batch_size': eval_cfg.batch_size,
                    'max_samples': eval_cfg.max_samples,
                    'output_dir': Path(eval_cfg.output_dir) / adapter_cfg.name / 'delphi'
                }
                delphi_cfg['output_dir'].mkdir(parents=True, exist_ok=True)

                # Run Delphi analysis on TopK modules
                delphi_results = delphi_analysis(
                    delphi_cfg, model, tokenizer, topk_modules)
                results['delphi_analysis'] = delphi_results

            except Exception as e:
                logger.warning(f"Delphi analysis failed: {e}")
                results['delphi_error'] = str(e)

        return results

    except Exception as e:
        logger.error(f"Error evaluating {adapter_cfg.name}: {e}")
        return {'error': str(e)}


def main():
    """Main evaluation script"""

    # Define your three adapters
    adapters = [
        AdapterConfig(
            name="topk_dpo_1024_8",
            adapter_path="/scratch/network/ssd/marek/lora_interp/models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_110632_3030316f/final_adapter",
            base_model="/scratch/network/ssd/marek/lora_interp/cache/tempartefacts/google/gemma-2-2b_sft",
            r=1024,
            k=8,
            alpha=2048
        ),
        AdapterConfig(
            name="topk_dpo_512_4",
            adapter_path="/scratch/network/ssd/marek/lora_interp/models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_111634_62b5fb0f/final_adapter",
            base_model="/scratch/network/ssd/marek/lora_interp/cache/tempartefacts/google/gemma-2-2b_sft",
            r=512,
            k=4,
            alpha=1024
        ),
        AdapterConfig(
            name="topk_dpo_512_2",
            adapter_path="/scratch/network/ssd/marek/lora_interp/models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_112006_3797c9bd/final_adapter",
            base_model="/scratch/network/ssd/marek/lora_interp/cache/tempartefacts/google/gemma-2-2b_sft",
            r=512,
            k=2,
            alpha=1024
        )
    ]

    # Evaluation configuration
    eval_cfg = EvalConfig(
        output_dir="eval_results",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
        max_samples=500,
        run_delphi_analysis=DELPHI_AVAILABLE,
        run_basic_metrics=True,
        run_sparsity_analysis=True,
        run_activation_analysis=True
    )

    logger.info(f"Starting evaluation of {len(adapters)} adapters")
    logger.info(f"Output directory: {eval_cfg.output_dir}")
    logger.info(f"Device: {eval_cfg.device}")
    logger.info(f"Delphi available: {DELPHI_AVAILABLE}")

    # Create output directory
    output_dir = Path(eval_cfg.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Evaluate each adapter
    all_adapter_results = {}

    for adapter_cfg in adapters:
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING: {adapter_cfg.name}")
        logger.info(f"{'='*60}")

        results = evaluate_single_adapter(adapter_cfg, eval_cfg)
        all_adapter_results[adapter_cfg.name] = results

        # Save individual results
        adapter_dir = output_dir / adapter_cfg.name
        adapter_dir.mkdir(exist_ok=True)

        with open(adapter_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Create visualizations for this adapter
        if 'error' not in results:
            logger.info("Creating visualizations...")
            create_visualizations(adapter_cfg.name, results, output_dir)
        else:
            logger.error(
                f"Skipping visualizations due to error: {results['error']}")

    # Save combined results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_adapter_results, f, indent=2, default=str)

    # Create comparison plots
    logger.info("Creating comparison visualizations...")
    create_comparison_plots(all_adapter_results, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    for adapter_name, results in all_adapter_results.items():
        print(f"\nAdapter: {adapter_name}")
        if 'error' in results:
            print(f"  Status: FAILED - {results['error']}")
        else:
            print(f"  Status: SUCCESS")
            modules_count = len(results.get('modules_evaluated', []))
            print(f"  Modules evaluated: {modules_count}")

            # Show key metrics if available
            if 'sparsity_analysis' in results:
                sa = results['sparsity_analysis']
                avg_sparsity = np.mean(
                    [sa[m]['actual_sparsity'] for m in sa.keys()])
                avg_l0 = np.mean([sa[m]['actual_l0'] for m in sa.keys()])
                print(f"  Average sparsity: {avg_sparsity:.3f}")
                print(f"  Average L0: {avg_l0:.1f}")

            if 'reconstruction_analysis' in results:
                ra = results['reconstruction_analysis']
                avg_cosine = np.mean([ra[m]['cosine_similarity']
                                     for m in ra.keys()])
                avg_explained = np.mean(
                    [ra[m]['explained_variance'] for m in ra.keys()])
                print(f"  Average cosine similarity: {avg_cosine:.3f}")
                print(f"  Average explained variance: {avg_explained:.3f}")


if __name__ == "__main__":
    main()

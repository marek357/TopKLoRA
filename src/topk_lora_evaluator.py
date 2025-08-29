"""
Top-K LoRA Evaluator: Comprehensive evaluation framework for assessing
the quality and monosemanticity of latents in Top-K sparse LoRA adapters.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
from pathlib import Path
from tqdm import tqdm
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets library not available. Data loading will be limited.")
    DATASETS_AVAILABLE = False
try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cosine
    from sklearn.metrics import normalized_mutual_info_score
except ImportError:
    print("Warning: scipy/sklearn not available. Some stability analysis features will be limited.")
    linear_sum_assignment = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not available. Plotting features disabled.")
    PLOTTING_AVAILABLE = False

from models import TopKLoRALinear


@dataclass
class EvaluationConfig:
    """Configuration for Top-K LoRA evaluation"""
    adapter_paths: List[str]
    eval_datasets: List[str] = None
    output_dir: str = "evaluation_results"
    max_samples: int = 1000
    batch_size: int = 8
    device: str = "cuda"

    # Causal intervention parameters
    intervention_scales: List[float] = None

    # Stability evaluation
    stability_threshold: float = 0.7

    def __post_init__(self):
        if self.intervention_scales is None:
            self.intervention_scales = [0.0, 0.5, 1.0, 2.0, 5.0]
        if self.eval_datasets is None:
            self.eval_datasets = ["anthropic/hh-rlhf"]


class TopKLoRAEvaluator:
    """Main evaluator for Top-K LoRA adapters"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {}
        self.logger = logging.getLogger(__name__)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def evaluate_all_adapters(self, models_and_tokenizers: List[Tuple]) -> Dict[str, Any]:
        """Run comprehensive evaluation on all adapters"""

        all_results = {}

        for i, (model, tokenizer, wrapped_modules) in enumerate(models_and_tokenizers):
            adapter_name = f"adapter_{i}"
            self.logger.info(f"Evaluating {adapter_name}")

            adapter_results = {
                'causal_interventions': self.evaluate_causal_interventions(
                    model, tokenizer, wrapped_modules
                ),
                'monosemanticity': self.evaluate_monosemanticity(
                    model, tokenizer, wrapped_modules
                ),
                'cost_analysis': self.evaluate_cost(
                    model, tokenizer, wrapped_modules
                )
            }

            all_results[adapter_name] = adapter_results

        # Cross-adapter stability analysis
        if len(models_and_tokenizers) > 1:
            all_results['stability'] = self.evaluate_stability(
                [(model, wrapped_modules)
                 for model, _, wrapped_modules in models_and_tokenizers]
            )

        # Save results
        self.save_results(all_results)
        return all_results

    def evaluate_causal_interventions(
        self,
        model,
        tokenizer,
        wrapped_modules: Dict[str, TopKLoRALinear]
    ) -> Dict[str, Any]:
        """
        Experiment 1 & 2: Causal Δ-loss and behavior shift per latent
        """
        import os
        import json
        self.logger.info("Running causal intervention analysis...")

        results = {
            'per_latent_effects': {},
            'behavior_shifts': {},
            'global_effects': {}
        }

        # Checkpoint file path
        checkpoint_dir = self.config.output_dir if hasattr(
            self.config, 'output_dir') else '.'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Look for any existing causal checkpoint files in the directory
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('causal_checkpoint_') and f.endswith('.json')]
        checkpoint_file = os.path.join(checkpoint_dir, f"causal_checkpoint_{id(model)}.json")
        
        # Try to load existing checkpoint first
        loaded_checkpoint = False
        if checkpoint_files:
            # Try to load the most recent checkpoint file
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            for cf in checkpoint_files:
                checkpoint_path = os.path.join(checkpoint_dir, cf)
                try:
                    with open(checkpoint_path, 'r') as f:
                        results = json.load(f)
                    self.logger.info(f"Loaded existing causal checkpoint from {cf} with {len(results['per_latent_effects'])} layers.")
                    loaded_checkpoint = True
                    # Update the checkpoint file to use current model ID for future saves
                    checkpoint_file = checkpoint_path
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint {cf}: {e}")
                    continue
        
        # If no checkpoint loaded, try the current model ID checkpoint
        if not loaded_checkpoint and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    results = json.load(f)
                self.logger.info(
                    f"Loaded causal interventions checkpoint with {len(results['per_latent_effects'])} layers.")
                loaded_checkpoint = True
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        
        # Initialize results if no checkpoint was loaded
        if not loaded_checkpoint:
            results = {
                'per_latent_effects': {},
                'behavior_shifts': {},
                'global_effects': {}
            }

        # Load evaluation data
        eval_data = self._load_evaluation_data(tokenizer)

        for layer_name, module in wrapped_modules.items():
            if layer_name in results['per_latent_effects']:
                self.logger.info(
                    f"Skipping already completed layer: {layer_name}")
                continue

            self.logger.info(f"Analyzing layer: {layer_name}")

            layer_results = self._analyze_layer_causal_effects(
                model, tokenizer, module, eval_data, layer_name
            )

            results['per_latent_effects'][layer_name] = layer_results['per_latent']
            results['behavior_shifts'][layer_name] = layer_results['behavior_shifts']
            results['global_effects'][layer_name] = layer_results['global']

            # Save checkpoint after each layer
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Checkpoint saved after layer: {layer_name}")
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint: {e}")

        # # Clean up checkpoint file after completion
        # try:
        #     if os.path.exists(checkpoint_file):
        #         os.remove(checkpoint_file)
        #         self.logger.info(
        #             "Causal checkpoint file cleaned up after completion")
        # except Exception as e:
        #     self.logger.warning(f"Failed to clean up checkpoint: {e}")

        return results

    def _analyze_layer_causal_effects(
        self,
        model,
        tokenizer,
        module: TopKLoRALinear,
        eval_data: List[Dict],
        layer_name: str
    ) -> Dict[str, Any]:
        """Analyze causal effects for a single layer"""

        per_latent_effects = []
        behavior_shifts = []
        global_effects = {}

        # Hook to capture intermediate activations
        activations = {}

        def capture_hook(name):
            def hook(module, input, output):
                # Store the z activations before top-k masking
                x = input[0]
                A = module.A.to(dtype=x.dtype, device=x.device)
                z = F.linear(x, A)
                activations[name] = z.detach().clone()
            return hook

        # Register hook
        hook_handle = module.register_forward_hook(capture_hook(layer_name))

        try:
            # Baseline evaluation
            baseline_metrics = self._compute_baseline_metrics(
                model, tokenizer, eval_data)

            # For each latent dimension
            for latent_idx in tqdm(range(module.r)):
                latent_effects = self._evaluate_latent_interventions(
                    model, tokenizer, module, eval_data, layer_name,
                    latent_idx, baseline_metrics, activations
                )

                per_latent_effects.append({
                    'latent_idx': latent_idx,
                    'layer_name': layer_name,
                    **latent_effects
                })

            # Global layer effects
            global_effects = self._compute_global_layer_effects(
                model, tokenizer, module, eval_data, baseline_metrics
            )

        finally:
            hook_handle.remove()

        return {
            'per_latent': per_latent_effects,
            'behavior_shifts': behavior_shifts,
            'global': global_effects
        }

    def _evaluate_latent_interventions(
        self,
        model,
        tokenizer,
        module: TopKLoRALinear,
        eval_data: List[Dict],
        layer_name: str,
        latent_idx: int,
        baseline_metrics: Dict,
        activations: Dict
    ) -> Dict[str, Any]:
        """Evaluate interventions on a specific latent"""

        intervention_results = {}

        for scale in self.config.intervention_scales:
            # Create intervention hook
            def intervention_hook(module, input, output):
                x = input[0]
                A = module.A.to(dtype=x.dtype, device=x.device)
                B = module.B.to(dtype=x.dtype, device=x.device)

                # Compute z
                z = F.linear(x, A)

                # Apply intervention: modify specific latent
                if scale == 0.0:
                    # Ablation: zero out the latent
                    z_modified = z.clone()
                    z_modified[..., latent_idx] = 0.0
                else:
                    # Amplification: scale the latent
                    z_modified = z.clone()
                    z_modified[..., latent_idx] *= scale

                # Apply top-k masking
                if module.k < module.r:
                    z_modified = module.topk(z_modified)

                # Compute output with intervention
                base_out = module.base_layer(x)
                lora_out = F.linear(z_modified, B) * module.scale
                return base_out + lora_out

            # Apply intervention and measure effects
            hook_handle = module.register_forward_hook(intervention_hook)

            try:
                intervened_metrics = self._compute_baseline_metrics(
                    model, tokenizer, eval_data
                )

                # Compute deltas
                delta_loss = intervened_metrics['loss'] - \
                    baseline_metrics['loss']
                delta_perplexity = intervened_metrics['perplexity'] - \
                    baseline_metrics['perplexity']

                # Behavior-specific deltas (if available)
                behavior_deltas = {}
                for behavior in ['toxicity', 'refusal', 'style']:
                    if behavior in baseline_metrics and behavior in intervened_metrics:
                        behavior_deltas[f'delta_{behavior}'] = (
                            intervened_metrics[behavior] -
                            baseline_metrics[behavior]
                        )

                intervention_results[scale] = {
                    'delta_loss': delta_loss,
                    'delta_perplexity': delta_perplexity,
                    **behavior_deltas
                }

            finally:
                hook_handle.remove()

        # Compute activation statistics for this latent
        if layer_name in activations:
            z_vals = activations[layer_name][..., latent_idx]
            activation_stats = {
                'mean_activation': z_vals.mean().item(),
                'std_activation': z_vals.std().item(),
                'sparsity': (z_vals == 0).float().mean().item(),
                'max_activation': z_vals.max().item(),
                'min_activation': z_vals.min().item()
            }
        else:
            activation_stats = {}

        return {
            'interventions': intervention_results,
            'activation_stats': activation_stats
        }

    def evaluate_monosemanticity(
        self,
        model,
        tokenizer,
        wrapped_modules: Dict[str, TopKLoRALinear]
    ) -> Dict[str, Any]:
        """
        Experiment 3: Monosemanticity evaluation
        """
        import os
        import json
        self.logger.info("Evaluating monosemanticity...")

        results = {}

        # Checkpoint file path (per adapter)
        checkpoint_dir = self.config.output_dir if hasattr(
            self.config, 'output_dir') else '.'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Look for any existing monosemanticity checkpoint files in the directory
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('monosemanticity_checkpoint_') and f.endswith('.json')]
        checkpoint_file = os.path.join(checkpoint_dir, f"monosemanticity_checkpoint_{id(model)}.json")
        
        # Try to load existing checkpoint first
        loaded_checkpoint = False
        if checkpoint_files:
            # Try to load the most recent checkpoint file
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            for cf in checkpoint_files:
                checkpoint_path = os.path.join(checkpoint_dir, cf)
                try:
                    with open(checkpoint_path, 'r') as f:
                        results = json.load(f)
                    self.logger.info(f"Loaded existing monosemanticity checkpoint from {cf} with {len(results)} layers.")
                    loaded_checkpoint = True
                    # Update the checkpoint file to use current model ID for future saves
                    checkpoint_file = checkpoint_path
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint {cf}: {e}")
                    continue
        
        # If no checkpoint loaded, try the current model ID checkpoint
        if not loaded_checkpoint and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    results = json.load(f)
                self.logger.info(
                    f"Loaded monosemanticity checkpoint with {len(results)} layers.")
                loaded_checkpoint = True
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        
        # Initialize results if no checkpoint was loaded
        if not loaded_checkpoint:
            results = {}

        # Load diverse evaluation data for monosemanticity analysis
        eval_data = self._load_evaluation_data(tokenizer, diverse=True)

        for layer_name, module in wrapped_modules.items():
            if layer_name in results:
                self.logger.info(
                    f"Skipping already completed layer: {layer_name}")
                continue

            self.logger.info(
                f"Analyzing monosemanticity for layer: {layer_name}")

            layer_results = self._analyze_layer_monosemanticity(
                model, tokenizer, module, eval_data, layer_name
            )

            results[layer_name] = layer_results

            # Save checkpoint after each layer
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Checkpoint saved after layer: {layer_name}")
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint: {e}")

        # Clean up checkpoint file after completion
        # try:
        #     if os.path.exists(checkpoint_file):
        #         os.remove(checkpoint_file)
        #         self.logger.info("Checkpoint file cleaned up after completion")
        # except Exception as e:
        #     self.logger.warning(f"Failed to clean up checkpoint: {e}")

        return results

    def _analyze_layer_monosemanticity(
        self,
        model,
        tokenizer,
        module: TopKLoRALinear,
        eval_data: List[Dict],
        layer_name: str
    ) -> Dict[str, Any]:
        """Analyze monosemanticity metrics for a single layer"""

        # Collect activations across diverse inputs
        all_activations = []
        token_types = []
        prompt_classes = []

        # Check if we have cached activations from previous run
        checkpoint_dir = self.config.output_dir if hasattr(
            self.config, 'output_dir') else '.'
        activations_cache_file = os.path.join(
            checkpoint_dir, f"activations_cache_{layer_name}_{id(model)}.pt")

        if os.path.exists(activations_cache_file):
            try:
                self.logger.info(
                    f"Loading cached activations for {layer_name}")
                cached_data = torch.load(
                    activations_cache_file, map_location='cpu', weights_only=False)

                # Reshape existing activations to correct format
                if 'activations' in cached_data:
                    old_activations = cached_data['activations']
                    self.logger.info(
                        f"Found {len(old_activations)} cached activation tensors")

                    # Reshape each activation tensor from [batch, seq, r] to [batch*seq, r]
                    for i, act in enumerate(old_activations):
                        if len(act.shape) == 3:  # [batch, seq, r]
                            # [batch*seq, r]
                            act_flat = act.reshape(-1, act.shape[-1])
                            all_activations.append(act_flat)
                            self.logger.info(
                                f"Reshaped activation {i} from {act.shape} to {act_flat.shape}")
                        elif len(act.shape) == 2:  # Already flattened
                            all_activations.append(act)
                            self.logger.info(
                                f"Using pre-flattened activation {i} with shape {act.shape}")

                    # Use cached metadata or generate default
                    if 'token_types' in cached_data:
                        token_types = cached_data['token_types']
                    if 'prompt_classes' in cached_data:
                        prompt_classes = cached_data['prompt_classes']

                    self.logger.info(
                        f"Successfully loaded and reshaped cached activations for {layer_name}")

            except Exception as e:
                self.logger.warning(f"Failed to load cached activations: {e}")
                all_activations = []

        # If no cached activations, collect new ones
        if not all_activations:
            self.logger.info(
                f"No cached activations found, collecting new activations for {layer_name}")

            activation_hook_data = {}

            def activation_hook(module, input, output):
                x = input[0]
                A = module.A.to(dtype=x.dtype, device=x.device)
                z = F.linear(x, A)
                activation_hook_data['z'] = z.detach().clone()

            hook_handle = module.register_forward_hook(activation_hook)

            try:
                model.eval()
                with torch.no_grad():
                    for batch_data in eval_data:
                        inputs = tokenizer(
                            batch_data['text'],
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(self.config.device)

                        # Forward pass
                        _ = model(**inputs)

                        if 'z' in activation_hook_data:
                            z = activation_hook_data['z']  # [batch, seq, r]
                            # [batch*seq, r]
                            z_flat = z.reshape(-1, z.shape[-1])
                            all_activations.append(z_flat.cpu())

                            # Track metadata for selectivity analysis
                            num_tokens = z.shape[0] * z.shape[1]
                            token_types.extend(batch_data.get(
                                'token_types', ['unknown'] * num_tokens))
                            prompt_classes.extend(
                                [batch_data.get('prompt_class', 'unknown')] * num_tokens)

                # Cache the new activations for future use
                try:
                    cache_data = {
                        'activations': all_activations,
                        'token_types': token_types,
                        'prompt_classes': prompt_classes
                    }
                    torch.save(cache_data, activations_cache_file, _use_new_zipfile_serialization=False)
                    self.logger.info(
                        f"Cached activations saved to {activations_cache_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache activations: {e}")

            finally:
                hook_handle.remove()

        if not all_activations:
            return {'error': 'No activations collected'}

        # Generate default metadata if missing
        total_tokens = sum(act.shape[0] for act in all_activations)
        if not token_types:
            token_types = ['unknown'] * total_tokens
        if not prompt_classes:
            prompt_classes = ['general'] * total_tokens

        # Concatenate all activations along the token axis
        all_z = torch.cat(all_activations, dim=0)  # Shape: [total_tokens, r]
        self.logger.info(
            f"Final concatenated activations shape: {all_z.shape}")

        # 1. Mean active-k per sample
        active_k_per_sample = (all_z != 0).sum(dim=-1).float()
        mean_active_k = active_k_per_sample.mean().item()
        std_active_k = active_k_per_sample.std().item()

        # 2. Selectivity analysis
        selectivity_results = self._compute_selectivity(
            all_z, token_types, prompt_classes
        )

        # 3. Duplication rate (cosine similarity between B columns)
        B_matrix = module.B.detach().cpu()  # Shape: [output_dim, r]
        duplication_rate = self._compute_duplication_rate(
            B_matrix.T)  # [r, output_dim]

        # 4. Feature interpretability metrics
        interpretability_metrics = self._compute_interpretability_metrics(
            all_z)

        return {
            'mean_active_k': mean_active_k,
            'std_active_k': std_active_k,
            'selectivity': selectivity_results,
            'duplication_rate': duplication_rate,
            'interpretability': interpretability_metrics,
            'total_samples': all_z.size(0),
            'latent_dim': all_z.size(1)
        }

    def _compute_selectivity(
        self,
        activations: torch.Tensor,
        token_types: List[str],
        prompt_classes: List[str]
    ) -> Dict[str, float]:
        """Compute selectivity metrics using entropy and Gini coefficient"""

        results = {}

        # Convert to numpy for easier manipulation
        z = activations.numpy()

        # For each latent dimension
        latent_selectivities = []

        for dim in range(z.shape[1]):
            dim_activations = z[:, dim]

            # Only consider positions where this latent is active
            active_mask = dim_activations != 0

            if active_mask.sum().item() == 0:
                continue

            # Compute selectivity over token types
            if len(set(token_types)) > 1:
                token_selectivity = self._compute_entropy_selectivity(
                    dim_activations, token_types, active_mask
                )
                latent_selectivities.append(token_selectivity)

        if latent_selectivities:
            results['mean_token_selectivity'] = np.mean(latent_selectivities)
            results['std_token_selectivity'] = np.std(latent_selectivities)

        # Global sparsity
        results['global_sparsity'] = (z == 0).mean()

        # Gini coefficient for activation distribution
        results['gini_coefficient'] = self._compute_gini_coefficient(
            np.abs(z[z != 0]).flatten()
        )

        return results

    def _compute_entropy_selectivity(
        self,
        activations: np.ndarray,
        labels: List[str],
        active_mask: np.ndarray
    ) -> float:
        """Compute entropy-based selectivity"""

        # Get active activations and corresponding labels
        active_activations = activations[active_mask]
        active_labels = [labels[i]
                         for i in range(len(labels)) if active_mask[i]]

        if len(active_labels) == 0:
            return 0.0

        # Compute activation weights per label
        label_weights = defaultdict(float)
        for act, label in zip(active_activations, active_labels):
            label_weights[label] += abs(act)

        # Normalize to get distribution
        total_weight = sum(label_weights.values())
        if total_weight == 0:
            return 0.0

        probs = [w / total_weight for w in label_weights.values()]

        # Compute entropy
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)
        max_entropy = np.log(len(probs))

        # Return normalized selectivity (1 - normalized_entropy)
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for inequality measurement"""
        if len(values) == 0:
            return 0.0

        values = np.sort(np.abs(values))
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n

    def _compute_duplication_rate(self, features: torch.Tensor) -> Dict[str, float]:
        """Compute duplication rate using cosine similarity"""

        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)

        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(features_norm, features_norm.T)

        # Remove diagonal (self-similarity)
        mask = torch.eye(features.size(0), dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)

        # Find nearest neighbors
        max_similarities, _ = similarity_matrix.max(dim=1)

        # Duplication metrics
        high_similarity_threshold = 0.8
        duplication_rate = (max_similarities >
                            high_similarity_threshold).float().mean().item()

        return {
            'duplication_rate': duplication_rate,
            'mean_max_similarity': max_similarities.mean().item(),
            'std_max_similarity': max_similarities.std().item(),
            'median_max_similarity': max_similarities.median().item()
        }

    def _compute_interpretability_metrics(self, activations: torch.Tensor) -> Dict[str, float]:
        """Compute additional interpretability metrics"""

        # Activation distribution metrics
        nonzero_activations = activations[activations != 0]

        if len(nonzero_activations) == 0:
            return {'error': 'No non-zero activations'}

        return {
            'mean_nonzero_activation': nonzero_activations.mean().item(),
            'std_nonzero_activation': nonzero_activations.std().item(),
            'kurtosis': self._compute_kurtosis(nonzero_activations),
            'skewness': self._compute_skewness(nonzero_activations),
            'activation_range': (nonzero_activations.max() - nonzero_activations.min()).item()
        }

    def _compute_skewness(self, tensor: torch.Tensor) -> float:
        """Compute skewness of tensor values"""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return 0.0
        return ((tensor - mean) ** 3).mean().item() / (std ** 3).item()

    def _compute_kurtosis(self, tensor: torch.Tensor) -> float:
        """Compute kurtosis of tensor values (excess kurtosis, normal distribution = 0)"""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return 0.0
        centered = tensor - mean
        return ((centered ** 4).mean() / (std ** 4) - 3).item()

    def evaluate_stability(
        self,
        models_and_modules: List[Tuple]
    ) -> Dict[str, Any]:
        """
        Experiment 4: Stability analysis across adapters
        """
        self.logger.info("Evaluating stability across adapters...")

        if len(models_and_modules) < 2:
            return {'error': 'Need at least 2 adapters for stability analysis'}

        results = {}

        # Extract B matrices from all adapters
        all_b_matrices = {}

        for i, (model, wrapped_modules) in enumerate(models_and_modules):
            adapter_name = f"adapter_{i}"
            all_b_matrices[adapter_name] = {}

            for layer_name, module in wrapped_modules.items():
                B = module.B.detach().cpu().T  # Shape: [r, output_dim]
                all_b_matrices[adapter_name][layer_name] = F.normalize(
                    B, p=2, dim=1)

        # Pairwise stability analysis
        adapter_names = list(all_b_matrices.keys())
        stability_results = {}

        for i in range(len(adapter_names)):
            for j in range(i + 1, len(adapter_names)):
                adapter_a = adapter_names[i]
                adapter_b = adapter_names[j]

                pair_key = f"{adapter_a}_vs_{adapter_b}"
                stability_results[pair_key] = self._compute_pairwise_stability(
                    all_b_matrices[adapter_a],
                    all_b_matrices[adapter_b]
                )

        results['pairwise_stability'] = stability_results

        # Overall stability metrics
        results['overall_stability'] = self._compute_overall_stability(
            stability_results)

        return results

    def _compute_pairwise_stability(
        self,
        matrices_a: Dict[str, torch.Tensor],
        matrices_b: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Compute stability metrics between two adapters"""

        if linear_sum_assignment is None:
            return {'error': 'scipy not available for Hungarian matching'}

        layer_stabilities = {}

        for layer_name in matrices_a.keys():
            if layer_name not in matrices_b:
                continue

            B_a = matrices_a[layer_name]  # [r, output_dim]
            B_b = matrices_b[layer_name]  # [r, output_dim]

            # Hungarian matching based on cosine similarity
            similarity_matrix = torch.mm(B_a, B_b.T).numpy()

            # Solve assignment problem (maximize similarity)
            row_indices, col_indices = linear_sum_assignment(
                -similarity_matrix)

            # Compute matched similarities
            matched_similarities = similarity_matrix[row_indices, col_indices]

            # Stability metrics
            percent_matched = (matched_similarities >
                               self.config.stability_threshold).mean()
            mean_similarity = matched_similarities.mean()
            std_similarity = matched_similarities.std()

            layer_stabilities[layer_name] = {
                'percent_matched': percent_matched,
                'mean_similarity': mean_similarity,
                'std_similarity': std_similarity,
                'matched_similarities': matched_similarities.tolist()
            }

        return layer_stabilities

    def _compute_overall_stability(self, stability_results: Dict) -> Dict[str, float]:
        """Compute overall stability metrics across all pairs and layers"""

        all_similarities = []
        all_percent_matched = []

        for pair_results in stability_results.values():
            for layer_results in pair_results.values():
                all_similarities.extend(layer_results['matched_similarities'])
                all_percent_matched.append(layer_results['percent_matched'])

        return {
            'overall_mean_similarity': float(np.mean(all_similarities)),
            'overall_std_similarity': float(np.std(all_similarities)),
            'overall_percent_matched': float(np.mean(all_percent_matched)),
            'num_comparisons': len(all_similarities)
        }

    def evaluate_cost(
        self,
        model,
        tokenizer,
        wrapped_modules: Dict[str, TopKLoRALinear]
    ) -> Dict[str, Any]:
        """
        Experiment 5: Cost analysis
        """
        self.logger.info("Evaluating computational costs...")

        results = {}

        # 1. Parameter count analysis
        results['parameters'] = self._analyze_parameter_costs(wrapped_modules)

        # 2. Inference overhead
        results['inference'] = self._analyze_inference_costs(model, tokenizer)

        # 3. Memory usage
        results['memory'] = self._analyze_memory_costs(model, wrapped_modules)

        return results

    def _analyze_parameter_costs(self, wrapped_modules: Dict[str, TopKLoRALinear]) -> Dict[str, Any]:
        """Analyze parameter counts and efficiency"""

        total_base_params = 0
        total_lora_params = 0
        total_effective_params = 0

        layer_details = {}

        for layer_name, module in wrapped_modules.items():
            # Base layer parameters
            base_params = sum(p.numel()
                              for p in module.base_layer.parameters())

            # LoRA parameters
            lora_params = module.A.numel() + module.B.numel()

            # Effective parameters (considering sparsity)
            effective_params = lora_params * (module.k / module.r)

            layer_details[layer_name] = {
                'base_params': base_params,
                'lora_params': lora_params,
                'effective_params': effective_params,
                'sparsity_ratio': 1.0 - (module.k / module.r),
                'r': module.r,
                'k': module.k
            }

            total_base_params += base_params
            total_lora_params += lora_params
            total_effective_params += effective_params

        return {
            'total_base_params': total_base_params,
            'total_lora_params': total_lora_params,
            'total_effective_params': total_effective_params,
            'compression_ratio': total_lora_params / total_base_params,
            'effective_compression_ratio': total_effective_params / total_base_params,
            'layer_details': layer_details
        }

    def _analyze_inference_costs(self, model, tokenizer) -> Dict[str, Any]:
        """Analyze inference time overhead"""

        # Prepare test inputs
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "To be or not to be, that is the question."
        ] * 10  # Repeat for more stable timing

        inputs = tokenizer(
            test_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.config.device)

        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(**inputs)

        # Time inference
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(20):
                _ = model(**inputs)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time_per_batch = (end_time - start_time) / 20
        avg_time_per_sample = avg_time_per_batch / len(test_texts)

        return {
            'avg_time_per_batch': avg_time_per_batch,
            'avg_time_per_sample': avg_time_per_sample,
            'batch_size': len(test_texts),
            'sequence_length': inputs['input_ids'].size(1)
        }

    def _analyze_memory_costs(self, model, wrapped_modules: Dict[str, TopKLoRALinear]) -> Dict[str, Any]:
        """Analyze memory usage"""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Get baseline memory
            baseline_memory = torch.cuda.memory_allocated()

            # Dummy forward pass to measure peak memory
            dummy_input = torch.randn(
                1, 128, model.config.hidden_size).to(self.config.device)

            with torch.no_grad():
                _ = model(inputs_embeds=dummy_input)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_overhead = peak_memory - baseline_memory

            return {
                'baseline_memory_mb': baseline_memory / 1e6,
                'peak_memory_mb': peak_memory / 1e6,
                'memory_overhead_mb': memory_overhead / 1e6
            }
        else:
            return {'error': 'CUDA not available for memory analysis'}

    def _load_evaluation_data(self, tokenizer, diverse: bool = False) -> List[Dict]:
        """Load evaluation data for experiments using HH-RLHF dataset"""

        if not DATASETS_AVAILABLE:
            self.logger.warning(
                "Datasets library not available, falling back to placeholder data")
            return self._load_placeholder_data(diverse)

        try:
            # Load HH-RLHF test split
            dataset = load_dataset("Anthropic/hh-rlhf", split="test")

            # Sample data based on max_samples
            if len(dataset) > self.config.max_samples:
                # Use deterministic sampling for reproducibility
                indices = list(range(0, len(dataset), len(
                    dataset) // self.config.max_samples))[:self.config.max_samples]
                dataset = dataset.select(indices)

            eval_data = []

            for i, example in enumerate(dataset):
                if diverse:
                    # For monosemanticity analysis, extract diverse conversation types
                    chosen = example['chosen']
                    rejected = example['rejected']

                    # Classify conversation type based on content
                    prompt_class = self._classify_conversation_type(chosen)

                    # Add both chosen and rejected responses for diversity
                    eval_data.append({
                        "text": chosen,
                        "prompt_class": prompt_class,
                        "token_types": self._extract_token_types(chosen, tokenizer),
                        "conversation_type": "chosen"
                    })

                    eval_data.append({
                        "text": rejected,
                        "prompt_class": prompt_class,
                        "token_types": self._extract_token_types(rejected, tokenizer),
                        "conversation_type": "rejected"
                    })
                else:
                    # For causal analysis, use chosen responses
                    eval_data.append({
                        "text": example['chosen'],
                        "example_id": i
                    })

                if len(eval_data) >= self.config.max_samples:
                    break

            self.logger.info(
                f"Loaded {len(eval_data)} examples from HH-RLHF test split")
            return eval_data[:self.config.max_samples]

        except Exception as e:
            self.logger.error(f"Failed to load HH-RLHF dataset: {e}")
            self.logger.warning("Falling back to placeholder data")
            return self._load_placeholder_data(diverse)

    def _classify_conversation_type(self, conversation: str) -> str:
        """Classify conversation type based on content patterns"""
        conversation_lower = conversation.lower()

        # Simple heuristic classification
        if any(word in conversation_lower for word in ['help', 'question', 'how', 'what', 'why', 'when', 'where']):
            return "helpful"
        elif any(word in conversation_lower for word in ['sorry', 'cannot', 'inappropriate', 'harmful']):
            return "harmless"
        elif any(word in conversation_lower for word in ['write', 'create', 'story', 'poem', 'code']):
            return "creative"
        elif any(word in conversation_lower for word in ['fact', 'true', 'false', 'correct', 'accurate']):
            return "factual"
        else:
            return "general"

    def _extract_token_types(self, text: str, tokenizer) -> List[str]:
        """Extract simple token type classification for selectivity analysis"""
        # Simple token type classification based on content
        words = text.lower().split()
        token_types = []

        for word in words[:50]:  # Limit to first 50 words for efficiency
            if word in ['human:', 'assistant:', 'user:']:
                token_types.append('role')
            elif any(char.isdigit() for char in word):
                token_types.append('numeric')
            elif word in ['please', 'thank', 'sorry', 'help']:
                token_types.append('polite')
            elif word in ['not', 'no', 'never', 'cannot', 'won\'t']:
                token_types.append('negative')
            elif len(word) > 8:
                token_types.append('long_word')
            else:
                token_types.append('general')

        return token_types

    def _load_placeholder_data(self, diverse: bool = False) -> List[Dict]:
        """Fallback placeholder data loading"""

        if diverse:
            # For monosemanticity analysis, load diverse data
            sample_texts = [
                {"text": "This is a positive review.",
                    "prompt_class": "sentiment", "token_types": ["pos"] * 6},
                {"text": "This movie was terrible.",
                    "prompt_class": "sentiment", "token_types": ["neg"] * 5},
                {"text": "The capital of France is Paris.",
                    "prompt_class": "factual", "token_types": ["fact"] * 7},
                {"text": "Once upon a time in a faraway land.",
                    "prompt_class": "narrative", "token_types": ["story"] * 8},
            ] * (self.config.max_samples // 4)
        else:
            # For causal analysis, standard evaluation data
            sample_texts = [
                {"text": "The quick brown fox jumps over the lazy dog."},
                {"text": "In machine learning, we often use neural networks."},
                {"text": "Climate change is a significant global challenge."},
            ] * (self.config.max_samples // 3)

        return sample_texts[:self.config.max_samples]

    def _compute_baseline_metrics(self, model, tokenizer, eval_data: List[Dict]) -> Dict[str, float]:
        """Compute baseline metrics for comparison"""

        model.eval()
        total_loss = 0.0
        total_tokens = 0

        # Initialize behavior-specific metrics
        toxicity_scores = []
        refusal_indicators = []
        style_consistency_scores = []

        with torch.no_grad():
            for batch_data in eval_data[:50]:  # Sample for efficiency
                inputs = tokenizer(
                    batch_data['text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.config.device)

                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

                total_loss += loss.item() * inputs['input_ids'].numel()
                total_tokens += inputs['input_ids'].numel()

                # Compute behavior-specific metrics
                text = batch_data['text']
                toxicity_scores.append(self._compute_toxicity_score(text))
                refusal_indicators.append(self._compute_refusal_score(text))
                style_consistency_scores.append(
                    self._compute_style_score(text))

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity
        }

        # Add behavior metrics if we have scores
        if toxicity_scores:
            metrics['toxicity'] = np.mean(toxicity_scores)
            metrics['refusal'] = np.mean(refusal_indicators)
            metrics['style'] = np.mean(style_consistency_scores)

        return metrics

    def _compute_toxicity_score(self, text: str) -> float:
        """Compute a simple toxicity score based on keyword detection"""
        # This is a simplified toxicity detector - in practice you'd use a proper model
        toxic_keywords = [
            'hate', 'kill', 'death', 'violence', 'hurt', 'harm', 'stupid', 'idiot',
            'racist', 'sexist', 'offensive', 'inappropriate', 'disgusting'
        ]

        text_lower = text.lower()
        toxic_count = sum(
            1 for keyword in toxic_keywords if keyword in text_lower)

        # Normalize by text length (approximate)
        word_count = len(text.split())
        return toxic_count / max(word_count, 1) if word_count > 0 else 0.0

    def _compute_refusal_score(self, text: str) -> float:
        """Compute refusal/helpfulness score"""
        refusal_phrases = [
            "i can't", "i cannot", "i'm not able", "i'm sorry", "i apologize",
            "that's inappropriate", "i shouldn't", "i won't", "not appropriate",
            "cannot help", "unable to", "refuse to"
        ]

        text_lower = text.lower()
        refusal_count = sum(
            1 for phrase in refusal_phrases if phrase in text_lower)

        # Higher score means more refusal/less helpful
        word_count = len(text.split())
        return refusal_count / max(word_count, 1) if word_count > 0 else 0.0

    def _compute_style_score(self, text: str) -> float:
        """Compute style consistency score"""
        # Simple metrics for style consistency
        sentences = text.split('.')
        if len(sentences) < 2:
            return 1.0  # Single sentence, perfectly consistent

        # Check for consistent capitalization and punctuation
        capitalized_sentences = sum(
            1 for s in sentences if s.strip() and s.strip()[0].isupper())
        consistency_ratio = capitalized_sentences / \
            len(sentences) if sentences else 1.0

        # Check for consistent sentence length (less variation = more consistent)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(sentence_lengths) > 1:
            length_std = np.std(sentence_lengths)
            length_mean = np.mean(sentence_lengths)
            length_consistency = 1.0 / (1.0 + length_std / max(length_mean, 1))
        else:
            length_consistency = 1.0

        return float((consistency_ratio + length_consistency) / 2.0)

    def _compute_global_layer_effects(
        self,
        model,
        tokenizer,
        module: TopKLoRALinear,
        eval_data: List[Dict],
        baseline_metrics: Dict
    ) -> Dict[str, Any]:
        """Compute global effects when entire layer is ablated"""

        # Ablate entire layer
        def ablation_hook(module, input, output):
            # Return just the base layer output (no LoRA)
            x = input[0]
            return module.base_layer(x)

        hook_handle = module.register_forward_hook(ablation_hook)

        try:
            ablated_metrics = self._compute_baseline_metrics(
                model, tokenizer, eval_data)

            global_delta_loss = ablated_metrics['loss'] - \
                baseline_metrics['loss']
            global_delta_perplexity = ablated_metrics['perplexity'] - \
                baseline_metrics['perplexity']

        finally:
            hook_handle.remove()

        return {
            'global_delta_loss': global_delta_loss,
            'global_delta_perplexity': global_delta_perplexity,
            'layer_importance': abs(global_delta_loss)
        }

    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""

        output_path = Path(self.config.output_dir) / \
            "topk_lora_evaluation_results.json"

        # Convert any tensors to lists for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
                return float(obj)
            elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

        # Also save summary CSV for easy analysis
        self._save_summary_csv(results)

    def _save_summary_csv(self, results: Dict[str, Any]):
        """Save summary results as CSV"""

        summary_data = []

        for adapter_name, adapter_results in results.items():
            if adapter_name == 'stability':
                continue

            # Extract key metrics for each layer
            if 'monosemanticity' in adapter_results:
                for layer_name, layer_results in adapter_results['monosemanticity'].items():
                    if isinstance(layer_results, dict) and 'error' not in layer_results:
                        summary_data.append({
                            'adapter': adapter_name,
                            'layer': layer_name,
                            'mean_active_k': layer_results.get('mean_active_k', 0),
                            'global_sparsity': layer_results.get('selectivity', {}).get('global_sparsity', 0),
                            'duplication_rate': layer_results.get('duplication_rate', {}).get('duplication_rate', 0),
                            'gini_coefficient': layer_results.get('selectivity', {}).get('gini_coefficient', 0)
                        })

        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_path = Path(self.config.output_dir) / \
                "evaluation_summary.csv"
            df.to_csv(summary_path, index=False)
            self.logger.info(f"Summary saved to {summary_path}")


def create_visualization_plots(results_path: str, output_dir: str):
    """Create visualization plots for the evaluation results"""

    if not PLOTTING_AVAILABLE:
        print("Warning: Plotting libraries not available. Skipping visualization.")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')

    # 1. Monosemanticity overview plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Top-K LoRA Monosemanticity Analysis', fontsize=16)

    # Extract monosemanticity data
    mono_data = []
    for adapter_name, adapter_results in results.items():
        if 'monosemanticity' in adapter_results:
            for layer_name, layer_results in adapter_results['monosemanticity'].items():
                if isinstance(layer_results, dict) and 'error' not in layer_results:
                    mono_data.append({
                        'adapter': adapter_name,
                        'layer': layer_name,
                        'mean_active_k': layer_results.get('mean_active_k', 0),
                        'sparsity': layer_results.get('selectivity', {}).get('global_sparsity', 0),
                        'duplication_rate': layer_results.get('duplication_rate', {}).get('duplication_rate', 0),
                        'gini': layer_results.get('selectivity', {}).get('gini_coefficient', 0)
                    })

    if mono_data:
        df = pd.DataFrame(mono_data)

        # Plot 1: Active-k distribution
        df.boxplot(column='mean_active_k', by='adapter', ax=axes[0, 0])
        axes[0, 0].set_title('Mean Active-k per Adapter')
        axes[0, 0].set_xlabel('Adapter')
        axes[0, 0].set_ylabel('Mean Active-k')

        # Plot 2: Sparsity vs Duplication
        for adapter in df['adapter'].unique():
            adapter_data = df[df['adapter'] == adapter]
            axes[0, 1].scatter(adapter_data['sparsity'], adapter_data['duplication_rate'],
                               label=adapter, alpha=0.7)
        axes[0, 1].set_xlabel('Global Sparsity')
        axes[0, 1].set_ylabel('Duplication Rate')
        axes[0, 1].set_title('Sparsity vs Duplication Rate')
        axes[0, 1].legend()

        # Plot 3: Gini coefficient distribution
        df.boxplot(column='gini', by='adapter', ax=axes[1, 0])
        axes[1, 0].set_title('Gini Coefficient Distribution')
        axes[1, 0].set_xlabel('Adapter')
        axes[1, 0].set_ylabel('Gini Coefficient')

        # Plot 4: Layer-wise comparison
        layer_means = df.groupby(['adapter', 'layer'])[
            'mean_active_k'].mean().unstack()
        layer_means.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Mean Active-k by Layer')
        axes[1, 1].set_xlabel('Adapter')
        axes[1, 1].set_ylabel('Mean Active-k')
        axes[1, 1].legend(title='Layer', bbox_to_anchor=(
            1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / 'monosemanticity_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Stability analysis plot (if available)
    if 'stability' in results:
        stability_data = results['stability']

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if 'pairwise_stability' in stability_data:
            pairs = []
            similarities = []

            for pair_name, pair_results in stability_data['pairwise_stability'].items():
                for layer_name, layer_results in pair_results.items():
                    pairs.extend([f"{pair_name}_{layer_name}"] *
                                 len(layer_results['matched_similarities']))
                    similarities.extend(layer_results['matched_similarities'])

            if pairs and similarities:
                df_stability = pd.DataFrame(
                    {'pair': pairs, 'similarity': similarities})
                df_stability.boxplot(column='similarity', by='pair', ax=ax)
                ax.set_title('Cross-Adapter Latent Stability')
                ax.set_xlabel('Adapter Pair & Layer')
                ax.set_ylabel('Cosine Similarity')
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / 'stability_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualization plots saved to {output_dir}")

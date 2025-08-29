"""
E                        if 'z' in activation_hook_data:
                            z = activation_hook_data['z']  # Shape: [batch_size, seq_len, r]
                            
                            # Add the full batch tensor (since our batch size is 1, this is fine)
                            # z has shape [1, seq_len, r] 
                            all_activations.append(z.cpu())
                            
                            # Track metadata for this batch
                            prompt_classes.append(batch_data.get('prompt_class', 'unknown'))op-K LoRA Evaluation with Progress Tracking

This version is optimized for speed:
- Limited causal interventions (sample of latents)
- Progre                        if 'z' in activation_hook_data:
                            z = activation_hook_data['z']
                            # Flatten batch and sequence dimensions: [batch, seq, r] -> [batch*seq, r]
                            z_flat = z.view(-1, z.size(-1)).cpu()
                            all_activations.append(z_flat)
                            
                            # Track metadata for each token
                            n_tokens = z_flat.size(0)
                            token_types.extend(['token'] * n_tokens)
                                                        prompt_classes.extend([batch_data.get('prompt_class', 'unknown')] * n_tokens)s and time estimates
- Configurable depth of analysis
"""

import sys
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from topk_lora_evaluator import TopKLoRAEvaluator, EvaluationConfig
    from run_comprehensive_evaluation import load_adapter_safely, ModelConfig
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)


class EfficientTopKLoRAEvaluator(TopKLoRAEvaluator):
    """Efficient version with progress tracking and sampling"""

    def __init__(self, config: EvaluationConfig, sample_latents: int = 10):
        super().__init__(config)
        self.sample_latents = sample_latents

    def evaluate_monosemanticity(self, model, tokenizer, wrapped_modules):
        """Override parent method with efficient implementation"""
        results = {}

        # Simple evaluation data (just a few prompts)
        eval_data = [
            {'text': "The quick brown fox jumps over the lazy dog.",
                'prompt_class': 'simple'},
            {'text': "What is the capital of France?", 'prompt_class': 'question'},
            {'text': "Once upon a time in a land far away.", 'prompt_class': 'story'},
            {'text': "The weather today is sunny and warm.",
                'prompt_class': 'description'},
            {'text': "How to make a delicious chocolate cake?",
                'prompt_class': 'instruction'}
        ] * 4  # Repeat to get 20 samples

        for layer_name, module in wrapped_modules.items():
            try:
                layer_results = self._analyze_layer_monosemanticity(
                    model, tokenizer, module, eval_data, layer_name
                )
                results[layer_name] = layer_results
            except Exception as e:
                self.logger.error(f"Error analyzing {layer_name}: {e}")
                results[layer_name] = {'error': str(e)}

        return results

    def _analyze_layer_causal_effects(self, model, tokenizer, module, eval_data, layer_name):
        """Efficient version that samples latents instead of testing all"""

        self.logger.info(
            f"Analyzing layer: {layer_name} (r={module.r}, sampling {self.sample_latents} latents)")

        per_latent_effects = []
        behavior_shifts = []
        global_effects = {}

        # Sample latent indices instead of testing all
        if self.sample_latents >= module.r:
            latent_indices = list(range(module.r))
        else:
            import random
            random.seed(42)  # Reproducible sampling
            latent_indices = random.sample(
                range(module.r), self.sample_latents)

        # Baseline evaluation
        baseline_metrics = self._compute_baseline_metrics(
            model, tokenizer, eval_data)

        # Progress bar for latent interventions
        with tqdm(latent_indices, desc=f"Testing latents in {layer_name.split('.')[-1]}") as pbar:
            for latent_idx in pbar:
                pbar.set_postfix({'latent': f"{latent_idx}/{module.r}"})

                # Quick intervention test (only ablation, skip amplification for speed)
                intervention_results = self._quick_latent_intervention(
                    model, tokenizer, module, eval_data, latent_idx, baseline_metrics
                )

                per_latent_effects.append({
                    'latent_idx': latent_idx,
                    'layer_name': layer_name,
                    **intervention_results
                })

        # Global layer effects
        self.logger.info(f"Computing global effects for {layer_name}")
        global_effects = self._compute_global_layer_effects(
            model, tokenizer, module, eval_data, baseline_metrics
        )

        return {
            'per_latent': per_latent_effects,
            'behavior_shifts': behavior_shifts,
            'global': global_effects
        }

    def _quick_latent_intervention(self, model, tokenizer, module, eval_data, latent_idx, baseline_metrics):
        """Quick latent intervention (ablation only)"""

        # Only test ablation (scale=0.0) for speed
        def intervention_hook(module, input, output):
            x = input[0]
            A = module.A.to(dtype=x.dtype, device=x.device)
            B = module.B.to(dtype=x.dtype, device=x.device)

            z = F.linear(x, A)
            z_modified = z.clone()
            z_modified[..., latent_idx] = 0.0  # Ablation

            if module.k < module.r:
                z_modified = module.topk(z_modified)

            base_out = module.base_layer(x)
            lora_out = F.linear(z_modified, B) * module.scale
            return base_out + lora_out

        hook_handle = module.register_forward_hook(intervention_hook)

        try:
            # Use fewer samples for intervention (speed up)
            small_eval_data = eval_data[:5]  # Just 5 samples
            intervened_metrics = self._compute_baseline_metrics(
                model, tokenizer, small_eval_data)

            delta_loss = intervened_metrics['loss'] - baseline_metrics['loss']
            delta_perplexity = intervened_metrics['perplexity'] - \
                baseline_metrics['perplexity']

            return {
                'delta_loss_ablation': delta_loss,
                'delta_perplexity_ablation': delta_perplexity,
                'effect_magnitude': abs(delta_loss)
            }

        finally:
            hook_handle.remove()

    def _analyze_layer_monosemanticity(self, model, tokenizer, module, eval_data, layer_name):
        """Efficient monosemanticity analysis with progress tracking"""

        self.logger.info(f"Monosemanticity analysis for {layer_name}")

        all_activations = []
        token_types = []
        prompt_classes = []

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
                # Use progress bar for data processing
                # Limit to 20 samples
                with tqdm(eval_data[:20], desc="Processing samples") as pbar:
                    for batch_data in pbar:
                        inputs = tokenizer(
                            batch_data['text'],
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=256  # Shorter sequences for speed
                        ).to(self.config.device)

                        _ = model(**inputs)

                        if 'z' in activation_hook_data:
                            z = activation_hook_data['z']
                            all_activations.append(z.cpu())

                            token_types.extend(batch_data.get(
                                'token_types', ['unknown'] * z.size(1)))
                            prompt_classes.extend(
                                [batch_data.get('prompt_class', 'unknown')] * z.size(0))

        finally:
            hook_handle.remove()

        if not all_activations:
            return {'error': 'No activations collected'}

        # Debug: Check shapes before concatenation
        shapes = [z.shape for z in all_activations]
        print(f"DEBUG: Activation shapes before cat: {shapes}")

        if not shapes:
            return {'error': 'No activations collected'}

        # Check if all tensors have same number of features (last dimension)
        r_dims = [z.shape[-1] for z in all_activations]
        if len(set(r_dims)) > 1:
            return {'error': f'Inconsistent r dimensions: {r_dims}'}

        # Find max sequence length - note z has shape [batch, seq, r] from batching
        # so we need to get seq_len from dimension 1
        # z is [batch, seq_len, r] - get seq dimension
        seq_lens = [z.shape[1] for z in all_activations]
        max_seq_len = max(seq_lens)
        r_dim = all_activations[0].shape[-1]

        print(
            f"DEBUG: seq_lens={seq_lens}, max_seq_len={max_seq_len}, r_dim={r_dim}")

        # Pad all tensors to same sequence length
        # We need to work with [batch, seq, r] tensors, then flatten after padding
        padded_activations = []
        for i, z in enumerate(all_activations):
            # z is [batch, seq, r], we want to pad the seq dimension
            current_seq_len = z.shape[1]
            if current_seq_len < max_seq_len:
                # Pad along sequence dimension: [batch, seq, r] -> [batch, max_seq, r]
                batch_size, _, r_dim = z.shape
                padding = torch.zeros(
                    batch_size, max_seq_len - current_seq_len, r_dim)
                z_padded = torch.cat([z, padding], dim=1)
            else:
                z_padded = z

            # Now flatten batch and seq dimensions: [batch, seq, r] -> [batch*seq, r]
            z_flat = z_padded.view(-1, r_dim)
            padded_activations.append(z_flat)
            print(
                f"DEBUG: padded activation {i}: {z.shape} -> {z_padded.shape} -> {z_flat.shape}")

        print(
            f"DEBUG: About to concat {len(padded_activations)} padded tensors")
        all_z = torch.cat(padded_activations, dim=0)

        self.logger.info(
            f"Computing metrics for {all_z.shape[0]} samples, {all_z.shape[1]} latents")

        # Core metrics
        active_k_per_sample = (all_z != 0).sum(dim=-1).float()
        mean_active_k = active_k_per_sample.mean().item()
        std_active_k = active_k_per_sample.std().item()

        # Selectivity (simplified)
        z_np = all_z.numpy()
        global_sparsity = (z_np == 0).mean()

        # Duplication rate
        B_matrix = module.B.detach().cpu()
        duplication_rate = self._compute_duplication_rate(B_matrix.T)

        return {
            'mean_active_k': mean_active_k,
            'std_active_k': std_active_k,
            'global_sparsity': global_sparsity,
            'duplication_rate': duplication_rate,
            'total_samples': all_z.size(0),
            'latent_dim': all_z.size(1)
        }


def quick_evaluation_with_progress(adapter_paths, max_samples=20, sample_latents=5):
    """Run quick evaluation with progress tracking"""

    print("Quick Top-K LoRA Evaluation with Progress Tracking")
    print("=" * 60)
    print(f"Settings:")
    print(f"  - Max samples per test: {max_samples}")
    print(f"  - Latents sampled per layer: {sample_latents}")
    print(f"  - Adapters: {len(adapter_paths)}")
    print()

    # Load adapters with progress
    models_and_tokenizers = []

    print("Loading adapters...")
    for i, adapter_path in enumerate(tqdm(adapter_paths, desc="Loading")):
        result = load_adapter_safely(adapter_path, device="cuda")
        if result[0] is not None:
            models_and_tokenizers.append(result[:3])
            adapter_info = result[3]
            print(
                f"  ✓ Adapter {i}: r={adapter_info.get('r', '?')}, k={adapter_info.get('k', '?')}")
        else:
            print(f"  ✗ Failed: {adapter_path}")

    if not models_and_tokenizers:
        print("No adapters loaded successfully.")
        return {}

    # Create efficient evaluator
    config = EvaluationConfig(
        adapter_paths=adapter_paths,
        output_dir="quick_evaluation_results",
        max_samples=max_samples,
        device="cuda"
    )

    evaluator = EfficientTopKLoRAEvaluator(
        config, sample_latents=sample_latents)

    # Run evaluations
    print(f"\nRunning evaluations on {len(models_and_tokenizers)} adapters...")

    all_results = {}
    total_start_time = time.time()

    for i, (model, tokenizer, wrapped_modules) in enumerate(models_and_tokenizers):
        adapter_name = f"adapter_{i}"
        print(f"\n--- {adapter_name.upper()} ---")

        adapter_start_time = time.time()

        # Monosemanticity (fast)
        print("1. Monosemanticity analysis...")
        mono_results = evaluator.evaluate_monosemanticity(
            model, tokenizer, wrapped_modules)

        # Causal interventions (sampled)
        print("2. Causal interventions (sampled)...")
        causal_results = evaluator.evaluate_causal_interventions(
            model, tokenizer, wrapped_modules)

        # Cost analysis (fast)
        print("3. Cost analysis...")
        cost_results = evaluator.evaluate_cost(
            model, tokenizer, wrapped_modules)

        adapter_time = time.time() - adapter_start_time

        all_results[adapter_name] = {
            'monosemanticity': mono_results,
            'causal_interventions': causal_results,
            'cost_analysis': cost_results,
            'evaluation_time': adapter_time
        }

        print(f"   Completed in {adapter_time:.1f}s")

    # Stability analysis (if multiple adapters)
    if len(models_and_tokenizers) > 1:
        print("\n4. Cross-adapter stability...")
        stability_results = evaluator.evaluate_stability(
            [(model, wrapped_modules)
             for model, _, wrapped_modules in models_and_tokenizers]
        )
        all_results['stability'] = stability_results

    total_time = time.time() - total_start_time

    # Summary
    print(f"\n{'='*60}")
    print("QUICK EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")

    # Print key metrics
    for adapter_name in all_results:
        if not adapter_name.startswith('adapter_'):
            continue

        print(f"\n{adapter_name}:")

        # Monosemanticity summary
        mono = all_results[adapter_name].get('monosemanticity', {})
        if mono:
            avg_active_k = []
            avg_sparsity = []
            for layer_name, layer_results in mono.items():
                if isinstance(layer_results, dict) and 'error' not in layer_results:
                    avg_active_k.append(layer_results.get('mean_active_k', 0))
                    avg_sparsity.append(
                        layer_results.get('global_sparsity', 0))

            if avg_active_k:
                print(
                    f"  Mean active-k: {sum(avg_active_k)/len(avg_active_k):.2f}")
                print(
                    f"  Mean sparsity: {sum(avg_sparsity)/len(avg_sparsity):.3f}")

        # Cost summary
        cost = all_results[adapter_name].get('cost_analysis', {})
        if 'parameters' in cost:
            params = cost['parameters']
            print(
                f"  Compression ratio: {params.get('compression_ratio', 0):.4f}")

    # Stability summary
    if 'stability' in all_results:
        stability = all_results['stability'].get('overall_stability', {})
        if stability:
            print(
                f"\nCross-adapter similarity: {stability.get('overall_mean_similarity', 0):.3f}")

    # Save results
    evaluator.save_results(all_results)
    print(f"\nResults saved to: {config.output_dir}/")

    return all_results


def main():
    """Run quick evaluation"""

    adapter_paths = [
        "models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_110632_3030316f/final_adapter",
        "models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_111634_62b5fb0f/final_adapter",
        "models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_112006_3797c9bd/final_adapter"
    ]

    # Check if adapters exist
    existing_adapters = []
    for path in adapter_paths:
        if Path(path).exists():
            existing_adapters.append(path)
        else:
            print(f"Warning: {path} not found")

    if not existing_adapters:
        print("No adapters found!")
        return

    try:
        results = quick_evaluation_with_progress(
            existing_adapters,
            max_samples=20,      # Small for speed
            sample_latents=10    # Sample 10 latents per layer
        )

        if results:
            print("\n✓ Quick evaluation completed successfully!")
            print("\nFor full evaluation, run:")
            print("python run_comprehensive_evaluation.py --max_samples 500")

    except KeyboardInterrupt:
        print("\n✗ Evaluation interrupted.")
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SAEBench evaluation pipeline for TopK sparse LoRA adapters.
This script treats the TopK LoRA latent space as SAE features for interpretability analysis.
"""

from models import TopKLoRALinear
from evals import init_model_tokenizer
import os
import sys
import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


# SAEBench imports
try:
    import sae_bench
    from sae_bench.evals.core import eval_config as core_config, main as core_main
    from sae_bench.evals.sparse_probing import eval_config as sparse_config, main as sparse_main
    from sae_bench.evals.absorption import eval_config as absorption_config, main as absorption_main
    from sae_bench.evals.autointerp import eval_config as auto_interp_config, main as auto_interp_main
    from sae_bench.evals.ravel import eval_config as ravel_config, main as ravel_main
    from sae_bench.evals.scr_and_tpp import eval_config as scr_tpp_config, main as scr_tpp_main
    from sae_bench.evals.unlearning import eval_config as unlearning_config, main as unlearning_main
    SAEBENCH_AVAILABLE = True
except ImportError as e:
    print(f"SAEBench not found: {e}")
    print("Please install SAEBench: pip install sae-bench")
    SAEBENCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SAEBench expects these specific attribute names


@dataclass
class SAEConfig:
    """Config class to mimic SAE-Lens SAE configs"""
    d_in: int
    d_sae: int
    l1_coefficient: float = 0.0
    architecture: str = "topk_lora"
    activation_fn: str = "topk_mask"
    hook_point: str = ""
    hook_point_layer: int = 0
    normalize_sae_decoder: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False
    normalize_activations: str = "none"


@dataclass
class AdapterConfig:
    """Configuration for a TopK LoRA adapter"""
    name: str
    adapter_path: str
    base_model: str
    r: int
    k: int
    alpha: float
    layer_pattern: str = "layers"  # Pattern to identify target layers
    model_it_name: Optional[str] = None


@dataclass
class EvalConfig:
    """Configuration for SAEBench evaluation"""
    output_dir: str = "eval_results"
    device: str = "cuda"
    batch_size: int = 32
    max_samples: int = 1000
    dtype: str = "float16"
    cache_activations: bool = True
    run_core: bool = True
    run_sparse_probing: bool = True
    run_absorption: bool = True
    run_auto_interp: bool = False  # Requires OpenAI API key
    run_ravel: bool = True
    run_scr_tpp: bool = True
    run_unlearning: bool = False  # Requires WMDP-bio access


class TopKLoRASAEAdapter:
    """
    Adapter class to make TopKLoRALinear modules compatible with SAEBench.
    This treats the adapter's latent space (z) as SAE features.
    """

    def __init__(self, topk_module: TopKLoRALinear, hook_point: str, layer_idx: int):
        self.topk_module = topk_module
        self.hook_point = hook_point
        self.layer_idx = layer_idx
        self.d_in = topk_module.A.shape[1]  # Input dimension
        self.d_sae = topk_module.r  # Feature dimension (r)
        self.k = topk_module.k  # Sparsity level
        self.layer_name = topk_module.layer_name

        # Cache for intermediate activations
        self._last_z = None
        self._last_z_sparse = None

        # Create config object that SAEBench expects
        self.cfg = SAEConfig(
            d_in=self.d_in,
            d_sae=self.d_sae,
            hook_point=hook_point,
            hook_point_layer=layer_idx,
            architecture="topk_lora",
            activation_fn="topk_mask",
        )

        # SAEBench expects these attributes
        self.W_enc = self.topk_module.A  # Encoder weights
        self.W_dec = self.topk_module.B  # Decoder weights
        self.b_enc = torch.zeros(
            self.d_sae, device=self.W_enc.device, dtype=self.W_enc.dtype)
        self.b_dec = torch.zeros(
            self.d_in, device=self.W_dec.device, dtype=self.W_dec.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features.
        This is the core of treating LoRA latents as SAE features.
        """
        with torch.no_grad():
            # Project to latent space using LoRA A matrix
            A = self.topk_module.A.to(dtype=x.dtype, device=x.device)
            z = torch.nn.functional.linear(x, A)

            # Apply TopK masking if needed
            if self.k < self.topk_module.r:
                z_sparse = self.topk_module.topk(z)
            else:
                z_sparse = z

            # Cache for potential decode operations
            self._last_z = z.clone()
            self._last_z_sparse = z_sparse.clone()

            return z_sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.
        """
        with torch.no_grad():
            B = self.topk_module.B.to(dtype=z.dtype, device=z.device)
            # Reconstruct using LoRA B matrix (without scaling for pure reconstruction)
            reconstruction = torch.nn.functional.linear(z, B)
            return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (encode -> decode)"""
        z = self.encode(x)
        return self.decode(z)

    def get_activation_pattern(self) -> torch.Tensor:
        """Get the activation pattern of the last forward pass"""
        if self._last_z_sparse is not None:
            return (self._last_z_sparse != 0).float()
        return None

    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics"""
        if self._last_z_sparse is not None:
            total_features = self._last_z_sparse.numel()
            active_features = (self._last_z_sparse != 0).sum().item()
            return {
                # Per sample
                'l0': active_features / self._last_z_sparse.shape[-1],
                'sparsity_ratio': 1.0 - (active_features / total_features),
                'theoretical_k': self.k,
                # Per batch
                'actual_active': active_features / self._last_z_sparse.shape[0]
            }
        return {}

    def to(self, device_or_dtype):
        """Move the adapter to device/dtype"""
        if isinstance(device_or_dtype, (torch.device, str)):
            self.W_enc = self.W_enc.to(device_or_dtype)
            self.W_dec = self.W_dec.to(device_or_dtype)
            self.b_enc = self.b_enc.to(device_or_dtype)
            self.b_dec = self.b_dec.to(device_or_dtype)
        return self

    def train(self, mode: bool = True):
        """Set training mode"""
        return self

    def eval(self):
        """Set evaluation mode"""
        return self


def extract_topk_modules(model) -> Dict[str, TopKLoRALinear]:
    """Extract all TopKLoRALinear modules from the model"""
    topk_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinear):
            topk_modules[name] = module
    return topk_modules


def create_sae_adapters(topk_modules: Dict[str, TopKLoRALinear]) -> Dict[str, TopKLoRASAEAdapter]:
    """Create SAE adapters for each TopK module"""
    sae_adapters = {}
    for name, module in topk_modules.items():
        # Extract layer index from module name (e.g., "model.layers.10.mlp.gate_proj" -> 10)
        layer_idx = 0
        try:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
        except (ValueError, IndexError):
            logger.warning(
                f"Could not extract layer index from {name}, using 0")

        # Create hook point name
        # Adjust based on your model
        hook_point = f"blocks.{layer_idx}.hook_mlp_out"
        adapter = TopKLoRASAEAdapter(module, hook_point, layer_idx)
        sae_adapters[name] = adapter
    return sae_adapters


def run_evaluation_suite(
    adapter_config: AdapterConfig,
    eval_config: EvalConfig,
    target_layers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run the complete SAEBench evaluation suite on TopK LoRA adapters
    """
    if not SAEBENCH_AVAILABLE:
        raise ImportError(
            "SAEBench is not available. Please install it first.")

    logger.info(f"Starting evaluation for adapter: {adapter_config.name}")

    # Load model and tokenizer
    model_cfg = type('ModelConfig', (), {
        'adapter_checkpoint_dir': adapter_config.adapter_path,
        'base_model': adapter_config.base_model,
        'k': adapter_config.k,
        'model_it_name': adapter_config.model_it_name,
        'name': adapter_config.name
    })()

    model, tokenizer, wrapped_modules = init_model_tokenizer(
        model_cfg, auto_interp=True)
    model.to(eval_config.device)

    # Extract TopK modules
    topk_modules = extract_topk_modules(model)
    logger.info(f"Found {len(topk_modules)} TopK LoRA modules")

    # Filter by target layers if specified
    if target_layers:
        topk_modules = {k: v for k, v in topk_modules.items() if any(
            layer in k for layer in target_layers)}
        logger.info(f"Filtered to {len(topk_modules)} target modules")

    # Create SAE adapters
    sae_adapters = create_sae_adapters(topk_modules)

    # Results storage
    all_results = {
        'adapter_config': asdict(adapter_config),
        'eval_config': asdict(eval_config),
        'modules_evaluated': list(sae_adapters.keys()),
        'results': {}
    }

    # Run evaluations for each adapter
    for module_name, sae_adapter in sae_adapters.items():
        logger.info(f"Evaluating module: {module_name}")
        module_results = {}

        try:
            # Core evaluation (L0, Loss Recovered, etc.)
            if eval_config.run_core:
                logger.info("Running core evaluation...")
                try:
                    core_cfg = core_config.CoreEvalConfig(
                        n_eval_reconstruction_batches=min(
                            10, eval_config.max_samples // eval_config.batch_size),
                        n_eval_sparsity_variance_batches=min(
                            5, eval_config.max_samples // eval_config.batch_size // 2),
                    )
                    # Create output directory
                    output_dir = Path(eval_config.output_dir) / \
                        "core" / module_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    core_result = core_main.run_eval(
                        sae=sae_adapter,
                        model=model,
                        eval_config=core_cfg,
                        device=eval_config.device,
                        output_path=str(output_dir / "results.json")
                    )
                    module_results['core'] = core_result
                except Exception as e:
                    logger.error(f"Core evaluation failed: {e}")
                    module_results['core_error'] = str(e)

            # Sparse Probing
            if eval_config.run_sparse_probing:
                logger.info("Running sparse probing...")
                try:
                    sparse_cfg = sparse_config.SparseDecodingEvalConfig(
                        n_batches_to_sample_from=min(
                            10, eval_config.max_samples // eval_config.batch_size),
                        n_prompts_to_select=min(4096, eval_config.max_samples),
                    )
                    output_dir = Path(eval_config.output_dir) / \
                        "sparse_probing" / module_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    sparse_result = sparse_main.run_eval(
                        sae=sae_adapter,
                        model=model,
                        eval_config=sparse_cfg,
                        device=eval_config.device,
                        output_path=str(output_dir / "results.json")
                    )
                    module_results['sparse_probing'] = sparse_result
                except Exception as e:
                    logger.error(f"Sparse probing failed: {e}")
                    module_results['sparse_probing_error'] = str(e)

            # Feature Absorption
            if eval_config.run_absorption:
                logger.info("Running absorption evaluation...")
                try:
                    absorption_cfg = absorption_config.AbsorptionEvalConfig(
                        n_batches_to_sample_from=min(
                            10, eval_config.max_samples // eval_config.batch_size),
                        n_prompts_to_select=min(4096, eval_config.max_samples),
                    )
                    output_dir = Path(eval_config.output_dir) / \
                        "absorption" / module_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    absorption_result = absorption_main.run_eval(
                        sae=sae_adapter,
                        model=model,
                        eval_config=absorption_cfg,
                        device=eval_config.device,
                        output_path=str(output_dir / "results.json")
                    )
                    module_results['absorption'] = absorption_result
                except Exception as e:
                    logger.error(f"Absorption evaluation failed: {e}")
                    module_results['absorption_error'] = str(e)

            # RAVEL
            if eval_config.run_ravel:
                logger.info("Running RAVEL evaluation...")
                try:
                    ravel_cfg = ravel_config.RavelEvalConfig(
                        n_batches_to_sample_from=min(
                            10, eval_config.max_samples // eval_config.batch_size),
                        n_prompts_to_select=min(4096, eval_config.max_samples),
                    )
                    output_dir = Path(eval_config.output_dir) / \
                        "ravel" / module_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    ravel_result = ravel_main.run_eval(
                        sae=sae_adapter,
                        model=model,
                        eval_config=ravel_cfg,
                        device=eval_config.device,
                        output_path=str(output_dir / "results.json")
                    )
                    module_results['ravel'] = ravel_result
                except Exception as e:
                    logger.error(f"RAVEL evaluation failed: {e}")
                    module_results['ravel_error'] = str(e)

            # SCR and TPP (combined)
            if eval_config.run_scr_tpp:
                logger.info("Running SCR and TPP evaluation...")
                try:
                    scr_tpp_cfg = scr_tpp_config.ScrAndTppEvalConfig(
                        n_batches_to_sample_from=min(
                            10, eval_config.max_samples // eval_config.batch_size),
                        n_prompts_to_select=min(4096, eval_config.max_samples),
                    )
                    output_dir = Path(eval_config.output_dir) / \
                        "scr_tpp" / module_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    scr_tpp_result = scr_tpp_main.run_eval(
                        sae=sae_adapter,
                        model=model,
                        eval_config=scr_tpp_cfg,
                        device=eval_config.device,
                        output_path=str(output_dir / "results.json")
                    )
                    module_results['scr_tpp'] = scr_tpp_result
                except Exception as e:
                    logger.error(f"SCR/TPP evaluation failed: {e}")
                    module_results['scr_tpp_error'] = str(e)

            # AutoInterp (requires OpenAI API key)
            if eval_config.run_auto_interp and os.path.exists("openai_api_key.txt"):
                logger.info("Running AutoInterp evaluation...")
                try:
                    auto_interp_cfg = auto_interp_config.AutoInterpEvalConfig(
                        # Limit for cost
                        n_features=min(100, sae_adapter.d_sae),
                        n_examples_per_feature=5,
                    )
                    output_dir = Path(eval_config.output_dir) / \
                        "autointerp" / module_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    auto_interp_result = auto_interp_main.run_eval(
                        sae=sae_adapter,
                        model=model,
                        eval_config=auto_interp_cfg,
                        device=eval_config.device,
                        output_path=str(output_dir / "results.json")
                    )
                    module_results['auto_interp'] = auto_interp_result
                except Exception as e:
                    logger.error(f"AutoInterp evaluation failed: {e}")
                    module_results['auto_interp_error'] = str(e)

            # Unlearning (requires WMDP-bio access)
            if eval_config.run_unlearning:
                logger.info("Running unlearning evaluation...")
                try:
                    unlearning_cfg = unlearning_config.UnlearningEvalConfig(
                        n_batches_to_sample_from=min(
                            10, eval_config.max_samples // eval_config.batch_size),
                    )
                    output_dir = Path(eval_config.output_dir) / \
                        "unlearning" / module_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    unlearning_result = unlearning_main.run_eval(
                        sae=sae_adapter,
                        model=model,
                        eval_config=unlearning_cfg,
                        device=eval_config.device,
                        output_path=str(output_dir / "results.json")
                    )
                    module_results['unlearning'] = unlearning_result
                except Exception as e:
                    logger.error(f"Unlearning evaluation failed: {e}")
                    module_results['unlearning_error'] = str(e)

            # Add adapter-specific metrics
            module_results['adapter_metrics'] = {
                'r': sae_adapter.topk_module.r,
                'k': sae_adapter.topk_module.k,
                'theoretical_sparsity': 1.0 - (sae_adapter.k / sae_adapter.topk_module.r),
                'scale': sae_adapter.topk_module.scale,
                'layer_name': sae_adapter.layer_name,
                'd_in': sae_adapter.d_in,
                'd_sae': sae_adapter.d_sae,
            }

        except Exception as e:
            logger.error(f"Error evaluating {module_name}: {str(e)}")
            module_results['error'] = str(e)

        all_results['results'][module_name] = module_results

    # Save results
    output_path = Path(eval_config.output_dir) / \
        f"{adapter_config.name}_sae_bench_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_path}")
    return all_results


def main():
    """Main evaluation script"""

    # Define your three adapters
    adapters = [
        AdapterConfig(
            name="topk_dpo_3030316f",
            adapter_path="models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_110632_3030316f/final_adapter",
            base_model="/scratch/network/ssd/marek/lora_interp/cache/tempartefacts/google/gemma-2-2b_sft",
            r=1024,  # Adjust these values based on your actual configs
            k=8,
            alpha=2048,
            model_it_name="google/gemma-2-2b-it"
        ),
        AdapterConfig(
            name="topk_dpo_62b5fb0f",
            adapter_path="models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_111634_62b5fb0f/final_adapter",
            base_model="/scratch/network/ssd/marek/lora_interp/cache/tempartefacts/google/gemma-2-2b_sft",
            r=512,
            k=4,
            alpha=1024,
            model_it_name="google/gemma-2-2b-it"
        ),
        AdapterConfig(
            name="topk_dpo_3797c9bd",
            adapter_path="models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_112006_3797c9bd/final_adapter",
            base_model="/scratch/network/ssd/marek/lora_interp/cache/tempartefacts/google/gemma-2-2b_sft",
            r=512,
            k=2,
            alpha=1024,
            model_it_name="google/gemma-2-2b-it"
        ),
    ]

    # Evaluation configuration
    eval_cfg = EvalConfig(
        output_dir="sae_bench_results",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=16,  # Adjust based on your VRAM
        max_samples=500,  # Start smaller for testing
        run_auto_interp=False,  # Enable if you have OpenAI API key
        run_unlearning=False,   # Enable if you have WMDP-bio access
    )

    # Target specific layers (optional - comment out to evaluate all)
    target_layers = [
        "model.layers.10",  # Middle layers often most interesting
        "model.layers.11",
        "model.layers.12",
    ]

    # Run evaluations
    all_adapter_results = {}

    for adapter_config in adapters:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating adapter: {adapter_config.name}")
        logger.info(f"{'='*50}")

        try:
            results = run_evaluation_suite(
                adapter_config=adapter_config,
                eval_config=eval_cfg,
                target_layers=target_layers
            )
            all_adapter_results[adapter_config.name] = results

        except Exception as e:
            logger.error(f"Failed to evaluate {adapter_config.name}: {str(e)}")
            all_adapter_results[adapter_config.name] = {"error": str(e)}

    # Save combined results
    combined_path = Path(eval_cfg.output_dir) / "combined_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_adapter_results, f, indent=2, default=str)

    logger.info(f"\nAll results saved to: {combined_path}")

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    for adapter_name, results in all_adapter_results.items():
        print(f"\nAdapter: {adapter_name}")
        if 'error' in results:
            print(f"  Status: FAILED - {results['error']}")
        else:
            modules_count = len(results.get('results', {}))
            print(f"  Status: SUCCESS")
            print(f"  Modules evaluated: {modules_count}")

            # Show some key metrics if available
            for module_name, module_results in results.get('results', {}).items():
                print(f"    {module_name}:")

                # Check if we have valid results (not just errors)
                has_results = any(
                    key for key in module_results.keys() if not key.endswith('_error'))
                if has_results:
                    if 'adapter_metrics' in module_results:
                        am = module_results['adapter_metrics']
                        print(
                            f"      r={am.get('r')}, k={am.get('k')}, sparsity={am.get('theoretical_sparsity', 0):.3f}")

                    # Count successful evaluations
                    success_count = sum(1 for key in module_results.keys()
                                        if not key.endswith('_error') and key != 'adapter_metrics')
                    error_count = sum(
                        1 for key in module_results.keys() if key.endswith('_error'))
                    print(
                        f"      Successful evaluations: {success_count}, Errors: {error_count}")
                else:
                    print("      All evaluations failed")


if __name__ == "__main__":
    main()

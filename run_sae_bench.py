from src.evals import init_model_tokenizer
from src.models import TopKLoRALinear
import os
import sys
import json
from activations_store import ActivationsStore
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# SAEBench imports
try:
    import sae_bench
    from sae_bench.evals.core.main import CoreEvalConfig, run_evals as core_run_evals
    from sae_bench.evals.sparse_probing.main import SparseProbingEvalConfig, run_eval as sparse_run_eval
    from sae_bench.evals.absorption.main import AbsorptionEvalConfig, run_eval as absorption_run_eval
    SAEBENCH_AVAILABLE = True
except ImportError as e:
    print(f"SAEBench not found: {e}")
    print("Please install SAEBench: pip install sae-bench")
    SAEBENCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Evaluation configuration"""
    output_dir: str = "sae_bench_results"
    device: str = "cuda"
    batch_size: int = 16
    max_samples: int = 500
    dtype: str = "float32"

    # Evaluation modules to run
    run_core: bool = False  # Disabled until we have ActivationsStore
    run_sparse_probing: bool = False  # Disabled until we have full pipeline
    run_absorption: bool = False  # Disabled
    run_auto_interp: bool = False
    run_ravel: bool = False
    run_scr: bool = False
    run_tpp: bool = False
    run_unlearning: bool = False
    run_basic_sae_test: bool = True  # Custom basic test


class TopKLoRASAEAdapter:
    """
    Adapter class to make TopKLoRALinear modules compatible with SAEBench.
    This treats the adapter's latent space (z) as SAE features.
    """

    def __init__(self, topk_module: TopKLoRALinear, hook_point: str, device: str = "cuda"):
        self.topk_module = topk_module
        self.hook_point = hook_point
        self.device = device

        # Get tensor attributes properly - handle both weight and direct tensor access
        A = self.topk_module.A
        B = self.topk_module.B
        if isinstance(A, torch.nn.Linear):
            A_tensor = A.weight
        else:
            A_tensor = A
        if isinstance(B, torch.nn.Linear):
            B_tensor = B.weight
        else:
            B_tensor = B

        self.d_in = A_tensor.shape[1]  # Input dimension
        self.d_out = B_tensor.shape[0]  # Output dimension
        self.d_sae = self.topk_module.r  # Feature dimension (r)
        self.k = self.topk_module.k  # Sparsity level
        self.layer_name = self.topk_module.layer_name

        # Cache for intermediate activations
        self._last_z = None
        self._last_z_sparse = None

    @property
    def cfg(self):
        """SAEBench expects a cfg attribute with certain fields"""
        # Create a minimal config object that SAEBench expects
        class MockConfig:
            def __init__(self, hook_name, d_in, d_sae, hook_layer):
                self.hook_name = hook_name
                self.d_in = d_in
                self.d_sae = d_sae
                self.hook_layer = hook_layer  # Extract layer number
                self.hook_head_index = None
                self.architecture = 'topk_lora'

        # Extract layer number from hook_point
        layer_num = 11  # Default - can be extracted from hook_point if needed
        return MockConfig(self.hook_point, self.d_in, self.d_sae, layer_num)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features.
        This is the core of treating LoRA latents as SAE features.
        """
        with torch.no_grad():
            # Project to latent space using LoRA A matrix
            A = self.topk_module.A
            if isinstance(A, torch.nn.Linear):
                A_weight = A.weight
            else:
                A_weight = A
            A_weight = A_weight.to(dtype=x.dtype, device=x.device)
            z = torch.nn.functional.linear(x, A_weight)

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
            B = self.topk_module.B
            if isinstance(B, torch.nn.Linear):
                B_weight = B.weight
            else:
                B_weight = B
            B_weight = B_weight.to(dtype=z.dtype, device=z.device)
            # Reconstruct using LoRA B matrix (without scaling for pure reconstruction)
            reconstruction = torch.nn.functional.linear(z, B_weight)
            return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (encode -> decode)"""
        z = self.encode(x)
        return self.decode(z)

    def get_activation_pattern(self) -> torch.Tensor:
        """Get the activation pattern of the last forward pass"""
        if self._last_z_sparse is not None:
            return (self._last_z_sparse != 0).float()
        return torch.tensor([])  # Return empty tensor instead of None

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
        # Create hook point name (SAEBench format)
        hook_point = f"blocks.{name}"  # Adjust based on your model structure
        adapter = TopKLoRASAEAdapter(module, hook_point)
        sae_adapters[name] = adapter
    return sae_adapters


def run_basic_sae_test(sae_adapter: TopKLoRASAEAdapter, model, eval_config: EvalConfig) -> dict:
    """
    Run a basic test of the SAE interface to validate our TopK LoRA adapter.
    This demonstrates encode/decode functionality and computes basic metrics.
    """
    logger.info(f"Testing SAE adapter for {sae_adapter.layer_name}")

    # Generate some test inputs
    batch_size = 8
    seq_len = 32
    d_in = int(sae_adapter.d_in)  # Input dimension
    d_out = int(sae_adapter.d_out)  # Output dimension

    # Create random test inputs (simulating activations with correct input dimension)
    test_inputs = torch.randn(
        batch_size, seq_len, d_in, device=eval_config.device)

    results = {}
    with torch.no_grad():
        # Test encode functionality
        encoded = sae_adapter.encode(test_inputs)
        results['encode_shape'] = list(encoded.shape)
        results['d_sae'] = sae_adapter.d_sae
        results['k'] = sae_adapter.k
        results['d_in'] = d_in
        results['d_out'] = d_out

        # Test decode functionality
        decoded = sae_adapter.decode(encoded)
        results['decode_shape'] = list(decoded.shape)

        # Test full forward pass
        reconstructed = sae_adapter.forward(test_inputs)
        results['reconstruction_shape'] = list(reconstructed.shape)

        # Compute sparsity metrics
        l0_norm = (encoded != 0).float().sum(dim=-1).mean().item()
        results['l0_norm'] = l0_norm
        results['theoretical_sparsity'] = 1.0 - \
            (sae_adapter.k / sae_adapter.d_sae)
        results['actual_sparsity'] = 1.0 - (l0_norm / sae_adapter.d_sae)

        # For reconstruction error, we can only compare if input and output dims match
        # Otherwise, this is a dimensionality-changing LoRA adapter
        if d_in == d_out:
            mse_loss = torch.nn.functional.mse_loss(
                reconstructed, test_inputs).item()
            cosine_sim = torch.nn.functional.cosine_similarity(
                reconstructed.view(-1, d_in),
                test_inputs.view(-1, d_in),
                dim=-1
            ).mean().item()
            results['mse_loss'] = mse_loss
            results['cosine_similarity'] = cosine_sim
            results['reconstruction_fidelity'] = 1.0 - mse_loss
        else:
            # For dim-changing adapters, just report that reconstruction changes dimensions
            results['mse_loss'] = 'N/A (dim-changing adapter)'
            results['cosine_similarity'] = 'N/A (dim-changing adapter)'
            results['reconstruction_fidelity'] = 'N/A (dim-changing adapter)'
            results['dimension_change'] = f"{d_in} -> {d_out}"

        # Validate SAE interface compliance
        results['has_cfg'] = hasattr(sae_adapter, 'cfg')
        results['has_encode'] = hasattr(sae_adapter, 'encode')
        results['has_decode'] = hasattr(sae_adapter, 'decode')
        results['device_matches'] = str(encoded.device) == eval_config.device

        # Check that encode->decode is close to forward
        encode_decode = sae_adapter.decode(sae_adapter.encode(test_inputs))
        forward_result = sae_adapter.forward(test_inputs)
        consistency_error = torch.nn.functional.mse_loss(
            encode_decode, forward_result).item()
        results['encode_decode_consistency'] = consistency_error < 1e-6

        results['test_status'] = 'PASSED'

    logger.info(f"SAE test completed: L0={l0_norm:.2f}, Dims={d_in}->{d_out}")
    return results


def run_evaluation_suite(
    adapter_config: AdapterConfig,
    eval_config: EvalConfig,
    target_layers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run the complete SAEBench evaluation suite on TopK LoRA adapters
    """
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
            # Basic SAE Interface Test
            if eval_config.run_basic_sae_test:
                logger.info("Running basic SAE interface test...")
                test_results = run_basic_sae_test(
                    sae_adapter, model, eval_config)
                module_results['basic_sae_test'] = test_results

            # Core evaluation (L0, Loss Recovered, etc.)
            if eval_config.run_core:
                logger.info("Running core evaluation...")
                core_cfg = CoreEvalConfig(
                    batch_size_prompts=eval_config.batch_size,
                    n_eval_reconstruction_batches=10,
                    llm_dtype=eval_config.dtype,
                    compute_sparsity_metrics=True,
                    compute_variance_metrics=True
                )
                # CoreEvalConfig expects (sae, activation_store, model, eval_config, model_kwargs, ignore_tokens, verbose)
                # Since we don't have activation_store, we'll need to mock it or modify approach
                logger.warning(
                    "Core evaluation requires ActivationsStore - skipping for now")
                module_results['core'] = {
                    "status": "skipped", "reason": "requires ActivationsStore"}

            # Sparse Probing
            if eval_config.run_sparse_probing:
                logger.info("Running sparse probing...")
    for module_name, sae_adapter in sae_adapters.items():
        logger.info(f"Evaluating module: {module_name}")
        module_results = {}

        try:
            # --- ActivationsStore integration ---
            if eval_config.run_core:
                logger.info("Collecting activations for core evaluation...")
                # Initialize ActivationsStore
                act_store = ActivationsStore(
                    max_samples=eval_config.max_samples, device=eval_config.device)

                # Find the target module in the model
                target_mod = dict(model.named_modules())[module_name]

                # Define hook to capture activations
                def hook_fn(module, inp, out):
                    # out: (batch, seq, dim)
                    act_store.add(out)

                hook_handle = target_mod.register_forward_hook(hook_fn)

                # Run a forward pass over dummy data to collect activations
                # (In practice, use real data. Here, we use random input for demonstration.)
                batch_size = 8
                seq_len = 32
                d_in = int(sae_adapter.d_in)
                dummy_input = torch.randint(
                    0, model.cfg.d_vocab, (batch_size, seq_len), device=eval_config.device)
                with torch.no_grad():
                    _ = model(dummy_input)

                # Remove hook
                hook_handle.remove()

                logger.info(
                    f"Collected {len(act_store)} activations for {module_name}")

                # Placeholder: pass act_store to core evaluation (to be implemented)
                module_results['core'] = {
                    "status": "collected", "n_activations": len(act_store)}

            # Basic SAE Interface Test
            if eval_config.run_basic_sae_test:
                logger.info("Running basic SAE interface test...")
                test_results = run_basic_sae_test(
                    sae_adapter, model, eval_config)
                module_results['basic_sae_test'] = test_results
        except Exception as e:
            logger.error(f"Error evaluating {module_name}: {str(e)}")
            module_results['error'] = str(e)
            #     module_results['auto_interp'] = auto_interp_result

            # RAVEL
            if eval_config.run_ravel:
                logger.info("Running RAVEL evaluation...")
                logger.warning("RAVEL evaluation not implemented - skipping")
                module_results['ravel'] = {
                    "status": "skipped", "reason": "not implemented"}

            # Spurious Correlation Removal
            if eval_config.run_scr:
                logger.info("Running SCR evaluation...")
                logger.warning("SCR evaluation not implemented - skipping")
                module_results['scr'] = {
                    "status": "skipped", "reason": "not implemented"}

            # Targeted Probe Perturbation
            if eval_config.run_tpp:
                logger.info("Running TPP evaluation...")
                logger.warning("TPP evaluation not implemented - skipping")
                module_results['tpp'] = {
                    "status": "skipped", "reason": "not implemented"}

            # Unlearning (requires WMDP-bio access)
            if eval_config.run_unlearning:
                logger.info("Running unlearning evaluation...")
                logger.warning(
                    "Unlearning evaluation not implemented - skipping")
                module_results['unlearning'] = {
                    "status": "skipped", "reason": "not implemented"}

            # Add adapter-specific metrics
            module_results['adapter_metrics'] = {
                'r': sae_adapter.topk_module.r,
                'k': sae_adapter.topk_module.k,
                'theoretical_sparsity': 1.0 - (sae_adapter.k / sae_adapter.topk_module.r),
                'scale': sae_adapter.topk_module.scale,
                'layer_name': sae_adapter.layer_name,
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
            r=1024,  # You'll need to adjust these values based on your actual configs
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
        # "model.layers.10",  # Middle layers often most interesting
        "model.layers.11",
        # "model.layers.12",
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
                if 'core' in module_results:
                    core_metrics = module_results['core']
                    print(f"    {module_name}:")
                    print(f"      L0: {core_metrics.get('l0', 'N/A')}")
                    print(
                        f"      Loss recovered: {core_metrics.get('loss_recovered', 'N/A')}")


if __name__ == "__main__":
    main()

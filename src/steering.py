"""
Steering functionality for TopKLoRALinearSTE adapters.

This module provides a friendly API to enable/disable specific features (latents)
in trained TopKLoRALinearSTE adapters during inference, enabling fine-grained
control over model behavior.

Example usage:
    from src.steering import steer_features, FeatureSteeringContext
    
    # Define features to steer
    feature_dict = {
        "base_model.model.model.layers.11.self_attn.q_proj.topk": [
            (217, "enable"),   # Enable feature 217
            (45, "disable"),   # Disable feature 45
        ],
        "base_model.model.model.layers.11.mlp.down_proj.topk": [
            (128, "enable"),
        ]
    }
    
    # Option 1: Context manager (recommended)
    with FeatureSteeringContext(model, feature_dict):
        outputs = model.generate(...)
    
    # Option 2: Manual control
    hooks = steer_features(model, feature_dict)
    outputs = model.generate(...)
    remove_steering_hooks(hooks)
    
    # Option 3: Isolate a single latent (enable only one, ablate all others)
    feature_dict_isolate = {
        "base_model.model.model.layers.11.self_attn.q_proj.topk": [
            (344, "isolate"),  # Enable ONLY feature 344, disable all others
        ]
    }
    with FeatureSteeringContext(model, feature_dict_isolate, amplification=10.0):
        outputs = model.generate(...)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from contextlib import contextmanager

from src.models import TopKLoRALinearSTE

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class FeatureSteerer:
    """
    Hook handler for steering specific features in TopKLoRALinearSTE layers.

    Supports three modes:
    - "enable": Force specific latent(s) to be active (gate=1)
    - "disable": Force specific latent(s) to be inactive (gate=0)
    - "isolate": Enable ONLY the specified latent(s), ablate all others
    """

    def __init__(self, feature_indices: List[int], effects: List[str],
                 amplification: float = 1.0):
        """
        Args:
            feature_indices: List of feature/latent indices to steer
            effects: List of effects ("enable", "disable", or "isolate") for each feature
            amplification: Multiplier for enabled/isolated features (default 1.0, try 5.0-10.0 for stronger effect)
        """
        self.feature_indices = feature_indices
        self.effects = effects
        self.amplification = amplification
        self._collected_stats = []  # For tracking steering magnitudes
        self.original_forward = None  # Will store the original forward method

        # Validate inputs
        assert len(feature_indices) == len(effects), \
            "feature_indices and effects must have same length"
        for effect in effects:
            assert effect in ["enable", "disable", "isolate"], \
                f"Invalid effect: {effect}. Must be 'enable', 'disable', or 'isolate'"

        # Check for isolate mode
        self.has_isolate = "isolate" in effects
        if self.has_isolate:
            # Validate: if isolate is used, it should be the only effect for this layer
            if len(effects) > 1 and any(e != "isolate" for e in effects):
                logging.warning(
                    "Using 'isolate' mode with other effects. Note: isolate will ablate ALL other latents, "
                    "so 'enable'/'disable' on other latents will have no effect."
                )

    def create_steered_forward(self, module: TopKLoRALinearSTE):
        """
        Creates a replacement forward function that applies steering.
        
        This replaces the module's forward method to compute:
        base_output + steering_vector (bypassing normal LoRA)
        
        Args:
            module: The TopKLoRALinearSTE module to steer
            
        Returns:
            A forward function that applies steering
        """
        # Store reference to self for closure
        steerer = self
        
        def steered_forward(x: torch.Tensor) -> torch.Tensor:
            """
            Steered forward pass that computes base + steering vector only.

            This function computes a steering vector by:
            1. Computing latent activations through LoRA's A matrix
            2. Selecting which latents to steer (enable/disable/isolate)
            3. Applying the selected latents through LoRA's B matrix
            4. Adding the result to the BASE MODEL output (not the full LoRA output)

            The key difference from standard LoRA: we only add our steering vector,
            ignoring the normal LoRA contribution.

            Args:
                x: Input tensor

            Returns:
                Base model output + steering vector
            """
            with torch.no_grad():
                # Get LoRA matrices
                A_weight = module.A_module.weight
                B_weight = module.B_module.weight

                # Apply dropout if present
                x_lora = module.dropout(x)

                # Compute latent activations: x @ A^T -> latents
                z_pre = torch.nn.functional.linear(x_lora, A_weight)
                if module.relu_latents:
                    z = torch.nn.functional.relu(z_pre)
                else:
                    z = z_pre

                # Create steering mask: which latents to activate
                # Start with all zeros (no steering by default)
                steering_mask = torch.zeros_like(z)

                # Handle isolate mode: we're starting from zero, only enable selected
                if steerer.has_isolate:
                    # All latents already zero, just enable selected ones below
                    pass

                # Apply steering by setting mask values
                for idx, effect in zip(steerer.feature_indices, steerer.effects):
                    if idx >= z.shape[-1]:
                        continue

                    if effect == "enable" or effect == "isolate":
                        # Enable this latent
                        steering_mask[..., idx] = 1.0

                        # AMPLIFY the activation if amplification != 1.0
                        if steerer.amplification != 1.0:
                            z[..., idx] = z[..., idx] * steerer.amplification

                    elif effect == "disable":
                        # Keep at zero (already is)
                        steering_mask[..., idx] = 0.0

                # Compute steering vector: (masked latents) @ B^T
                # Only the selected latents contribute to the steering
                steering_vector = torch.nn.functional.linear(
                    z * steering_mask, B_weight) * module.scale

                # Compute base output (without any LoRA)
                base_out = module.base_layer(x)

                # Return: base model + steering vector ONLY
                # (no normal LoRA contribution)
                return base_out + steering_vector
        
        return steered_forward

    def hook_fn(self, module: TopKLoRALinearSTE, input: Tuple[torch.Tensor],
                output: torch.Tensor) -> torch.Tensor:
        """
        DEPRECATED: This hook-based approach doesn't work because PyTorch 
        forward hooks cannot modify outputs (return value is ignored).
        
        Kept for reference but not used. See create_steered_forward() instead.

        This hook computes a steering vector by:
        1. Computing latent activations through LoRA's A matrix
        2. Selecting which latents to steer (enable/disable/isolate)
        3. Applying the selected latents through LoRA's B matrix
        4. Adding the result to the BASE MODEL output (not the full LoRA output)

        The key difference from standard LoRA: we only add our steering vector,
        ignoring the normal LoRA contribution.

        Args:
            module: The TopKLoRALinearSTE module
            input: Input tuple to the module
            output: Output tensor from the module (ignored - we recompute base)

        Returns:
            Base model output + steering vector
        """
        x = input[0]

        with torch.no_grad():
            # Get LoRA matrices
            A_weight = module.A_module.weight
            B_weight = module.B_module.weight

            # Apply dropout if present
            x_lora = module.dropout(x)

            # Compute latent activations: x @ A^T -> latents
            z_pre = torch.nn.functional.linear(x_lora, A_weight)
            if module.relu_latents:
                z = torch.nn.functional.relu(z_pre)
            else:
                z = z_pre

            # Create steering mask: which latents to activate
            # Start with all zeros (no steering by default)
            steering_mask = torch.zeros_like(z)

            # Handle isolate mode: we're starting from zero, only enable selected
            if self.has_isolate:
                # All latents already zero, just enable selected ones below
                pass

            # Apply steering by setting mask values
            for idx, effect in zip(self.feature_indices, self.effects):
                if idx >= z.shape[-1]:
                    continue

                if effect == "enable" or effect == "isolate":
                    # Enable this latent
                    steering_mask[..., idx] = 1.0

                    # AMPLIFY the activation if amplification != 1.0
                    if self.amplification != 1.0:
                        z[..., idx] = z[..., idx] * self.amplification

                elif effect == "disable":
                    # Keep at zero (already is)
                    steering_mask[..., idx] = 0.0

            # Compute steering vector: (masked latents) @ B^T
            # Only the selected latents contribute to the steering
            steering_vector = torch.nn.functional.linear(
                z * steering_mask, B_weight) * module.scale

            # Compute base output (without any LoRA)
            base_out = module.base_layer(x)

            # Collect magnitude statistics
            try:
                base_norm = base_out.norm(dim=-1).mean().item()
                steering_norm = steering_vector.norm(dim=-1).mean().item()
                ratio = steering_norm / (base_norm + 1e-8)

                self._collected_stats.append({
                    'base_norm': base_norm,
                    'lora_norm': steering_norm,  # Keep name for compatibility
                    'ratio': ratio,
                    'amplification': self.amplification
                })
            except Exception:
                pass

            # Return: base model + steering vector ONLY
            # (no normal LoRA contribution)
            return base_out + steering_vector


def steer_features(
    model: nn.Module,
    feature_dict: Dict[str, List[Tuple[int, str]]],
    verbose: bool = True,
    amplification: float = 1.0,
    wrapped_modules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply feature steering to a model with TopKLoRALinearSTE adapters.

    This function replaces the forward methods on specified adapter layers to enable,
    disable, or isolate specific features during inference.

    Args:
        model: PyTorch model with TopKLoRALinearSTE adapters loaded
        feature_dict: Dictionary mapping adapter names to list of (feature_num, effect) tuples
                     where effect is "enable", "disable", or "isolate"
                     Example: {
                         "base_model.model.model.layers.11.self_attn.q_proj.topk": [
                             (217, "enable"),   # Enable feature 217, leave others as-is
                             (45, "disable")    # Disable feature 45, leave others as-is
                         ]
                     }

                     For isolate mode (enable ONLY one feature, ablate all others):
                     {
                         "base_model.model.model.layers.11.self_attn.q_proj.topk": [
                             (344, "isolate")   # Enable ONLY 344, set all other gates to 0
                         ]
                     }
        verbose: If True, log information about applied steering
        amplification: Multiplier for enabled/isolated features (default 1.0). 
                      Try 5.0-10.0 for stronger steering effect.
                      Only affects "enable" and "isolate", not "disable".
        wrapped_modules: Optional dictionary of wrapped TopK modules from init_model_tokenizer_fixed.
                        If provided, will use these instead of scanning model.named_modules().

    Returns:
        Dictionary containing:
            - "modules": List of modules that had their forward methods replaced
            - "steerers": Dictionary mapping adapter names to FeatureSteerer objects
            - "applied_count": Number of successfully applied steering rules

    Usage:
        steering_info = steer_features(model, feature_dict, wrapped_modules=wrapped_modules)
        # ... run inference ...
        restore_forward_methods(steering_info["modules"], steering_info["steerers"])
    """
    steered_modules = []
    steerers = {}
    applied_count = 0

    # Find all TopKLoRALinearSTE modules in the model
    # Use wrapped_modules if provided (from eval pipeline), otherwise scan model
    if wrapped_modules is not None:
        available_adapters = wrapped_modules
        if verbose:
            logging.info(
                f"Using provided wrapped_modules: {len(available_adapters)} adapters")
    else:
        available_adapters = {}
        for name, module in model.named_modules():
            if isinstance(module, TopKLoRALinearSTE):
                available_adapters[name] = module

    if verbose:
        logging.info(
            f"Found {len(available_adapters)} TopKLoRALinearSTE adapters in model")

    # Apply steering to requested adapters
    for adapter_name, feature_specs in feature_dict.items():
        # Try to find the adapter (with flexible matching)
        target_module = None
        matched_name = None

        # Exact match first
        if adapter_name in available_adapters:
            target_module = available_adapters[adapter_name]
            matched_name = adapter_name
        else:
            # Try partial matching (e.g., user provides shortened name)
            for avail_name, module in available_adapters.items():
                if adapter_name in avail_name or avail_name.endswith(adapter_name):
                    target_module = module
                    matched_name = avail_name
                    break

        if target_module is None:
            logging.warning(
                f"Adapter '{adapter_name}' not found in model. Available adapters:\n"
                f"{list(available_adapters.keys())[:5]}..."
            )
            continue

        # Extract feature indices and effects
        feature_indices = [spec[0] for spec in feature_specs]
        effects = [spec[1] for spec in feature_specs]

        # Create steerer and replace forward method
        steerer = FeatureSteerer(
            feature_indices, effects, amplification=amplification)
        
        # Store original forward method
        steerer.original_forward = target_module.forward
        
        # Replace with steered forward
        target_module.forward = steerer.create_steered_forward(target_module)

        steered_modules.append(target_module)
        steerers[matched_name] = steerer
        applied_count += len(feature_specs)

        if verbose:
            logging.info(
                f"Applied steering to '{matched_name}': "
                f"{len(feature_specs)} features ({dict(zip(feature_indices, effects))})"
            )

    if verbose:
        logging.info(
            f"✅ Steering enabled: {applied_count} feature rules applied "
            f"across {len(steerers)} adapters"
        )

    return {
        "modules": steered_modules,
        "steerers": steerers,
        "applied_count": applied_count
    }


def restore_forward_methods(modules: List[nn.Module], steerers: Dict[str, Any]) -> None:
    """
    Restore original forward methods for steered modules.

    Args:
        modules: List of modules that had their forward methods replaced
        steerers: Dictionary of steerers (contains original_forward references)
    """
    restored_count = 0
    for module in modules:
        # Find the steerer that has the original forward for this module
        for steerer in steerers.values():
            if steerer.original_forward is not None:
                module.forward = steerer.original_forward
                steerer.original_forward = None
                restored_count += 1
                break
    
    logging.info(f"Restored {restored_count} original forward methods")


def remove_steering_hooks(hooks: List[Any]) -> None:
    """
    DEPRECATED: Remove previously registered steering hooks.
    
    This function is kept for backward compatibility but is no longer used
    since we now replace forward methods instead of using hooks.

    Args:
        hooks: List of hook handles returned by steer_features()
    """
    if hooks:
        for hook in hooks:
            hook.remove()
        logging.info(f"Removed {len(hooks)} steering hooks")


def get_steering_statistics(steerers: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Retrieve statistics about steering magnitudes from steerer objects.

    This shows the relative magnitude of the steering intervention compared to
    base model activations, helping you understand the strength of the effect.

    Args:
        steerers: Dictionary of steerers returned by steer_features()

    Returns:
        Dictionary mapping adapter names to statistics (mean base norm, lora norm, ratio)
    """
    stats = {}

    for adapter_name, steerer in steerers.items():
        # Check if any modules have collected stats
        if hasattr(steerer, '_collected_stats') and steerer._collected_stats:
            base_norms = [s['base_norm'] for s in steerer._collected_stats]
            lora_norms = [s['lora_norm'] for s in steerer._collected_stats]
            ratios = [s['ratio'] for s in steerer._collected_stats]

            stats[adapter_name] = {
                'mean_base_norm': sum(base_norms) / len(base_norms) if base_norms else 0.0,
                'mean_lora_norm': sum(lora_norms) / len(lora_norms) if lora_norms else 0.0,
                'mean_ratio': sum(ratios) / len(ratios) if ratios else 0.0,
                'max_ratio': max(ratios) if ratios else 0.0,
                'num_samples': len(base_norms)
            }

    return stats


def log_steering_statistics(steerers: Dict[str, Any], verbose: bool = True) -> None:
    """
    Log statistics about steering magnitudes.

    Args:
        steerers: Dictionary of steerers returned by steer_features()
        verbose: If True, print detailed statistics
    """
    stats = get_steering_statistics(steerers)

    if not stats:
        if verbose:
            logging.info(
                "No steering statistics collected (run inference to collect stats)")
        return

    if verbose:
        logging.info("\n" + "="*80)
        logging.info("Steering Magnitude Statistics")
        logging.info("="*80)

        for adapter_name, adapter_stats in stats.items():
            logging.info(f"\n{adapter_name}:")
            logging.info(
                f"  Mean base activation norm: {adapter_stats['mean_base_norm']:.4f}")
            logging.info(
                f"  Mean LoRA steering norm:   {adapter_stats['mean_lora_norm']:.4f}")
            logging.info(
                f"  Mean ratio (LoRA/base):    {adapter_stats['mean_ratio']:.4f}")
            logging.info(
                f"  Max ratio (LoRA/base):     {adapter_stats['max_ratio']:.4f}")
            logging.info(
                f"  Samples:                   {adapter_stats['num_samples']}")

        logging.info("\n" + "="*80 + "\n")


@contextmanager
def FeatureSteeringContext(
    model: nn.Module,
    feature_dict: Dict[str, List[Tuple[int, str]]],
    verbose: bool = True,
    amplification: float = 1.0,
    wrapped_modules: Optional[Dict[str, Any]] = None
):
    """
    Context manager for feature steering (recommended usage pattern).

    Automatically handles forward method replacement and restoration.

    Args:
        model: PyTorch model with TopKLoRALinearSTE adapters loaded
        feature_dict: Dictionary mapping adapter names to list of (feature_num, effect) tuples
                     where effect is "enable", "disable", or "isolate"
        verbose: If True, log information about applied steering
        amplification: Multiplier for enabled/isolated features (default 1.0). 
                      Try 5.0-10.0 for stronger steering effect.
        wrapped_modules: Optional dictionary of wrapped TopK modules from init_model_tokenizer_fixed

    Example:
        # Standard enable/disable
        with FeatureSteeringContext(model, feature_dict, amplification=5.0, wrapped_modules=wrapped_modules):
            outputs = model.generate(input_ids, max_length=100)

        # Isolate mode: enable ONLY feature 344, ablate all others
        isolate_dict = {
            "base_model.model.model.layers.11.self_attn.q_proj.topk": [
                (344, "isolate")
            ]
        }
        with FeatureSteeringContext(model, isolate_dict, amplification=10.0, wrapped_modules=wrapped_modules):
            outputs = model.generate(input_ids, max_length=100)
    """
    steering_info = steer_features(model, feature_dict, verbose=verbose,
                                   amplification=amplification, wrapped_modules=wrapped_modules)
    try:
        yield steering_info
    finally:
        # Log steering statistics before cleanup
        if verbose:
            log_steering_statistics(steering_info["steerers"], verbose=True)

        restore_forward_methods(steering_info["modules"], steering_info["steerers"])
        if verbose:
            logging.info("Steering context exited, forward methods restored")


def list_available_adapters(model: nn.Module, verbose: bool = True, wrapped_modules: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """
    List all available TopKLoRALinearSTE adapters in a model.

    Useful for discovering adapter names for steering.

    Args:
        model: PyTorch model with TopKLoRALinearSTE adapters loaded
        verbose: If True, print formatted output
        wrapped_modules: Optional dictionary of wrapped TopK modules from init_model_tokenizer_fixed

    Returns:
        Dictionary mapping adapter names to their properties (r, k, etc.)
    """
    adapters_info = {}

    # Use wrapped_modules if provided, otherwise scan model
    if wrapped_modules is not None:
        for name, module in wrapped_modules.items():
            if isinstance(module, TopKLoRALinearSTE):
                adapters_info[name] = {
                    "r": module.r,
                    "k": module._current_k(),
                    "temperature": module._tau(),
                    "num_features": module.r
                }
    else:
        for name, module in model.named_modules():
            if isinstance(module, TopKLoRALinearSTE):
                adapters_info[name] = {
                    "r": module.r,
                    "k": module._current_k(),
                    "temperature": module._tau(),
                    "num_features": module.r
                }

    if verbose:
        logging.info(
            f"\n{'='*80}\nAvailable TopKLoRALinearSTE Adapters ({len(adapters_info)} found):\n{'='*80}")
        for name, info in adapters_info.items():
            logging.info(f"\n{name}")
            logging.info(f"  - Features (r): {info['r']}")
            logging.info(f"  - Active (k): {info['k']}")
            logging.info(f"  - Temperature: {info['temperature']:.4f}")

    return adapters_info


def create_steering_dict_from_script_output(
    script_results: List[Dict[str, Any]]
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Helper function to convert assess_autointerp_dpo.py output to steering dict.

    Args:
        script_results: List of dictionaries with keys:
            - "adapter_name": Full adapter name
            - "feature_num": Feature/latent index
            - "effect": "enable" or "disable"

    Returns:
        Dictionary suitable for steer_features()

    Example:
        results = [
            {"adapter_name": "base_model.model.model.layers.11.self_attn.q_proj.topk",
             "feature_num": 217, "effect": "enable"},
            {"adapter_name": "base_model.model.model.layers.11.self_attn.q_proj.topk",
             "feature_num": 45, "effect": "disable"}
        ]
        steering_dict = create_steering_dict_from_script_output(results)
        steer_features(model, steering_dict)
    """
    from collections import defaultdict

    steering_dict = defaultdict(list)

    for result in script_results:
        adapter_name = result["adapter_name"]
        feature_num = result["feature_num"]
        effect = result.get("effect", "enable")  # default to enable

        steering_dict[adapter_name].append((feature_num, effect))

    return dict(steering_dict)

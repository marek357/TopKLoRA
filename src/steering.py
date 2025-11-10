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
    
    def hook_fn(self, module: TopKLoRALinearSTE, input: Tuple[torch.Tensor],
                output: torch.Tensor) -> torch.Tensor:
        """
        Forward hook that modifies the output to steer specific features.

        The steering works by manipulating the gate values for specific latents
        in the TopKLoRALinearSTE module before they're applied to the activations.
        
        Three modes are supported:
        - "enable": Force the specified latent(s) to be active (gate=1), leave others as-is
        - "disable": Force the specified latent(s) to be inactive (gate=0), leave others as-is
        - "isolate": Enable ONLY the specified latent(s), set ALL other latents to gate=0
                    This mode is useful for causal analysis of individual latent effects.

        Args:
            module: The TopKLoRALinearSTE module
            input: Input tuple to the module
            output: Output tensor from the module

        Returns:
            Modified output tensor with steered features
        """
        # Access the input and recompute with modified gates
        x = input[0]

        with torch.no_grad():
            # Recompute forward pass up to z (latent activations)
            A_weight = module.A_module.weight
            B_weight = module.B_module.weight

            # Apply dropout if present
            x_lora = module.dropout(x)

            # Compute latent activations
            z_pre = torch.nn.functional.linear(
                x_lora, A_weight)  # type: ignore
            if module.relu_latents:
                z = torch.nn.functional.relu(z_pre)
            else:
                z = z_pre

            # Get current k and tau
            k_now = module._current_k()
            tau = module._tau()

            # Compute gates (hard and soft)
            from src.models import _soft_topk_mass, _hard_topk_mask
            g_soft = _soft_topk_mass(z, k_now, tau)
            g_hard = _hard_topk_mask(z, k_now)

            # Handle isolate mode: ablate ALL latents first, then enable selected ones
            if self.has_isolate:
                # Start with all gates at zero
                g_soft = torch.zeros_like(g_soft)
                g_hard = torch.zeros_like(g_hard)
                
                # Logging moved outside forward hook to avoid breaking torch.compile
                # logging.debug(f"ISOLATE mode: All gates set to 0, will enable only {self.feature_indices}")

            # Apply steering by directly modifying the gate values
            for idx, effect in zip(self.feature_indices, self.effects):
                if idx >= z.shape[-1]:
                    # Skip logging to avoid compile issues
                    # logging.warning(f"Feature index {idx} out of bounds (max: {z.shape[-1]-1}), skipping")
                    continue

                # Debug: check activation magnitude (kept for monitoring, no logging in hook)
                z_magnitude = z[..., idx].abs().mean()
                original_gate_soft = g_soft[..., idx].mean()
                original_gate_hard = g_hard[..., idx].mean()
                
                if effect == "enable" or effect == "isolate":
                    # Force this feature to be active
                    g_soft[..., idx] = 1.0
                    g_hard[..., idx] = 1.0
                    
                    # AMPLIFY the activation if amplification > 1.0
                    if self.amplification > 1.0:
                        z[..., idx] = z[..., idx] * self.amplification
                    
                    # Removed debug logging to avoid breaking torch.compile
                    # mode_str = "ISOLATED" if effect == "isolate" else "ENABLED"
                    # logging.debug(f"Feature {idx}: z_mag={z_magnitude:.6f}, ...")
                    
                elif effect == "disable":
                    # Force this feature to be inactive
                    g_soft[..., idx] = 0.0
                    g_hard[..., idx] = 0.0
                    # Removed debug logging to avoid breaking torch.compile
                    # logging.debug(f"Feature {idx}: z_mag={z_magnitude:.6f}, ...")

            # Combine gates with straight-through estimator logic
            # (but we're in eval mode, so just use the hard gates)
            if not module.training and module.hard_eval:
                # In eval mode, use hard gates
                g = g_hard
            else:
                # In training mode, use STE combination
                g = g_hard + g_soft - g_soft.detach()

            # Compute LoRA contribution with steered gates
            lora_out = torch.nn.functional.linear(
                z * g, B_weight) * module.scale  # type: ignore

            # Compute base output
            base_out = module.base_layer(x)

            # Return combined output
            return base_out + lora_out


def steer_features(
    model: nn.Module,
    feature_dict: Dict[str, List[Tuple[int, str]]],
    verbose: bool = True,
    amplification: float = 1.0
) -> Dict[str, Any]:
    """
    Apply feature steering to a model with TopKLoRALinearSTE adapters.

    This function registers forward hooks on specified adapter layers to enable,
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

    Returns:
        Dictionary containing:
            - "hooks": List of registered hook handles (for cleanup)
            - "steerers": Dictionary mapping adapter names to FeatureSteerer objects
            - "applied_count": Number of successfully applied steering rules

    Usage:
        hooks_info = steer_features(model, feature_dict)
        # ... run inference ...
        remove_steering_hooks(hooks_info["hooks"])
    """
    hooks = []
    steerers = {}
    applied_count = 0

    # Find all TopKLoRALinearSTE modules in the model
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

        # Create steerer and register hook
        steerer = FeatureSteerer(feature_indices, effects, amplification=amplification)
        hook_handle = target_module.register_forward_hook(steerer.hook_fn)

        hooks.append(hook_handle)
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
        "hooks": hooks,
        "steerers": steerers,
        "applied_count": applied_count
    }


def remove_steering_hooks(hooks: List[Any]) -> None:
    """
    Remove previously registered steering hooks.

    Args:
        hooks: List of hook handles returned by steer_features()
    """
    for hook in hooks:
        hook.remove()
    logging.info(f"Removed {len(hooks)} steering hooks")


@contextmanager
def FeatureSteeringContext(
    model: nn.Module,
    feature_dict: Dict[str, List[Tuple[int, str]]],
    verbose: bool = True,
    amplification: float = 1.0
):
    """
    Context manager for feature steering (recommended usage pattern).

    Automatically handles hook registration and cleanup.

    Args:
        model: PyTorch model with TopKLoRALinearSTE adapters loaded
        feature_dict: Dictionary mapping adapter names to list of (feature_num, effect) tuples
                     where effect is "enable", "disable", or "isolate"
        verbose: If True, log information about applied steering
        amplification: Multiplier for enabled/isolated features (default 1.0). 
                      Try 5.0-10.0 for stronger steering effect.

    Example:
        # Standard enable/disable
        with FeatureSteeringContext(model, feature_dict, amplification=5.0):
            outputs = model.generate(input_ids, max_length=100)
        
        # Isolate mode: enable ONLY feature 344, ablate all others
        isolate_dict = {
            "base_model.model.model.layers.11.self_attn.q_proj.topk": [
                (344, "isolate")
            ]
        }
        with FeatureSteeringContext(model, isolate_dict, amplification=10.0):
            outputs = model.generate(input_ids, max_length=100)
    """
    hooks_info = steer_features(model, feature_dict, verbose=verbose, amplification=amplification)
    try:
        yield hooks_info
    finally:
        remove_steering_hooks(hooks_info["hooks"])
        if verbose:
            logging.info("Steering context exited, hooks cleaned up")


def list_available_adapters(model: nn.Module, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    List all available TopKLoRALinearSTE adapters in a model.

    Useful for discovering adapter names for steering.

    Args:
        model: PyTorch model with TopKLoRALinearSTE adapters loaded
        verbose: If True, print formatted output

    Returns:
        Dictionary mapping adapter names to their properties (r, k, etc.)
    """
    adapters_info = {}

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

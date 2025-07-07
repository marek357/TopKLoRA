#!/usr/bin/env python3
"""
Complete TopK LoRA DPO Training Script
All components integrated into a single file for easy use.

Usage:
    python train_topk_complete.py --model_name gemma-2-2b
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import argparse
import gc
import re
from typing import Optional, Dict, Any
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from trl import DPOTrainer, DPOConfig
from torch.optim import AdamW
import wandb

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ============================================================================
# PART 0: CALLBACKS
# ============================================================================
class DeadNeuronDetectionCallback(TrainerCallback):
    """
    Callback to detect dead neurons in TopK LoRA layers during training.
    Runs every `check_interval` steps and logs statistics to wandb.
    """
    
    def __init__(self, 
                 check_interval: int = 1000,
                 num_samples: int = 500,
                 activation_threshold: float = 0.01,
                 use_soft_detection: bool = True):
        """
        Args:
            check_interval: Check dead neurons every N steps
            num_samples: Number of samples to use for detection
            activation_threshold: Threshold below which a neuron is considered "dead"
            use_soft_detection: If True, uses soft activation values; if False, uses hard top-k
        """
        self.check_interval = check_interval
        self.num_samples = num_samples
        self.activation_threshold = activation_threshold
        self.use_soft_detection = use_soft_detection
        self.last_check_step = 0
        
    def on_step_end(self, args, state, control, model=None, train_dataloader=None, **kwargs):
        """Check for dead neurons at specified intervals."""
        if model is None or train_dataloader is None:
            return
            
        # Check if it's time to evaluate
        if state.global_step - self.last_check_step < self.check_interval:
            return
            
        self.last_check_step = state.global_step
        
        # Run dead neuron detection
        dead_neuron_stats = self.detect_dead_neurons(
            model, 
            train_dataloader, 
            state.global_step
        )
        
        # Log to wandb if enabled
        # if args.report_to and "wandb" in args.report_to:
        #     import wandb
        wandb.log(dead_neuron_stats, step=state.global_step)
    
    def detect_dead_neurons(self, model, dataloader, current_step):
        """Detect dead neurons by tracking activations over a subset of data."""
        model.eval()
        
        # Dictionary to store activation statistics for each TopK layer
        activation_stats = defaultdict(lambda: {
            'activations': None,  # Will be tensor of shape (r,)
            'counts': None,       # Number of times each neuron activated
            'layer_name': '',
            'r': 0,
            'k': 0
        })
        
        # Register hooks on all FixedTopKLoRALinear modules
        hooks = []
        hook_handles = []
        
        for name, module in model.named_modules():
            if isinstance(module, FixedTopKLoRALinear):
                layer_info = activation_stats[name]
                layer_info['layer_name'] = name
                layer_info['r'] = module.r
                layer_info['k'] = module.k
                layer_info['activations'] = torch.zeros(module.r, device='cpu')
                layer_info['counts'] = torch.zeros(module.r, device='cpu', dtype=torch.long)
                
                def make_hook(layer_name, layer_stats):
                    def hook_fn(module, input, output):
                        # Get the low-rank activations before top-k
                        x = input[0]  # Input to the module
                        with torch.no_grad():
                            # Compute z = x @ A^T (low-rank projection)
                            A = module.A.to(dtype=x.dtype, device=x.device)
                            z = F.linear(x, A)  # Shape: (batch, seq_len, r)
                            
                            if self.use_soft_detection:
                                # Use the soft mask values to determine activation
                                current_temp = module.get_temperature()
                                if module.k < module.r:
                                    abs_z = z.abs()
                                    topk_vals = abs_z.topk(module.k, dim=-1)[0]
                                    threshold = topk_vals[..., -1:].clone()
                                    soft_mask = torch.sigmoid((abs_z - threshold) / current_temp)
                                    # Average activation strength per neuron
                                    activation_strength = soft_mask.mean(dim=(0, 1))
                                else:
                                    activation_strength = torch.ones(module.r, device=z.device)
                            else:
                                # Use hard top-k detection
                                abs_z = z.abs()
                                # For each position, find which neurons are in top-k
                                topk_indices = abs_z.topk(module.k, dim=-1).indices
                                # Create binary mask
                                mask = torch.zeros_like(z)
                                mask.scatter_(-1, topk_indices, 1.0)
                                # Count activations per neuron
                                activation_strength = mask.sum(dim=(0, 1)) / mask.shape[0] / mask.shape[1]
                            
                            # Update statistics
                            layer_stats['activations'] += activation_strength.cpu()
                            layer_stats['counts'] += (activation_strength > self.activation_threshold).cpu().long()
                    
                    return hook_fn
                
                hook = make_hook(name, layer_info)
                handle = module.register_forward_hook(hook)
                hooks.append(hook)
                hook_handles.append(handle)
        
        # Process a subset of training data
        samples_processed = 0
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if samples_processed >= self.num_samples:
                        break
                    
                    # Move batch to device
                    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
                    
                    # Forward pass
                    _ = model(**batch)
                    
                    samples_processed += batch['input_ids'].shape[0]
        
        finally:
            # Remove hooks
            for handle in hook_handles:
                handle.remove()
        
        # Analyze results
        stats_to_log = {
            'dead_neurons/global_step': current_step,
            'dead_neurons/samples_analyzed': samples_processed
        }
        
        total_neurons = 0
        total_dead = 0
        total_low_activation = 0
        
        for layer_name, layer_stats in activation_stats.items():
            if layer_stats['activations'] is None:
                continue
                
            # Normalize by number of batches seen
            avg_activation = layer_stats['activations'] / max(batch_idx + 1, 1)
            
            # Dead neurons: never activated above threshold
            dead_mask = layer_stats['counts'] == 0
            num_dead = dead_mask.sum().item()
            
            # Low activation neurons: average activation below threshold
            low_activation_mask = avg_activation < self.activation_threshold
            num_low_activation = low_activation_mask.sum().item()
            
            # Update totals
            total_neurons += layer_stats['r']
            total_dead += num_dead
            total_low_activation += num_low_activation
            
            # Log per-layer statistics
            clean_layer_name = layer_name.replace('.', '_')
            stats_to_log.update({
                f'dead_neurons/layers/{clean_layer_name}/num_dead': num_dead,
                f'dead_neurons/layers/{clean_layer_name}/pct_dead': 100.0 * num_dead / layer_stats['r'],
                f'dead_neurons/layers/{clean_layer_name}/num_low_activation': num_low_activation,
                f'dead_neurons/layers/{clean_layer_name}/pct_low_activation': 100.0 * num_low_activation / layer_stats['r'],
                f'dead_neurons/layers/{clean_layer_name}/avg_activation_mean': avg_activation.mean().item(),
                f'dead_neurons/layers/{clean_layer_name}/avg_activation_std': avg_activation.std().item(),
            })
            
            # Log detailed info for debugging
            if num_dead > 0:
                dead_indices = torch.where(dead_mask)[0].tolist()
                logging.info(f"Layer {layer_name}: {num_dead}/{layer_stats['r']} dead neurons "
                           f"(indices: {dead_indices[:10]}{'...' if len(dead_indices) > 10 else ''})")
        
        # Log global statistics
        stats_to_log.update({
            'dead_neurons/total_neurons': total_neurons,
            'dead_neurons/total_dead': total_dead,
            'dead_neurons/total_pct_dead': 100.0 * total_dead / total_neurons if total_neurons > 0 else 0,
            'dead_neurons/total_low_activation': total_low_activation,
            'dead_neurons/total_pct_low_activation': 100.0 * total_low_activation / total_neurons if total_neurons > 0 else 0,
        })
        
        # Log summary
        logging.info(f"Step {current_step}: {total_dead}/{total_neurons} "
                    f"({100.0 * total_dead / total_neurons:.1f}%) dead neurons detected")
        
        model.train()
        return stats_to_log


class ActivationTrackingCallback(TrainerCallback):
    """
    Alternative callback that continuously tracks activation statistics
    and computes dead neurons from accumulated stats.
    """
    
    def __init__(self, 
                 check_interval: int = 1000,
                 reset_interval: int = 5000):
        """
        Args:
            check_interval: Report statistics every N steps
            reset_interval: Reset activation counters every N steps
        """
        self.check_interval = check_interval
        self.reset_interval = reset_interval
        self.last_check_step = 0
        self.last_reset_step = 0
        self.activation_trackers = {}
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize activation trackers for each TopK layer."""
        if model is None:
            return
            
        for name, module in model.named_modules():
            if isinstance(module, FixedTopKLoRALinear):
                self.activation_trackers[name] = {
                    'total_activations': torch.zeros(module.r),
                    'activation_counts': torch.zeros(module.r, dtype=torch.long),
                    'samples_seen': 0,
                    'r': module.r,
                    'k': module.k
                }
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update activation statistics from each TopK module."""
        if model is None:
            return
        
        # Collect activation stats from modules
        for name, module in model.named_modules():
            if isinstance(module, FixedTopKLoRALinear) and name in self.activation_trackers:
                if module._last_z is not None:
                    with torch.no_grad():
                        # Get activation magnitudes
                        z_abs = module._last_z.abs()
                        # Average over batch and sequence dimensions
                        avg_activations = z_abs.mean(dim=(0, 1)).cpu()
                        
                        # Update tracker
                        tracker = self.activation_trackers[name]
                        tracker['total_activations'] += avg_activations
                        tracker['activation_counts'] += (avg_activations > 0.01).long()
                        tracker['samples_seen'] += z_abs.shape[0]
        
        # Check if it's time to report
        if state.global_step - self.last_check_step >= self.check_interval:
            self.last_check_step = state.global_step
            self.report_dead_neurons(args, state)
        
        # Check if it's time to reset
        if state.global_step - self.last_reset_step >= self.reset_interval:
            self.last_reset_step = state.global_step
            self.reset_trackers()
    
    def report_dead_neurons(self, args, state):
        """Compute and log dead neuron statistics."""
        stats_to_log = {'dead_neurons/global_step': state.global_step}
        
        total_neurons = 0
        total_dead = 0
        
        for layer_name, tracker in self.activation_trackers.items():
            if tracker['samples_seen'] == 0:
                continue
                
            # Compute average activations
            avg_activations = tracker['total_activations'] / tracker['samples_seen']
            dead_mask = tracker['activation_counts'] == 0
            num_dead = dead_mask.sum().item()
            
            total_neurons += tracker['r']
            total_dead += num_dead
            
            # Log per-layer stats
            clean_name = layer_name.replace('.', '_')
            stats_to_log.update({
                f'dead_neurons/layers/{clean_name}/num_dead': num_dead,
                f'dead_neurons/layers/{clean_name}/pct_dead': 100.0 * num_dead / tracker['r'],
                f'dead_neurons/layers/{clean_name}/samples_seen': tracker['samples_seen']
            })
        
        # Log global stats
        stats_to_log.update({
            'dead_neurons/total_dead': total_dead,
            'dead_neurons/total_pct_dead': 100.0 * total_dead / total_neurons if total_neurons > 0 else 0
        })
        
        logging.info(f"Step {state.global_step}: {total_dead}/{total_neurons} "
                    f"({100.0 * total_dead / total_neurons:.1f}%) dead neurons")
        
        # Log to wandb
        if args.report_to and "wandb" in args.report_to:
            import wandb
            wandb.log(stats_to_log, step=state.global_step)
    
    def reset_trackers(self):
        """Reset activation trackers to avoid overflow."""
        for tracker in self.activation_trackers.values():
            tracker['total_activations'].zero_()
            tracker['activation_counts'].zero_()
            tracker['samples_seen'] = 0



# ============================================================================
# PART 1: TOPK MODELS
# ============================================================================

class SoftTopKMask(torch.autograd.Function):
    """
    Soft top-k selection with temperature-controlled smoothness.
    Key improvement over hard TopK: differentiable selection for better gradients.
    """
    @staticmethod
    def forward(ctx, z, k, temperature=1.0):
        abs_z = z.abs()
        topk_vals, topk_indices = abs_z.topk(k, dim=-1)
        threshold = topk_vals[..., -1:].clone()
        soft_mask = torch.sigmoid((abs_z - threshold) / temperature)
        ctx.save_for_backward(z, soft_mask)
        ctx.temperature = temperature
        ctx.k = k
        return z * soft_mask

    @staticmethod
    def backward(ctx, grad_output):
        z, soft_mask = ctx.saved_tensors
        temperature = ctx.temperature
        
        # Gradient through the masked values
        grad_z = grad_output * soft_mask
        
        # Gradient through the mask itself
        abs_z = z.abs()
        topk_vals = abs_z.topk(ctx.k, dim=-1)[0]
        threshold = topk_vals[..., -1:].clone()
        
        # Sigmoid gradient
        sigmoid_grad = soft_mask * (1 - soft_mask)
        mask_grad = grad_output * z * sigmoid_grad * torch.sign(z) / temperature
        
        # Prevent gradient explosion
        near_threshold = (abs_z - threshold).abs() < 3 * temperature
        mask_grad = mask_grad * near_threshold.float()
        
        grad_z = grad_z + mask_grad
        return grad_z, None, None


class FixedTopKLoRALinear(nn.Module):
    """
    Fixed TopK LoRA layer with soft masking.
    Improvements over original:
    - Soft masking instead of hard masking
    - Better gradient flow
    - Proper scaling
    - Comprehensive statistics tracking
    """
    def __init__(self, base: LoraLayer, *, layer_name: str,
                 r, alpha, k: int, temperature: float = 0.1, 
                 temperature_schedule: str = "anneal"):
        super().__init__()
        self.lora_module = base
        self.base_layer = base.base_layer
        
        # Get LoRA matrices
        adapter = base.active_adapter[0]
        self.A = base.lora_A[adapter].weight
        self.B = base.lora_B[adapter].weight
        
        # Handle dict configs
        r_val = r["default"] if isinstance(r, dict) else r
        alpha_val = alpha["default"] if isinstance(alpha, dict) else alpha
        
        self.r = int(r_val)
        self.k = int(k)
        self.temperature = temperature
        self.temperature_schedule = temperature_schedule
        self.initial_temperature = temperature
        self.register_buffer('training_progress', torch.tensor(0.0))
        # Scale by alpha/k since only k components are active
        self.scale = alpha_val / self.k
        
        self.layer_name = layer_name
        
        # Statistics tracking
        self.register_buffer('activation_stats', torch.zeros(self.r))
        self.register_buffer('update_count', torch.tensor(0))
        
        # For monitoring
        self._last_z = None
        self._last_z_sparse = None
        self.counter = 0
        
        logging.info(
            f"Fixed TopK LoRA: {layer_name} "
            f"(r={self.r}, k={self.k}, sparsity={(1-self.k/self.r)*100:.1f}%, "
            f"scale=α/k={self.scale:.3f}, temp={temperature})"
        )

    def update_training_progress(self, progress: float):
        """Update training progress (0 to 1)"""
        self.training_progress.fill_(progress)


    def get_temperature(self):
        """Temperature based on current training progress"""
        if self.temperature_schedule == "constant":
            return self.temperature
        elif self.temperature_schedule == "anneal":
            # Exponential decay from initial to 0.01 * initial
            progress = self.training_progress.item()
            return self.initial_temperature * (0.01 ** progress)
        elif self.temperature_schedule == "linear":
            # Linear decay to 0.1 * initial
            progress = self.training_progress.item()
            return self.initial_temperature * (1.0 - 0.9 * progress)


    def forward(self, x: torch.Tensor):
        # print('calling forward!!!!')
        self.counter += 1
        # Cast to match input
        A = self.A.to(dtype=x.dtype, device=x.device)
        B = self.B.to(dtype=x.dtype, device=x.device)
        
        # Low-rank projection
        z = F.linear(x, A)
        
        # Track statistics
        # if self.training:
        with torch.no_grad():
            # self.activation_stats += z.abs().mean(dim=0)
            self.activation_stats += z.detach().clone().cpu().abs().mean(dim=(0, 1))
            self.update_count += 1
            self._last_z = z.detach().clone()

        current_temp = self.get_temperature()
        if self.k < self.r:
            z_sparse = SoftTopKMask.apply(z, self.k, current_temp)
        else:
            z_sparse = z

        # if self.counter % 100 == 0:
        #     self.log({
        #         f"sparsity/{self.layer_name}/temp": current_temp,
        #         'progress': self.training_progress.item()
        #     })
            # logging.debug(f"{self.layer_name}: temp={current_temp:.4f}, progress={self.training_progress.item():.3f}")


        
        # # Apply soft top-k
        # if self.k < self.r:
        #     z_sparse = SoftTopKMask.apply(z, self.k, self.temperature)
        # else:
        #     z_sparse = z
        
        # Store for monitoring
        # if self.training:
        with torch.no_grad():
            self._last_z_sparse = z_sparse.detach().clone()
        
        # Apply base layer + sparse LoRA
        out = self.base_layer(x)
        lora_out = F.linear(z_sparse, B) * self.scale
        
        return out + lora_out

    def get_sparsity_stats(self):
        """Return sparsity statistics"""
        stats = {
            'k': self.k,
            'r': self.r,
            'sparsity_ratio': 1.0 - (self.k / self.r),
            'temperature': self.temperature,
            'scale': self.scale,
            'counter': self.counter,
        }
        
        if self.update_count > 0:
            stats['activation_mean'] = self.activation_stats.mean().item()
            stats['activation_std'] = self.activation_stats.std().item()
        
        return stats


class TopKProgressCallback(TrainerCallback):
    """Update training progress in TopK modules"""
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.total_steps = state.max_steps
        
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            progress = state.global_step / self.total_steps
            # Update all TopK modules
            for module in model.modules():
                if isinstance(module, FixedTopKLoRALinear):
                    module.update_training_progress(progress)

# ============================================================================
# PART 2: TRAINING COMPONENTS
# ============================================================================

class EnhancedDPOTrainer(DPOTrainer):
    """DPO Trainer with improved logging and adaptive beta"""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        
        # Log individual LoRA module-level stats
        for name, module in model.named_modules():
            if isinstance(module, FixedTopKLoRALinear):
                if module.counter % 100 == 0:
                    current_temp = module.get_temperature()
                    self.log({
                        f"{module.layer_name}/temp": current_temp,
                        'progress': module.training_progress.item()
                    })

        
        return (loss, outputs) if return_outputs else loss


class MemoryClearCallback(TrainerCallback):
    """Memory management callback"""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.gradient_accumulation_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# PART 3: HELPER FUNCTIONS
# ============================================================================

def get_optimal_layer_targets(model_name: str, num_layers: int = 4) -> list:
    """Select optimal layers for LoRA"""
    target_modules = []    
    selected_layers = [11]
    
    for layer_idx in selected_layers:
        prefix = f"layers.{layer_idx}" if layer_idx >= 0 else f"model.layers.{layer_idx}"
        target_modules.extend([
            f'{prefix}.self_attn.q_proj',
            f'{prefix}.self_attn.k_proj', 
            f'{prefix}.self_attn.v_proj',
            f'{prefix}.self_attn.o_proj',
            f'{prefix}.mlp.gate_proj',
            f'{prefix}.mlp.up_proj',
            f'{prefix}.mlp.down_proj'
        ])
    
    return target_modules


# def prepare_datasets(max_length=1024, train_size=None, eval_size=500):
#     """Load and prepare DPO datasets"""
#     cache_dir = os.path.join(os.getcwd(), 'cache')
#     os.makedirs(cache_dir, exist_ok=True)
    
#     def format_dataset(samples):
#         prompts = [f"###Question:\n{h}\n\n###Answer:\n" for h in samples["history"]]
#         chosen = [
#             A if lab == 1 else B
#             for lab, A, B in zip(samples["labels"], samples["human_ref_A"], samples["human_ref_B"])
#         ]
#         rejected = [
#             B if lab == 1 else A
#             for lab, A, B in zip(samples["labels"], samples["human_ref_A"], samples["human_ref_B"])
#         ]
#         return {"prompt": prompts, "chosen": chosen, "rejected": rejected}
    
#     # Load datasets
#     logging.info("Loading datasets...")
#     train_dataset = load_dataset("stanfordnlp/shp", cache_dir=cache_dir, split="train")
#     eval_dataset = load_dataset("stanfordnlp/shp", cache_dir=cache_dir, split="test")
    
#     # Format
#     train_dataset = train_dataset.map(
#         format_dataset, 
#         batched=True, 
#         remove_columns=train_dataset.column_names,
#         desc="Formatting train dataset"
#     )
#     eval_dataset = eval_dataset.map(
#         format_dataset, 
#         batched=True, 
#         remove_columns=eval_dataset.column_names,
#         desc="Formatting eval dataset"
#     )
    
#     # Filter by length
#     train_dataset = train_dataset.filter(
#         lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length and
#                   len(x["prompt"]) + len(x["rejected"]) <= max_length and
#                   len(x["chosen"]) > 10 and len(x["rejected"]) > 10
#     )
    
#     eval_dataset = eval_dataset.filter(
#         lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length and
#                   len(x["prompt"]) + len(x["rejected"]) <= max_length
#     )
    
#     # Limit sizes if specified
#     if train_size:
#         train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
#     if eval_size:
#         eval_dataset = eval_dataset.select(range(min(eval_size, len(eval_dataset))))
    
#     logging.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
#     return train_dataset, eval_dataset

def prepare_datasets(max_length=1024, train_size=None, eval_size=500, tokenizer=None):
    """Load and prepare DPO datasets using tokenizer's chat template"""
    cache_dir = os.path.join(os.getcwd(), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    def format_dataset(samples):
        formatted_prompts = []
        formatted_chosen = []
        formatted_rejected = []
        
        for h, lab, A, B in zip(samples["history"], samples["labels"], 
                                samples["human_ref_A"], samples["human_ref_B"]):
            # Create message format
            messages = [{"role": "user", "content": h}]
            
            # Apply tokenizer's chat template for the prompt
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Format chosen/rejected with the response
            chosen_text = A if lab == 1 else B
            rejected_text = B if lab == 1 else A
            
            # For responses, we just need the text (DPO trainer will handle the rest)
            formatted_prompts.append(prompt)
            formatted_chosen.append(chosen_text)
            formatted_rejected.append(rejected_text)
        
        return {
            "prompt": formatted_prompts, 
            "chosen": formatted_chosen, 
            "rejected": formatted_rejected
        }
    
    # Load datasets
    logging.info("Loading datasets...")
    train_dataset = load_dataset("stanfordnlp/shp", cache_dir=cache_dir, split="train")
    eval_dataset = load_dataset("stanfordnlp/shp", cache_dir=cache_dir, split="test")
    
    # Format
    train_dataset = train_dataset.map(
        format_dataset, 
        batched=True, 
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset"
    )
    eval_dataset = eval_dataset.map(
        format_dataset, 
        batched=True, 
        remove_columns=eval_dataset.column_names,
        desc="Formatting eval dataset"
    )
    
    # Filter by length
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length and
                  len(x["prompt"]) + len(x["rejected"]) <= max_length and
                  len(x["chosen"]) > 10 and len(x["rejected"]) > 10
    )
    
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length and
                  len(x["prompt"]) + len(x["rejected"]) <= max_length
    )
    
    # Limit sizes if specified
    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if eval_size:
        eval_dataset = eval_dataset.select(range(min(eval_size, len(eval_dataset))))
    
    logging.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


# ============================================================================
# PART 4: MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="gemma-2-2b", 
                        help="Model name (e.g., gemma-2-2b, gemma-7b)")
    parser.add_argument("--sft_model_path", type=str, default=None,
                        help="Path to SFT model (defaults to experiments/merged/{model_name}_sft)")
    parser.add_argument("--output_dir", type=str, default="./experiments/",
                        help="Output directory")
    
    # TopK LoRA arguments
    parser.add_argument("--r", type=int, default=256, help="LoRA rank")
    parser.add_argument("--k", type=int, default=32, help="TopK value (fixed)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Soft TopK temperature")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers to target")
    
    # DPO arguments
    parser.add_argument("--beta", type=float, default=0.15, help="DPO beta")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--eval_steps", type=int, default=100, help="Eval frequency")
    parser.add_argument("--save_steps", type=int, default=200, help="Save frequency")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=9000, help="Max sequence length")
    
    # Training arguments
    parser.add_argument("--wandb_project", type=str, default="topk-lora-dpo", help="WandB project")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")

    parser.add_argument("--check_dead_neurons", action="store_true", 
                        help="Enable dead neuron detection during training")
    parser.add_argument("--dead_neuron_interval", type=int, default=1000,
                        help="Check dead neurons every N steps")
    parser.add_argument("--dead_neuron_samples", type=int, default=500,
                        help="Number of samples to use for dead neuron detection")
    parser.add_argument("--dead_neuron_method", type=str, default="periodic",
                        choices=["continuous", "periodic"],
                        help="Method for dead neuron detection")

    
    args = parser.parse_args()
    
    # Setup paths
    if args.sft_model_path is None:
        args.sft_model_path = f"./experiments/merged/google/{args.model_name}_sft"
    
    output_dir = os.path.join(args.output_dir, f"{args.model_name}_topk_dpo_r{args.r}_k{args.k}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = args.run_name or f"{args.model_name}_r{args.r}_k{args.k}_beta{args.beta}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
    
    # Log configuration
    logging.info("="*60)
    logging.info("TopK LoRA DPO Training Configuration:")
    logging.info(f"  Model: {args.model_name}")
    logging.info(f"  LoRA: r={args.r}, k={args.k} (sparsity={(1-args.k/args.r)*100:.1f}%)")
    logging.info(f"  DPO: beta={args.beta}, lr={args.lr}")
    logging.info(f"  Training: epochs={args.epochs}, batch_size={args.batch_size}")
    logging.info(f"  Output: {output_dir}")
    logging.info("="*60)
    
    # Load datasets
    train_size = 1000 if args.debug else None
    eval_size = 100 if args.debug else 500
    
    # Load tokenizer
    logging.info(f"Loading tokenizer from {args.sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token



    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load policy model
    logging.info("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        attn_implementation='eager'
    )
    policy_model = prepare_model_for_kbit_training(policy_model)
    extra = False

    if not getattr(tokenizer, "chat_template", None):
        logging.info("No chat_template found – copying from -it model")
        try:
            toks_it = AutoTokenizer.from_pretrained(
                f'google/{args.model_name}-it',
                use_fast=False
            )
            if getattr(toks_it, "chat_template", None):
                tokenizer.chat_template = toks_it.chat_template
                logging.info("chat_template copied successfully")
            # Merge additional special tokens if needed
            extra = toks_it.special_tokens_map.get(
                "additional_special_tokens", []
            )
            if extra:
                new_tokens = [
                    t for t in extra if t not in tokenizer.get_vocab()
                ]
                if new_tokens:
                    tokenizer.add_special_tokens(
                        {"additional_special_tokens": new_tokens}
                    )
                    policy_model.resize_token_embeddings(len(tokenizer))
                    logging.info(
                        "Added %d extra special tokens",
                        len(new_tokens)
                    )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to copy -it tokenizer: %s", exc)


    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens", 
            [tokenizer.eos_token]
        )[1] if len(tokenizer.special_tokens_map.get(
            "additional_special_tokens", []
        )) > 1 else tokenizer.eos_token
    )

    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)


    if hasattr(policy_model.generation_config, 'eos_token_id'):
        if isinstance(policy_model.generation_config.eos_token_id, list):
            if eot_token_id not in policy_model.generation_config.eos_token_id:
                policy_model.generation_config.eos_token_id.append(eot_token_id)
        else:
            prev_eos = policy_model.generation_config.eos_token_id
            policy_model.generation_config.eos_token_id = [prev_eos, eot_token_id]
    else:
        policy_model.generation_config.eos_token_id = [tokenizer.eos_token_id, eot_token_id]

    
    # Load reference model
    logging.info("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager'
    )
    ref_model.generation_config.eos_token_id = policy_model.generation_config.eos_token_id
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if extra:
        if new_tokens:
            ref_model.resize_token_embeddings(len(tokenizer))
            logging.info(
                "Added %d extra special tokens to ref model",
                len(new_tokens)
            )

    train_dataset, eval_dataset = prepare_datasets(
        max_length=args.max_length,
        train_size=train_size,
        eval_size=eval_size,
        tokenizer=tokenizer
    )


    
    # Configure LoRA
    target_modules = get_optimal_layer_targets(args.model_name, args.num_layers)
    logging.info(f"Target modules: {len(target_modules)} modules across {args.num_layers} layers")
    
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(policy_model, lora_config)
    model.config.use_cache = False
    
    # Inject TopK wrappers
    logging.info("Injecting Fixed TopK wrappers...")
    replaced = 0
    for name, module in model.named_modules():
        if getattr(module, "lora_A", None) is None:
            continue
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        attr = name.split(".")[-1]
        setattr(parent, attr, FixedTopKLoRALinear(
            module,
            layer_name=name,
            r=module.r,
            alpha=module.lora_alpha,
            k=args.k,
            temperature=args.temperature,
        ))
        replaced += 1
    
    logging.info(f"Injected Fixed TopK wrappers in {replaced} layers")
    model.print_trainable_parameters()
    
    # DPO configuration
    dpo_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        bf16=True,
        report_to="none" if args.no_wandb else "wandb",
        run_name=run_name if not args.no_wandb else None,
        remove_unused_columns=False,
        max_grad_norm=0.5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/accuracies",
        greater_is_better=True,
        save_total_limit=3,
    )
    
    # Create trainer
    trainer = EnhancedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
        processing_class=tokenizer,
        callbacks=[
            MemoryClearCallback(),
            TopKProgressCallback(),
            DeadNeuronDetectionCallback(
                check_interval=args.dead_neuron_interval,
                num_samples=args.dead_neuron_samples,
                activation_threshold=0.001, # TODO: consider lowering to 0
                use_soft_detection=False
            )
            # EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )
    
    # Train
    logging.info("Starting training...")
    trainer.train()
    
    # Unwrap TopK wrappers before saving
    logging.info("Unwrapping TopK wrappers...")
    unwrapped = 0
    for name, module in model.named_modules():
        if isinstance(module, FixedTopKLoRALinear):
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            attr = name.split(".")[-1]
            setattr(parent, attr, module.lora_module)
            unwrapped += 1

    # for name, module in model.named_modules():
    #     if isinstance(module, FixedTopKLoRALinear):
    #         module.temperature_schedule = "anneal"  # or "linear"

    
    logging.info(f"Reverted {unwrapped} Fixed TopK wrappers")
    
    # Save final model
    final_path = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logging.info(f"Model saved to {final_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final model saved to: {final_path}")
    if hasattr(trainer.state, 'best_metric'):
        print(f"Best eval accuracy: {trainer.state.best_metric:.4f}")
    print("\nConfiguration summary:")
    print(f"  - LoRA: r={args.r}, k={args.k} (sparsity={(1-args.k/args.r)*100:.1f}%)")
    print(f"  - Soft masking with temperature={args.temperature}")
    print(f"  - DPO beta={args.beta}, lr={args.lr}")
    print(f"  - Targeted {args.num_layers} layers")
    print("="*60)
    
    if not args.no_wandb:
        wandb.finish()


# ============================================================================
# PART 5: INFERENCE HELPER
# ============================================================================

def inference_example(model_path: str, adapter_path: str):
    """Example of how to use the trained model for inference"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation='eager'
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Example prompt
    prompt = "###Question:\nWhat is the meaning of life?\n\n###Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
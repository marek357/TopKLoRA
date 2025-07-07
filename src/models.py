from transformers import TrainerCallback
from peft.tuners.lora import LoraLayer
from trl import DPOTrainer
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch
import gc



class CustomDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Extract rewards
        rewards_chosen = outputs["rewards_chosen"]
        rewards_rejected = outputs["rewards_rejected"]

        # Normalize rewards
        all_rewards = torch.cat([rewards_chosen, rewards_rejected], dim=0)
        mean = all_rewards.mean()
        std = all_rewards.std() + 1e-8
        rewards_chosen = (rewards_chosen - mean) / std
        rewards_rejected = (rewards_rejected - mean) / std

        # Recompute loss using logsigmoid
        beta = self.args.beta
        logits_diff = beta * (rewards_chosen - rewards_rejected)
        loss = -F.logsigmoid(logits_diff).mean()

        if return_outputs:
            return loss, outputs
        return loss




class TopKMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, k):
        # z: (…, r)
        # compute a mask of shape z indicating the top-k magnitudes *per row*
        abs_z = z.abs()
        # get indices of the top-k values along last dim
        topk_indices = abs_z.topk(k, dim=-1).indices
        # build a float mask
        mask = torch.zeros_like(z).scatter_(-1, topk_indices, 1.0)
        ctx.save_for_backward(mask)
        return z * mask

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pretend mask was all-ones, so every channel gets its gradient
        grad_z = grad_output
        # no gradient w.r.t. k
        return grad_z, None


class TopKLoRALinear(nn.Module):
    def __init__(self, base: LoraLayer, *, layer_name: str,
                 r, alpha, k: int, autointerp_evaluation_mode: bool = False):
        super().__init__()
        self.lora_module = base
        self.base_layer  = base.base_layer

        # always pick the *active* adapter
        adapter = base.active_adapter[0]
        self.A = base.lora_A[adapter].weight
        self.B = base.lora_B[adapter].weight

        # unpack dict-or-int
        r_val     = r["default"] if isinstance(r, dict) else r
        alpha_val = alpha["default"] if isinstance(alpha, dict) else alpha

        self.r = int(r_val)
        self.k = int(k)
        # scale by α/k since only k components survive
        self.scale = alpha_val / self.k

        self.layer_name = layer_name
        print(f"Using TopK LoRA Adapter (r={self.r}, k={self.k}, scale=α/k={self.scale:.3f})")
        if hasattr(base.base_layer, 'weight'):
            expected_dtype = base.base_layer.weight.dtype
            expected_device = base.base_layer.weight.device
            self.A = self.A.to(dtype=expected_dtype, device=expected_device)
            self.B = self.B.to(dtype=expected_dtype, device=expected_device)

        self.counter = 0
        self.autointerp_evaluation_mode = autointerp_evaluation_mode

    def forward(self, x: torch.Tensor):
        # if not self.autointerp_evaluation_mode:
        #     self.counter += 1

        # compute the low-rank update
        z = F.linear(x, self.A)                      # shape (..., r)
        if self.k < self.r:
            # hard top-k with STE
            z = TopKMask.apply(z, self.k)

        # apply base layer + sparse LoRA update
        out = self.base_layer(x)
        out = out + F.linear(z, self.B) * self.scale
        return out

    @property
    def weight(self):
        return self.base_layer.weight




# ────────────────────────────────────────────────────────────────────────────
# Top-k LoRA module  ── self-contained for convenience
# ────────────────────────────────────────────────────────────────────────────
def _first_weight(md: nn.ModuleDict):
    return next(iter(md.values())).weight


class TopKLoRALinearLegacy(nn.Module):
    def __init__(
            self,
            base: LoraLayer, *,
            layer_name: str, r,
            alpha, k: int
    ):
        super().__init__()
        # store for unwrapping
        self.lora_module = base
        # frozen quant/FP layer
        self.base_layer = base.base_layer
        # LoRA params
        self.A = _first_weight(base.lora_A)
        self.B = _first_weight(base.lora_B)
        # support dict or int
        r_val = r["default"] if isinstance(r, dict) else r
        alpha_val = alpha["default"] if isinstance(alpha, dict) else alpha
        self.r = int(r_val)
        self.k = int(k)
        self.scale = alpha_val / r_val
        self.layer_name = layer_name
        print(f"Using a TopK LoRA Adapter with r: {self.r}, k: {self.k}")

    @property
    def active_adapter(self):
        return getattr(self.lora_module, "active_adapter", None)

    @property
    def lora_A(self):
        return self.lora_module.lora_A


    def forward(self, x: torch.Tensor):
        # print(f"[TopKLoRALinear] Called on input with shape {x.shape}")
        # match dtype for mixed precision
        A = self.A.to(dtype=x.dtype, device=x.device)
        B = self.B.to(dtype=x.dtype, device=x.device)
        z = F.linear(x, A)
        if self.k < self.r:
            thresh = z.abs().topk(self.k, dim=-1)[0][..., -1:]
            z = torch.where(z.abs() >= thresh, z, z.new_zeros(()))
        out = self.base_layer(x)
        out += F.linear(z, B) * self.scale
        return out


class MemoryClearCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return control

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


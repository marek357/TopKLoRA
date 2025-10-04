from typing import Optional
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
        # get indices of the top-k values along last dim
        topk_indices = z.topk(k, dim=-1).indices
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


class TopKMaskModule(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, z: torch.Tensor):
        return TopKMask.apply(z, self.k)


class TopKLoRALinear(nn.Module):
    def __init__(
            self, base: LoraLayer, *, layer_name: str,
            r, alpha, k: int, autointerp_evaluation_mode: bool = False
    ):
        super().__init__()
        self.lora_module = base
        self.base_layer = base.base_layer

        # always pick the *active* adapter
        adapter = base.active_adapter[0]
        self.A = base.lora_A[adapter].weight
        self.B = base.lora_B[adapter].weight

        # unpack dict-or-int
        r_val = r["default"] if isinstance(r, dict) else r
        alpha_val = alpha["default"] if isinstance(alpha, dict) else alpha

        self.r = int(r_val)
        self.k = int(k)
        self.topk = TopKMaskModule(self.k)
        # scale by α/k since only k components survive
        self.scale = alpha_val / self.k

        self.layer_name = layer_name
        print(
            f"Using TopK LoRA Adapter (r={self.r}, k={self.k}, scale=α/k={self.scale:.3f})")
        self.counter = 0
        self.autointerp_evaluation_mode = autointerp_evaluation_mode

        # For evaluation: store intermediate activations
        self._last_z = None
        self._last_z_sparse = None

    def forward(self, x: torch.Tensor):
        # if not self.autointerp_evaluation_mode:
        #     self.counter += 1
        A = self.A.to(dtype=x.dtype, device=x.device)
        B = self.B.to(dtype=x.dtype, device=x.device)

        # compute the low-rank update
        # z = F.linear(x, self.A)                      # shape (..., r)
        z = F.linear(x, A)

        # Store for evaluation
        if self.autointerp_evaluation_mode:
            self._last_z = z.detach().clone()

        if self.k < self.r:
            # hard top-k with STE
            # z = TopKMask.apply(z, self.k)
            z = self.topk(z)

        # Store sparse activations for evaluation
        if self.autointerp_evaluation_mode:
            self._last_z_sparse = z.detach().clone()

        # apply base layer + sparse LoRA update
        out = self.base_layer(x)
        out = out + F.linear(z, B) * self.scale
        return out

    @property
    def weight(self):
        return self.base_layer.weight

    def get_last_activations(self):
        """Get the last recorded activations (for evaluation)"""
        return {
            'z_dense': self._last_z,
            'z_sparse': self._last_z_sparse
        }


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
        topk_vals, topk_indices = z.topk(k, dim=-1)
        threshold = topk_vals[..., -1:].clone()
        soft_mask = torch.sigmoid((z - threshold) / temperature)
        ctx.save_for_backward(z, soft_mask)
        ctx.temperature = temperature
        ctx.k = k
        return z * soft_mask
    # @staticmethod
    # def forward(ctx, z, k, temperature=1.0):
    #     abs_z = z.abs()
    #     topk_vals, topk_indices = abs_z.topk(k, dim=-1)
    #     threshold = topk_vals[..., -1:].clone()
    #     soft_mask = torch.sigmoid((abs_z - threshold) / temperature)
    #     ctx.save_for_backward(z, soft_mask)
    #     ctx.temperature = temperature
    #     ctx.k = k
    #     return z * soft_mask

    @staticmethod
    def backward(ctx, grad_output):
        z, soft_mask = ctx.saved_tensors
        temperature = ctx.temperature

        # Gradient through the masked values
        grad_z = grad_output * soft_mask

        # Gradient through the mask itself
        topk_vals = z.topk(ctx.k, dim=-1)[0]
        threshold = topk_vals[..., -1:].clone()

        # Sigmoid gradient
        sigmoid_grad = soft_mask * (1 - soft_mask)
        mask_grad = grad_output * z * \
            sigmoid_grad * torch.sign(z) / temperature

        # Prevent gradient explosion
        near_threshold = (z - threshold).abs() < 3 * temperature
        mask_grad = mask_grad * near_threshold.float()

        grad_z = grad_z + mask_grad
        return grad_z, None, None
    # @staticmethod
    # def backward(ctx, grad_output):
    #     z, soft_mask = ctx.saved_tensors
    #     temperature = ctx.temperature

    #     # Gradient through the masked values
    #     grad_z = grad_output * soft_mask

    #     # Gradient through the mask itself
    #     abs_z = z.abs()
    #     topk_vals = abs_z.topk(ctx.k, dim=-1)[0]
    #     threshold = topk_vals[..., -1:].clone()

    #     # Sigmoid gradient
    #     sigmoid_grad = soft_mask * (1 - soft_mask)
    #     mask_grad = grad_output * z * sigmoid_grad * torch.sign(z) / temperature

    #     # Prevent gradient explosion
    #     near_threshold = (abs_z - threshold).abs() < 3 * temperature
    #     mask_grad = mask_grad * near_threshold.float()

    #     grad_z = grad_z + mask_grad
    #     return grad_z, None, None


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


class TopKProgressCallback(TrainerCallback):
    """Update training progress in TopK modules"""

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and state.max_steps:
            p = state.global_step / state.max_steps
            for m in model.modules():
                if isinstance(m, TopKLoRALinearSTE):
                    m.set_progress(p)


class DeadLatentsLoggerCallback(TrainerCallback):
    def __init__(self, log_every=500, activation_threshold=1e-6):
        """
        Args:
            log_every: steps between W&B logs
            activation_threshold: below this value, a latent is considered inactive for that batch
        """
        self.log_every = log_every
        self.activation_threshold = activation_threshold
        self.stats = {}  # {layer_name: {"counts": tensor, "total": int}}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Initialize tracking for each TopKLoRALinearSTE layer
        if model is None:
            return
        for name, module in model.named_modules():
            if isinstance(module, TopKLoRALinearSTE):
                self.stats[name] = {
                    "counts": torch.zeros(module.r, dtype=torch.long),
                    "total": 0
                }

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        # Track activations for each TopK module
        for name, module in model.named_modules():
            if isinstance(module, TopKLoRALinearSTE) and hasattr(module, "_last_g_soft"):
                g_soft = module._last_g_soft  # [B, T, r]
                # Mean gate activation per latent
                mean_act = g_soft.mean(dim=(0, 1))
                active_mask = (mean_act > self.activation_threshold).cpu()
                self.stats[name]["counts"] += active_mask.long()
                self.stats[name]["total"] += 1

        # Periodic logging
        if state.global_step % self.log_every == 0 and args.report_to and "wandb" in args.report_to:
            log_dict = {}
            total_dead_all = 0
            total_latents_all = 0

            for layer_name, st in self.stats.items():
                total_seen = st["total"]
                if total_seen == 0:
                    continue
                dead_mask = st["counts"] == 0
                num_dead = dead_mask.sum().item()
                pct_dead = 100.0 * num_dead / len(dead_mask)

                total_dead_all += num_dead
                total_latents_all += len(dead_mask)

                log_dict[f"dead_latents/{layer_name}/count"] = num_dead
                log_dict[f"dead_latents/{layer_name}/pct"] = pct_dead

            if total_latents_all > 0:
                log_dict["dead_latents/total_count"] = total_dead_all
                log_dict["dead_latents/total_pct"] = 100.0 * \
                    total_dead_all / total_latents_all
            wandb.log(log_dict, step=state.global_step)


def _soft_topk_mass(z, k, tau):
    # compute in fp32 for stability, then rescale to sum=k
    z_fp32 = z.float() / max(tau, 1e-6)
    g = torch.softmax(z_fp32, dim=-1)
    g = g * (float(k) / (g.sum(dim=-1, keepdim=True) + 1e-8))
    return g.to(z.dtype)


def _hard_topk_mask(z, k):
    # returns 0/1 mask with exactly k ones along last dim
    idx = z.topk(k, dim=-1).indices
    hard = torch.zeros_like(z)
    return hard.scatter_(-1, idx, 1.0)


class TopKModule(nn.Module):
    """Base class for Top-K modules."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        mask = _hard_topk_mask(x, self.k)
        return x * mask


class TopKLoRALinearSTE(nn.Module):
    """
    LoRA with straight-through Top-K gating over the latent dim (r).
    - Forward: hard top-k mask (exact k active channels).
    - Backward: gradients flow through a soft surrogate.
    - Supports k and temperature schedules driven by `progress` (0..1).
    """

    def __init__(
        self,
        base: LoraLayer,
        *,
        layer_name: str,
        k: int,
        temperature: float = 1.0,
        # "constant" | "linear" | "exp" | "cubic"
        temperature_schedule: str = "linear",
        k_schedule: str = "constant",           # "constant" | "linear" | "exp" | "cubic"
        # target k at progress=1
        k_final: Optional[int] = None,
        hard_eval: bool = True,                 # use hard mask in eval
        relu_latents: bool = True,              # force z >= 0
        alpha_over_r: bool = True,              # scaling mode
        # optional target temperature at progress=1
        temperature_final: Optional[float] = None,
    ):
        super().__init__()
        self.lora_module = base
        self.base_layer = base.base_layer
        adapter = base.active_adapter if isinstance(
            base.active_adapter, str) else base.active_adapter[0]

        self.A_module = base.lora_A[adapter]
        self.B_module = base.lora_B[adapter]
        self.dropout = (base.lora_dropout[adapter]
                        if hasattr(base, "lora_dropout") and adapter in base.lora_dropout
                        else nn.Identity())

        self.r = int(base.r[adapter])
        self.alpha = float(base.lora_alpha[adapter])
        self.k_init = int(k)
        self.k_final = int(k_final) if k_final is not None else int(k)
        self.k_schedule = k_schedule
        self.t0 = float(temperature)
        self.t_final = float(
            temperature_final) if temperature_final is not None else 0.1 * self.t0
        self.temperature_schedule = temperature_schedule
        self.hard_eval = hard_eval
        self.relu_latents = relu_latents
        self.scale = (
            self.alpha / self.r) if alpha_over_r else (self.alpha / max(self.k_init, 1))
        self.layer_name = layer_name
        self.topk = TopKModule(k_final)

        # Progress variable (0..1)
        self.register_buffer("progress", torch.tensor(0.0))
        self.register_buffer("last_frac_grad_nonzero", torch.tensor(0.0))

        # Transient caches for regs/logging
        self._z_live: Optional[torch.Tensor] = None
        self._g_soft_live: Optional[torch.Tensor] = None
        self._last_z: Optional[torch.Tensor] = None
        self._last_g_soft: Optional[torch.Tensor] = None
        self._last_ghard_mean: torch.Tensor = torch.tensor(0.0)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override to delegate to the wrapped lora_module's state_dict.
        This makes the wrapper transparent to PEFT saving.
        """
        # Get the lora_module's state dict
        lora_state = self.lora_module.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars
        )

        # Also include our own buffers (progress, etc)
        for name, buf in self.named_buffers(recurse=False):
            if destination is None:
                destination = lora_state
            destination[prefix + name] = buf if keep_vars else buf.detach()

        return lora_state

    def load_state_dict(self, state_dict, strict=True):
        """
        Override to properly load both lora_module weights and our buffers.
        """
        # Separate our buffers from lora weights
        our_buffers = {}
        lora_state = {}

        for k, v in state_dict.items():
            if k in ['progress', 'last_frac_grad_nonzero']:
                our_buffers[k] = v
            else:
                lora_state[k] = v

        # Load lora weights
        if lora_state:
            self.lora_module.load_state_dict(lora_state, strict=strict)

            # Update our references to A and B modules
            adapter = self.lora_module.active_adapter if isinstance(
                self.lora_module.active_adapter, str) else self.lora_module.active_adapter[0]
            self.A_module = self.lora_module.lora_A[adapter]
            self.B_module = self.lora_module.lora_B[adapter]

        # Load our buffers
        for name, value in our_buffers.items():
            if hasattr(self, name):
                getattr(self, name).copy_(value)

        return

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Hook called by PyTorch during state_dict collection.
        Delegate to lora_module to maintain namespace compatibility.
        """
        # Save lora_module's parameters
        for name, param in self.lora_module.named_parameters():
            if param is not None:
                destination[prefix +
                            name] = param if keep_vars else param.detach()

        # Save our buffers
        for name, buf in self.named_buffers(recurse=False):
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Hook called by PyTorch during load_state_dict.
        """
        # Load our buffers
        for name in ['progress', 'last_frac_grad_nonzero']:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, state_dict[key])

        # Delegate lora weights to the lora_module
        lora_prefix = prefix  # Keep same prefix for transparency
        self.lora_module._load_from_state_dict(
            state_dict, lora_prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    # -------- Progress control --------
    def set_progress(self, p: float):
        """Set training progress in [0, 1]."""
        self.progress.fill_(float(min(max(p, 0.0), 1.0)))

    # -------- Scheduling --------
    def _tau(self):
        p = float(self.progress)
        # If constant or already at target, keep t0
        if self.temperature_schedule == "constant" or abs(self.t0 - self.t_final) < 1e-12:
            return self.t0
        if self.temperature_schedule == "linear":
            # linear interpolation from t0 to t_final
            return float(self.t0 + (self.t_final - self.t0) * p)
        if self.temperature_schedule == "cubic":
            # cubic interpolation, slower start, faster end
            return float(self.t0 + (self.t_final - self.t0) * (p ** 3))
        if self.temperature_schedule == "exp":
            # geometric interpolation (monotonic)
            ratio = max(self.t_final, 1e-12) / max(self.t0, 1e-12)
            return float(self.t0 * (ratio ** p))
        return self.t0

    def _current_k(self):
        # p in [0,1]; compress so k finishes warming up by 5% of training
        p = float(self.progress)
        warm = min(p / 0.05, 1.0)  # 0..1 grows only during first 5%
        if self.k_schedule == "constant" or self.k_init == self.k_final:
            return self.k_init
        if self.k_schedule == "linear":
            return int(round(self.k_init + (self.k_final - self.k_init) * warm))
        if self.k_schedule == "cubic":
            return int(round(self.k_init + (self.k_final - self.k_init) * (warm ** 3)))
        if self.k_schedule == "exp":
            ratio = (self.k_final / max(self.k_init, 1)) ** warm
            return int(round(self.k_init * ratio))
        return self.k_init

    # -------- Forward --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A_module.weight
        B = self.B_module.weight

        out = self.base_layer(x)        # base path
        x_lora = self.dropout(x)        # dropout only on LoRA

        z_pre = F.linear(x_lora, A)
        if self.relu_latents:
            z = F.relu(z_pre)
        else:
            z = z_pre

        # keep both: live for regs, detached for callbacks
        # self._z_live = z                         # may carry graph
        self._last_z = z.detach()                # safe for logging

        tau = self._tau()
        k_now = self._current_k()

        # print(tau, k_now, float(self.progress))

        if not self.training and self.hard_eval:
            # eval mode, hard top-k
            z = self.topk(z)
            lora_out = F.linear(z, B) * self.scale
            # print('LoRA contribution:', lora_out.mean().item())
            # print('Base layer contribution:', out.mean().item())
            return out + lora_out

        g_soft = _soft_topk_mass(z, k_now, tau)
        g_hard = _hard_topk_mask(z, k_now)
        # TODO: investigate the gradient magnitude
        # g = g_hard.detach() + g_soft - g_soft.detach()
        g = g_hard + g_soft - g_soft.detach()

        # keep both: live for regs, detached for callbacks
        # self._g_soft_live = g_soft               # may carry graph
        self._last_g_soft = g_soft.detach()

        self._last_ghard_mean = g.mean().detach()

        lora_out = F.linear(z * g, B) * self.scale
        return out + lora_out

    def get_gate_stats(self):
        if self._last_z is None:
            return {}
        k = self._current_k()
        r = self.r
        frac_active = float(self._last_ghard_mean) / max(k / r, 1e-8)
        return {
            "k": k, "r": r,
            "tau": self._tau(),
            "frac_active_vs_target": frac_active,
        }

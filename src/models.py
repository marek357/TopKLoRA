from typing import Optional
from transformers import TrainerCallback
from peft.tuners.lora import LoraLayer
from trl import DPOTrainer
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch
import gc



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
        is_topk_experiment: bool = False
    ):
        super().__init__()
        self.lora_module = base
        self.base_layer = base.base_layer
        adapter = base.active_adapter if isinstance(
            base.active_adapter, str) else base.active_adapter[0]
        self.is_topk_experiment = is_topk_experiment

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
        self._progress_scalar: float = 0.0

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

        self._progress_scalar = float(self.progress.detach().cpu().item())

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

        if isinstance(self.progress, torch.Tensor):
            self._progress_scalar = float(self.progress.detach().cpu().item())

        # Delegate lora weights to the lora_module
        lora_prefix = prefix  # Keep same prefix for transparency
        self.lora_module._load_from_state_dict(
            state_dict, lora_prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    # -------- Progress control --------
    def set_progress(self, p: float):
        """Set training progress in [0, 1]."""
        value = float(min(max(p, 0.0), 1.0))
        self._progress_scalar = value
        self.progress.fill_(value)

    @property
    def progress_scalar(self) -> float:
        return self._progress_scalar

    # -------- Scheduling --------
    def _tau(self):
        p = self._progress_scalar
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
        p = self._progress_scalar
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

        if not self.is_topk_experiment:
            lora_out = F.linear(z_pre, B) * self.scale
            return out + lora_out

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


class MemoryClearCallback(TrainerCallback):
    """Memory management callback"""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.gradient_accumulation_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return control


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



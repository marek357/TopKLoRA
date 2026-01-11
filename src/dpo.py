import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset, interleave_datasets, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import DPOConfig, DPOTrainer

import wandb
from src.models import (
    MemoryClearCallback,
    TopKLoRALinearSTE,
    TopKProgressCallback,
    _soft_topk_mass,
)
from src.sft import count_params, enable_topk_lora_grads
from src.utils import (
    configure_eos_eot,
    ensure_chat_template_and_special_tokens,
    wrap_topk_lora_modules,
    save_cfg_yaml,
    capture_env_snapshot,
    save_summary,
    maybe_update_wandb_config,
    get_local_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    normalize_chat_messages,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

L_DECORR = 1e-4  # decorrelate latents (small)
L_MASS = 1e-3  # enforce soft mass ~= k
L_ENTROPY = 0.0  # encourage sharp gates (set >0 if needed)
L_ORTHO_A = 1e-4  # orthogonality strength on A (rows ~ latents)
L_ORTHO_B = 1e-4  # orthogonality strength on B (columns ~ latents)
ORTHO_EVERY = 4  # compute every step; set to 2/4 to reduce overhead


class ActivationTrackingCallback(TrainerCallback):
    """
    Alternative callback that continuously tracks activation statistics
    and computes dead neurons from accumulated stats.
    """

    def __init__(self, check_interval: int = 1000, reset_interval: int = 5000):
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
            if isinstance(module, TopKLoRALinearSTE):
                self.activation_trackers[name] = {
                    "total_activations": torch.zeros(module.r),
                    "activation_counts": torch.zeros(module.r, dtype=torch.long),
                    "samples_seen": 0,
                    "r": module.r,
                    "k": module.k,
                }

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update activation statistics from each TopK module."""
        if model is None:
            return

        # Collect activation stats from modules
        for name, module in model.named_modules():
            if (
                isinstance(module, TopKLoRALinearSTE)
                and name in self.activation_trackers
            ):
                if module._last_z is not None:
                    with torch.no_grad():
                        # Get activation magnitudes
                        z_abs = module._last_z
                        # Average over batch and sequence dimensions
                        avg_activations = z_abs.mean(dim=(0, 1)).cpu()

                        # Update tracker
                        tracker = self.activation_trackers[name]
                        tracker["total_activations"] += avg_activations
                        tracker["activation_counts"] += (avg_activations > 0.01).long()
                        tracker["samples_seen"] += z_abs.shape[0]

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
        stats_to_log = {"dead_neurons/global_step": state.global_step}

        total_neurons = 0
        total_dead = 0

        for layer_name, tracker in self.activation_trackers.items():
            if tracker["samples_seen"] == 0:
                continue

            # Compute average activations
            # avg_activations = tracker["total_activations"] / tracker["samples_seen"]
            dead_mask = tracker["activation_counts"] == 0
            num_dead = dead_mask.sum().item()

            total_neurons += tracker["r"]
            total_dead += num_dead

            # Log per-layer stats
            clean_name = layer_name.replace(".", "_")
            stats_to_log.update(
                {
                    f"dead_neurons/layers/{clean_name}/num_dead": num_dead,
                    f"dead_neurons/layers/{clean_name}/pct_dead": 100.0
                    * num_dead
                    / tracker["r"],
                    f"dead_neurons/layers/{clean_name}/samples_seen": tracker[
                        "samples_seen"
                    ],
                }
            )

        # Log global stats
        stats_to_log.update(
            {
                "dead_neurons/total_dead": total_dead,
                "dead_neurons/total_pct_dead": 100.0 * total_dead / total_neurons
                if total_neurons > 0
                else 0,
            }
        )

        logging.info(
            f"Step {state.global_step}: {total_dead}/{total_neurons} "
            f"({100.0 * total_dead / total_neurons:.1f}%) dead neurons"
        )

        # Log to wandb
        if args.report_to and "wandb" in args.report_to:
            wandb.log(stats_to_log, step=state.global_step)

    def reset_trackers(self):
        """Reset activation trackers to avoid overflow."""
        for tracker in self.activation_trackers.values():
            tracker["total_activations"].zero_()
            tracker["activation_counts"].zero_()
            tracker["samples_seen"] = 0


# expects: TopKLoRALinearSTE, _soft_topk_mass already defined/imported


class EnhancedDPOTrainer(DPOTrainer):
    """
    DPO trainer with conditional regularizers and DS-safe scheduling.

    reg_mode:
      - "off"           : no regs (only base DPO loss)
      - "z_only"        : decorrelation + mass + entropy (from live z)
      - "z_plus_ortho"  : z_only + orthogonality on A (rows) and B (cols)

    reg_cfg keys (with defaults below):
      L_DECORR, L_MASS, L_ENTROPY, L_ORTHO_A, L_ORTHO_B, ORTHO_EVERY,
      sched_type("linear"/"quad"/"cubic"), sched_start, sched_end,
      schedule_decorr, schedule_mass, schedule_ent, schedule_ortho,
      log_every
    """

    def __init__(
        self,
        *args,
        reg_cfg: Optional[Dict[str, Any]] = None,
        reg_mode: str = "z_only",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # which blocks to enable
        assert reg_mode in {"off", "z_only", "z_plus_ortho"}
        self.reg_mode = reg_mode

        # defaults
        self.reg_cfg = {
            "L_DECORR": 1e-4,
            "L_MASS": 1e-3,
            "L_ENTROPY": 0.0,
            "L_ORTHO_A": 1e-4,
            "L_ORTHO_B": 1e-4,
            # compute orthogonality every n steps (0 disables)
            "ORTHO_EVERY": 10,  # Reduced from 1 to improve performance
            "DECORR_EVERY": 5,  # NEW: Only compute expensive decorrelation every N steps
            "MASS_EVERY": 1,  # Mass is cheap, keep it every step
            "sched_type": "cubic",  # "linear" | "quad" | "cubic"
            "sched_start": 0.0,  # fraction of training (0..1)
            "sched_end": 0.30,  # fraction of training (0..1)
            "schedule_decorr": True,
            "schedule_mass": True,
            "schedule_ent": True,
            "schedule_ortho": True,
            "log_every": 50,
        }
        if reg_cfg:
            self.reg_cfg.update(reg_cfg)

    # ---------- scheduling helpers ----------
    def _sched_scalar(self, p: float) -> float:
        s0 = float(self.reg_cfg["sched_start"])
        s1 = float(self.reg_cfg["sched_end"])
        if s1 <= s0:
            return 1.0  # always on
        if p <= s0:
            t = 0.0
        elif p >= s1:
            t = 1.0
        else:
            t = (p - s0) / (s1 - s0)
        ttype = self.reg_cfg["sched_type"]
        if ttype == "linear":
            return t
        if ttype == "cubic":
            return t**3
        return t**2  # quad default

    # DS-safe: only build term if it will actually contribute
    def _active(self, L: float, scheduled_flag: bool, w_sched: float) -> bool:
        if L <= 0.0:
            return False
        if not scheduled_flag:
            return True
        return w_sched > 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # base DPO loss
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        step = int(self.state.global_step or 0)
        max_steps = int(self.state.max_steps or 1)

        # short-circuit: fully off
        # log if step % log_every == 0
        log_every = int(self.reg_cfg.get("log_every", 50))
        if step % log_every == 0:
            for name, m in model.named_modules():
                if isinstance(m, TopKLoRALinearSTE):
                    st = m.get_gate_stats()
                    if st:
                        self.log(
                            {
                                f"{name}.k": st["k"],
                                f"{name}.tau": st["tau"],
                                f"{name}.frac_active_vs_target": st[
                                    "frac_active_vs_target"
                                ],
                            }
                        )
                        break

        if self.reg_mode == "off":
            # Always clear live caches to avoid cross-step graph retention
            for m in model.modules():
                if isinstance(m, TopKLoRALinearSTE):
                    if hasattr(m, "_z_live"):
                        m._z_live = None
                    if hasattr(m, "_g_soft_live"):
                        m._g_soft_live = None
            return (loss, outputs) if return_outputs else loss

        reg = loss.new_tensor(0.0)

        # logging accumulators
        log_every = int(self.reg_cfg["log_every"])
        do_log = (log_every > 0) and (step % log_every == 0)
        acc = {
            "reg/decorr": 0.0,
            "reg/mass": 0.0,
            "reg/entropy": 0.0,
            "reg/ortho_A": 0.0,
            "reg/ortho_B": 0.0,
            "reg/sched_w": 0.0,
        }
        n_layers = 0

        # pull config once
        L_DECORR = float(self.reg_cfg["L_DECORR"])
        L_MASS = float(self.reg_cfg["L_MASS"])
        L_ENTROPY = float(self.reg_cfg["L_ENTROPY"])
        L_ORTHO_A = float(self.reg_cfg["L_ORTHO_A"])
        L_ORTHO_B = float(self.reg_cfg["L_ORTHO_B"])
        ORTHO_EVERY = int(self.reg_cfg["ORTHO_EVERY"])
        DECORR_EVERY = int(self.reg_cfg.get("DECORR_EVERY", 1))
        MASS_EVERY = int(self.reg_cfg.get("MASS_EVERY", 1))

        try:
            for m in model.modules():
                if not isinstance(m, TopKLoRALinearSTE):
                    continue

                # live tensors from forward (set by your TopK wrapper)
                z_live = getattr(m, "_z_live", None)
                # If we don't have live z (e.g., first steps or different wrapper), skip safely
                if z_live is None:
                    continue

                # layer progress & schedule weight
                try:
                    p_layer = float(m.progress)
                except Exception:
                    p_layer = step / max(1, max_steps)
                w_sched = self._sched_scalar(p_layer)

                # which terms are (potentially) active
                need_decorr = (
                    (DECORR_EVERY > 0)
                    and (step % DECORR_EVERY == 0)
                    and self._active(
                        L_DECORR, self.reg_cfg.get("schedule_decorr", True), w_sched
                    )
                )
                need_mass = (
                    (MASS_EVERY > 0)
                    and (step % MASS_EVERY == 0)
                    and self._active(
                        L_MASS, self.reg_cfg.get("schedule_mass", True), w_sched
                    )
                )
                need_ent = self._active(
                    L_ENTROPY, self.reg_cfg.get("schedule_ent", True), w_sched
                )
                need_orthoA = (
                    (self.reg_mode == "z_plus_ortho")
                    and ORTHO_EVERY > 0
                    and (step % ORTHO_EVERY == 0)
                    and self._active(
                        L_ORTHO_A, self.reg_cfg.get("schedule_ortho", True), w_sched
                    )
                )
                need_orthoB = (
                    (self.reg_mode == "z_plus_ortho")
                    and ORTHO_EVERY > 0
                    and (step % ORTHO_EVERY == 0)
                    and self._active(
                        L_ORTHO_B, self.reg_cfg.get("schedule_ortho", True), w_sched
                    )
                )

                # If every term is scheduled & weight == 0, hard-skip this module (DS-safe)
                all_sched = (
                    self.reg_cfg.get("schedule_decorr", True)
                    and self.reg_cfg.get("schedule_mass", True)
                    and self.reg_cfg.get("schedule_ent", True)
                    and (
                        self.reg_mode != "z_plus_ortho"
                        or self.reg_cfg.get("schedule_ortho", True)
                    )
                )
                if all_sched and (
                    not (
                        need_decorr
                        or need_mass
                        or need_ent
                        or need_orthoA
                        or need_orthoB
                    )
                ):
                    if do_log:
                        acc["reg/sched_w"] += w_sched
                        n_layers += 1
                    continue

                # Prepare gates once (used by mass/entropy and usage balancing)
                k_now = m._current_k()
                tau = m._tau()
                g_soft_live = getattr(m, "_g_soft_live", None)
                g_soft = (
                    g_soft_live
                    if g_soft_live is not None
                    else _soft_topk_mass(z_live, k_now, tau)
                )

                # -------- z-based regs (always allowed in "z_only" and "z_plus_ortho") --------
                if need_decorr:
                    Z = z_live.reshape(-1, z_live.size(-1)).float()
                    Z = Z - Z.mean(dim=0, keepdim=True)
                    C = (Z.T @ Z) / (Z.size(0) + 1e-6)
                    off = C - torch.diag(torch.diag(C))
                    r_decorr = (off**2).mean().to(loss.dtype)
                    if self.reg_cfg.get("schedule_decorr", True):
                        r_decorr = r_decorr * r_decorr.new_tensor(w_sched)
                    reg = reg + L_DECORR * r_decorr
                    if do_log:
                        acc["reg/decorr"] += float(r_decorr.detach().cpu())

                if need_mass or need_ent:
                    # mass and entropy use precomputed g_soft
                    if need_mass:
                        r_mass = (g_soft.sum(dim=-1) - k_now).pow(2).mean()
                        if self.reg_cfg.get("schedule_mass", True):
                            r_mass = r_mass * r_mass.new_tensor(w_sched)
                        reg = reg + L_MASS * r_mass
                        if do_log:
                            acc["reg/mass"] += float(r_mass.detach().cpu())

                    if need_ent:
                        r_ent = (
                            -(g_soft.clamp_min(1e-8) * g_soft.clamp_min(1e-8).log())
                            .sum(dim=-1)
                            .mean()
                        )
                        if self.reg_cfg.get("schedule_ent", True):
                            r_ent = r_ent * r_ent.new_tensor(w_sched)
                        reg = reg + L_ENTROPY * r_ent
                        if do_log:
                            acc["reg/entropy"] += float(r_ent.detach().cpu())

                # -------- orthogonality (only in "z_plus_ortho") --------
                if need_orthoA:
                    Aw = m.A_module.weight
                    # rows ~ latents
                    A_rows = F.normalize(Aw.float(), p=2, dim=1)
                    GA = A_rows @ A_rows.T
                    GA_off = GA - torch.diag(torch.diag(GA))
                    r_oa = (GA_off**2).mean().to(loss.dtype)
                    if self.reg_cfg.get("schedule_ortho", True):
                        r_oa = r_oa * r_oa.new_tensor(w_sched)
                    reg = reg + L_ORTHO_A * r_oa
                    if do_log:
                        acc["reg/ortho_A"] += float(r_oa.detach().cpu())

                if need_orthoB:
                    Bw = m.B_module.weight
                    # cols ~ latents
                    B_cols = F.normalize(Bw.float(), p=2, dim=0)
                    GB = B_cols.T @ B_cols
                    GB_off = GB - torch.diag(torch.diag(GB))
                    r_ob = (GB_off**2).mean().to(loss.dtype)
                    if self.reg_cfg.get("schedule_ortho", True):
                        r_ob = r_ob * r_ob.new_tensor(w_sched)
                    reg = reg + L_ORTHO_B * r_ob
                    if do_log:
                        acc["reg/ortho_B"] += float(r_ob.detach().cpu())

                if do_log:
                    acc["reg/sched_w"] += w_sched
                    n_layers += 1

                # Optional: L1 and usage variance regularization (lightweight)
                # Only apply if needed - comment out for max speed
                # L1 = 1e-5
                # reg = reg + L1 * z_live.abs().mean()
                # usage = g_soft.mean(dim=(0, 1))                 # [r]
                # cov = ((usage - usage.mean())**2).mean()
                # reg = reg + 1e-4 * cov

        finally:
            # critical: drop live caches every step to avoid cross-step graphs
            for m in model.modules():
                if isinstance(m, TopKLoRALinearSTE):
                    if hasattr(m, "_z_live"):
                        m._z_live = None
                    if hasattr(m, "_g_soft_live"):
                        m._g_soft_live = None

        loss = loss + reg

        if do_log and n_layers > 0:
            for k in acc:
                acc[k] /= n_layers
            # also log one layer's gate stats
            for name, m in model.named_modules():
                if isinstance(m, TopKLoRALinearSTE):
                    st = m.get_gate_stats()
                    if st:
                        acc["gates/k"] = st["k"]
                        acc["gates/tau"] = st["tau"]
                        acc["gates/frac_active_vs_target"] = st["frac_active_vs_target"]
                    break
            self.log(acc)

        return (loss, outputs) if return_outputs else loss


def _resolve_device_map_for_ddp() -> Optional[Dict[str, int]]:
    """Resolve a safe device_map when running under DDP/Accelerate.
    try:
        if torch.cuda.is_available():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            num_visible = None
            if visible:
                try:
                    num_visible = len([v for v in visible.split(",") if v.strip() != ""])
                except Exception:
                    num_visible = None
            world_local_count = torch.cuda.device_count()
            # If only one device is visible to this process, always map to 0 (local index)
            if world_local_count == 1 or (num_visible == 1):
                local_idx = 0
            else:
                # Prefer LOCAL_RANK when multiple devices visible
                local_rank_env = os.environ.get("LOCAL_RANK")
                if local_rank_env is not None:
                    local_idx = int(local_rank_env)
                else:
                    local_idx = int(torch.cuda.current_device())
            logging.info(
                f"CUDA_VISIBLE_DEVICES={visible!r}, device_count={world_local_count}, mapping root module to local idx {local_idx}"
            )
            return {"": local_idx}
    except Exception as e:
    """
    try:
        if torch.cuda.is_available():
            local_idx = get_local_rank()
            logging.info(f"Using per-process device_map -> {{'': {local_idx}}}")
            return {"": local_idx}
    except Exception as e:
        logging.warning(
            f"Could not resolve per-process device_map, falling back to default: {e}"
        )
    return None


def _prompt_to_messages(prompt: Any) -> List[Dict[str, str]]:
    """Convert prompt payloads into a normalized message list.

    Supports None, str, dict, or list inputs. Strings are wrapped as a single
    user message; dict/list inputs are normalized with normalize_chat_messages.
    """
    if prompt is None:
        return []
    if isinstance(prompt, list):
        return normalize_chat_messages(prompt)
    if isinstance(prompt, dict):
        return normalize_chat_messages([prompt])
    if isinstance(prompt, str):
        prompt = prompt.strip()
        if not prompt:
            return []
        return [{"role": "user", "content": prompt}]
    return []


def _response_to_messages(
    prompt_messages: List[Dict[str, str]], response: Any
) -> List[Dict[str, str]]:
    """Convert response payloads into message lists, optionally prepending prompt.

    If the response is already a list/dict of messages and begins with an assistant
    role, the prompt_messages are prepended to preserve chat context.
    """
    if response is None:
        return []
    if isinstance(response, list):
        normalized = normalize_chat_messages(response)
        if not normalized:
            return []
        if normalized[0]["role"] == "assistant" and prompt_messages:
            return prompt_messages + normalized
        return normalized
    if isinstance(response, dict):
        normalized = normalize_chat_messages([response])
        if not normalized:
            return []
        if normalized[0]["role"] == "assistant" and prompt_messages:
            return prompt_messages + normalized
        return normalized
    if isinstance(response, str):
        response = response.strip()
        if not response:
            return []
        return prompt_messages + [{"role": "assistant", "content": response}]
    return []


def _is_valid_messages(messages: List[Dict[str, str]]) -> bool:
    """Return True for non-empty message lists ending with an assistant reply."""
    return bool(messages) and messages[-1].get("role") == "assistant"


def _apply_pairwise_choice(
    ex: Dict[str, Any],
    *,
    response_a_field: str,
    response_b_field: str,
    choice_field: str,
    choice_value_for_a: Any = 0,
) -> Dict[str, Any]:
    """Map pairwise preference fields into chosen/rejected responses."""
    resp_a = ex.get(response_a_field)
    resp_b = ex.get(response_b_field)
    choice = ex.get(choice_field)
    choose_a = choice == choice_value_for_a
    chosen = resp_a if choose_a else resp_b
    rejected = resp_b if choose_a else resp_a
    return {"chosen": chosen, "rejected": rejected}


def _prepare_preference_dataset(
    *,
    dataset_id: str,
    split: str,
    tokenizer,
    prompt_field: str | None = None,
    chosen_field: str | None = None,
    rejected_field: str | None = None,
    response_a_field: str | None = None,
    response_b_field: str | None = None,
    choice_field: str | None = None,
    choice_value_for_a: Any = 0,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
) -> HFDataset:
    """Prepare a preference dataset for DPO.

    Supports two formats:
      1) prompt + chosen/rejected fields
      2) prompt + response_a/response_b + choice_field

    Returns a dataset with "chosen" and "rejected" message lists. Optional length
    filtering is applied when tokenizer + length constraints are provided.
    """
    if not (
        (chosen_field and rejected_field)
        or (response_a_field and response_b_field and choice_field)
    ):
        raise ValueError(
            "Preference dataset configuration must specify either chosen/rejected "
            "fields or response_a/response_b with a choice_field."
        )
    ds = load_dataset(dataset_id, split=split)

    def to_pairs(ex):
        prompt = ex.get(prompt_field) if prompt_field else None
        if chosen_field and rejected_field:
            chosen = ex.get(chosen_field)
            rejected = ex.get(rejected_field)
        elif response_a_field and response_b_field and choice_field:
            pair = _apply_pairwise_choice(
                ex,
                response_a_field=response_a_field,
                response_b_field=response_b_field,
                choice_field=choice_field,
                choice_value_for_a=choice_value_for_a,
            )
            chosen = pair["chosen"]
            rejected = pair["rejected"]
        else:
            return {"chosen": [], "rejected": []}

        prompt_messages = _prompt_to_messages(prompt)
        chosen_messages = _response_to_messages(prompt_messages, chosen)
        rejected_messages = _response_to_messages(prompt_messages, rejected)
        return {"chosen": chosen_messages, "rejected": rejected_messages}

    ds = ds.map(to_pairs, remove_columns=ds.column_names)
    ds = ds.filter(
        lambda ex: _is_valid_messages(ex.get("chosen", []))
        and _is_valid_messages(ex.get("rejected", []))
    )

    if tokenizer and max_prompt_length and max_completion_length:
        if not getattr(tokenizer, "chat_template", None):
            logging.warning(
                "Tokenizer has no chat template configured; skipping DPO length "
                "filtering based on chat-formatted prompts."
            )
        else:
            def length_ok(ex):
                chosen = ex["chosen"]
                rejected = ex["rejected"]
                prompt = chosen[:-1]
                chosen_resp = chosen[-1]["content"]
                rejected_resp = rejected[-1]["content"]
                prompt_ids = tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=True
                )
                chosen_ids = tokenizer(chosen_resp, add_special_tokens=False)["input_ids"]
                rejected_ids = tokenizer(rejected_resp, add_special_tokens=False)["input_ids"]
                return (
                    len(prompt_ids) <= max_prompt_length
                    and len(chosen_ids) <= max_completion_length
                    and len(rejected_ids) <= max_completion_length
                )

            ds = ds.filter(length_ok)

    return ds


def _mix_dpo_datasets(
    datasets: List[HFDataset],
    *,
    weights: Optional[List[float]] = None,
    seed: int = 42,
) -> HFDataset:
    """Interleave multiple DPO datasets with optional weighting.

    When weights are provided, they are normalized into sampling probabilities.
    If weights are omitted, datasets are mixed equally. The "all_exhausted"
    strategy continues until all datasets are exhausted. Seed controls shuffling.
    """
    if not datasets:
        raise ValueError("No datasets provided for DPO mixing.")
    if not weights:
        return interleave_datasets(
            datasets, seed=seed, stopping_strategy="all_exhausted"
        )
    total = sum(weights)
    if total <= 0 or all(w == 0 for w in weights):
        logging.warning(
            "All provided DPO mixing weights are zero or sum to zero; "
            "defaulting to equal mixing probabilities across datasets."
        )
        probabilities = [1.0 / len(datasets)] * len(datasets)
    else:
        probabilities = [w / total for w in weights]
    return interleave_datasets(
        datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy="all_exhausted",
    )


def _get_cfg_value(cfg_obj: Any, key: str, default: Any = None) -> Any:
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def _load_single_dpo_dataset(
    spec: Any,
    *,
    tokenizer,
    dpo_args,
    seed: int = 42,
) -> Tuple[HFDataset, HFDataset]:
    """Load one DPO dataset from a dataset spec.

    Supports:
      - Special loaders: hh-rlhf, civil_comments
      - Generic HuggingFace datasets with prompt/chosen/rejected or
        prompt/response_a/response_b/choice fields.

    Config fields supported in spec:
      name, huggingface_dataset_id, train_split, eval_split, prompt_field,
      chosen_field, rejected_field, response_a_field, response_b_field,
      choice_field, choice_value_for_a, train_size, eval_size.

    Returns (train_dataset, eval_dataset), applying size limits when specified.
    """
    name = _get_cfg_value(spec, "name")
    if name == "hh-rlhf":
        return prepare_hh_rlhf_datasets(
            max_length=dpo_args.max_prompt_length,
            tokenizer=tokenizer,
            max_prompt_length=dpo_args.max_prompt_length,
            max_completion_length=dpo_args.max_completion_length,
            train_size=_get_cfg_value(spec, "train_size"),
            eval_size=_get_cfg_value(spec, "eval_size", 100),
        )
    if name == "civil_comments":
        return prepare_civil_comments_datasets(
            tokenizer=tokenizer,
            max_prompt_length=_get_cfg_value(
                spec, "max_prompt_length", dpo_args.max_prompt_length
            ),
            max_completion_length=_get_cfg_value(
                spec, "max_completion_length", dpo_args.max_completion_length
            ),
            train_size=_get_cfg_value(spec, "train_size"),
            eval_size=_get_cfg_value(spec, "eval_size", 100),
            label_field=_get_cfg_value(spec, "label_field", "toxicity"),
            threshold=float(_get_cfg_value(spec, "threshold", 0.5)),
            approval_token=_get_cfg_value(spec, "approval_token", "APPROVE"),
            rejection_token=_get_cfg_value(spec, "rejection_token", "REJECT"),
        )

    dataset_id = _get_cfg_value(spec, "huggingface_dataset_id")
    if not dataset_id:
        raise ValueError(
            f"Missing huggingface_dataset_id for dataset '{name}'. "
            "Please add 'huggingface_dataset_id: <dataset-path>' "
            "to the dataset configuration in your DPO recipe file."
        )
    train_split = _get_cfg_value(spec, "train_split", "train")
    eval_split = _get_cfg_value(spec, "eval_split", "test")
    prompt_field = _get_cfg_value(spec, "prompt_field", "prompt")
    chosen_field = _get_cfg_value(spec, "chosen_field", "chosen")
    rejected_field = _get_cfg_value(spec, "rejected_field", "rejected")
    response_a_field = _get_cfg_value(spec, "response_a_field")
    response_b_field = _get_cfg_value(spec, "response_b_field")
    choice_field = _get_cfg_value(spec, "choice_field")
    choice_value_for_a = _get_cfg_value(spec, "choice_value_for_a", 0)

    train_dataset = _prepare_preference_dataset(
        dataset_id=dataset_id,
        split=train_split,
        tokenizer=tokenizer,
        prompt_field=prompt_field,
        chosen_field=chosen_field,
        rejected_field=rejected_field,
        response_a_field=response_a_field,
        response_b_field=response_b_field,
        choice_field=choice_field,
        choice_value_for_a=choice_value_for_a,
        max_prompt_length=dpo_args.max_prompt_length,
        max_completion_length=dpo_args.max_completion_length,
    )
    eval_dataset = _prepare_preference_dataset(
        dataset_id=dataset_id,
        split=eval_split,
        tokenizer=tokenizer,
        prompt_field=prompt_field,
        chosen_field=chosen_field,
        rejected_field=rejected_field,
        response_a_field=response_a_field,
        response_b_field=response_b_field,
        choice_field=choice_field,
        choice_value_for_a=choice_value_for_a,
        max_prompt_length=dpo_args.max_prompt_length,
        max_completion_length=dpo_args.max_completion_length,
    )
    if train_split == eval_split:
        test_size = eval_size or 0.01
        if isinstance(test_size, int):
            test_size = min(test_size, len(train_dataset))
        split = train_dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    train_size = _get_cfg_value(spec, "train_size")
    eval_size = _get_cfg_value(spec, "eval_size")
    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if eval_size:
        eval_dataset = eval_dataset.select(range(min(eval_size, len(eval_dataset))))
    return train_dataset, eval_dataset


def load_dpo_datasets_from_cfg(cfg, tokenizer, dpo_args) -> Tuple[HFDataset, HFDataset]:
    """Load DPO datasets from cfg.

    If cfg.training.dpo_dataset.name != "mix", a single dataset is loaded.
    If name == "mix", datasets are loaded from the "datasets" list and mixed
    using their weights with deterministic interleaving.
    """
    dpo_cfg = cfg.training.dpo_dataset
    name = _get_cfg_value(dpo_cfg, "name")
    if name != "mix":
        seed = int(_get_cfg_value(dpo_cfg, "seed", 42))
        return _load_single_dpo_dataset(
            dpo_cfg, tokenizer=tokenizer, dpo_args=dpo_args, seed=seed
        )

    datasets_cfg = _get_cfg_value(dpo_cfg, "datasets", [])
    if not datasets_cfg:
        raise ValueError(
            "DPO mix requested but no datasets were provided in "
            "cfg.training.dpo_dataset.datasets. Please add at least one dataset "
            "to the 'datasets' list in your configuration."
        )
    seed = int(_get_cfg_value(dpo_cfg, "seed", 42))
    train_sets: List[HFDataset] = []
    eval_sets: List[HFDataset] = []
    weights: List[float] = []
    for spec in datasets_cfg:
        train_ds, eval_ds = _load_single_dpo_dataset(
            spec, tokenizer=tokenizer, dpo_args=dpo_args, seed=seed
        )
        train_sets.append(train_ds)
        eval_sets.append(eval_ds)
        weights.append(float(_get_cfg_value(spec, "weight", 1.0)))

    train_dataset = _mix_dpo_datasets(train_sets, weights=weights, seed=seed)
    eval_dataset = _mix_dpo_datasets(eval_sets, weights=weights, seed=seed)
    return train_dataset, eval_dataset


def prepare_hh_rlhf_datasets(
    max_length=1024,
    train_size=None,
    eval_size=100,
    tokenizer=None,
    max_prompt_length=512,
    max_completion_length=512,
):
    """Load and prepare Anthropic/hh-rlhf for reference-free DPO."""
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    # eos = tokenizer.eos_token

    ASSISTANT = "Assistant:"

    def split_reply(text: str):
        i = text.rfind(ASSISTANT)
        if i == -1:
            raise ValueError("No 'Assistant:' in example.")
        prompt = text[: i + len(ASSISTANT)]
        reply = text[i + len(ASSISTANT) :].strip()
        return prompt, reply

    # def format_hh(samples):
    #     prompts, chosens, rejecteds = [], [], []
    #     for c, r in zip(samples["chosen"], samples["rejected"]):
    #         p_c, ch = split_reply(c)
    #         p_r, rj = split_reply(r)
    #         # normalize whitespace before comparing
    #         if p_c.strip() != p_r.strip() or not ch or not rj:
    #             continue
    #         # ensure EOS termination
    #         if eos and not ch.endswith(eos):
    #             ch = ch + " " + eos
    #         if eos and not rj.endswith(eos):
    #             rj = rj + " " + eos
    #         p_c = re.sub(r"^\s*\n*", "", p_c)
    #         prompts.append(p_c)
    #         chosens.append(ch)
    #         rejecteds.append(rj)
    #     return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    def format_hh(samples):
        """Return message lists for DPOTrainer to process.

        DPOTrainer expects 'chosen' and 'rejected' fields containing message lists,
        and it will apply the chat template itself.
        """
        chosens, rejecteds = [], []

        for c, r in zip(samples["chosen"], samples["rejected"]):
            # Parse the conversation from Anthropic format
            c_messages = parse_anthropic_to_messages(c)
            r_messages = parse_anthropic_to_messages(r)

            if not c_messages or not r_messages:
                continue

            # Both should end with assistant responses
            if (
                c_messages[-1]["role"] != "assistant"
                or r_messages[-1]["role"] != "assistant"
            ):
                continue

            # Verify same conversation context (all messages except last assistant)
            if len(c_messages) != len(r_messages):
                continue

            c_prompt_msgs = c_messages[:-1]
            r_prompt_msgs = r_messages[:-1]

            # Verify prompts match
            if c_prompt_msgs != r_prompt_msgs:
                continue

            # Return full message lists - DPOTrainer will apply the chat template
            chosens.append(c_messages)
            rejecteds.append(r_messages)

        return {"chosen": chosens, "rejected": rejecteds}

    def parse_anthropic_to_messages(text):
        """Parse Anthropic's Human:/Assistant: format to messages list."""
        messages = []
        lines = text.split("\n")
        current_role = None
        current_content = []

        for line in lines:
            if line.startswith("Human: "):
                if current_role:
                    messages.append(
                        {
                            "role": "assistant"
                            if current_role == "Assistant"
                            else "user",
                            "content": "\n".join(current_content).strip(),
                        }
                    )
                current_role = "Human"
                current_content = [line[7:]]  # Remove "Human: "
            elif line.startswith("Assistant: "):
                if current_role:
                    messages.append(
                        {
                            "role": "assistant"
                            if current_role == "Assistant"
                            else "user",
                            "content": "\n".join(current_content).strip(),
                        }
                    )
                current_role = "Assistant"
                current_content = [line[11:]]  # Remove "Assistant: "
            else:
                current_content.append(line)

        # Add the last message
        if current_role:
            messages.append(
                {
                    "role": "assistant" if current_role == "Assistant" else "user",
                    "content": "\n".join(current_content).strip(),
                }
            )

        return messages

    logging.info("Loading harmless PM split")
    base_dataset = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="harmless-base",
        # data_dir="helpful-base",
        cache_dir=cache_dir,
    )

    train_dataset = base_dataset["train"]
    eval_dataset = base_dataset["test"]

    # apply formatting
    train_dataset = train_dataset.map(
        format_hh,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Formatting HH train",
    )
    eval_dataset = eval_dataset.map(
        format_hh,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Formatting HH eval",
    )

    def ok(ex):
        """Validate message list format.

        Since we're now passing message lists to DPOTrainer,
        we just need to ensure they're valid message lists.
        """
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # Must be lists
        if not isinstance(chosen, list) or not isinstance(rejected, list):
            return False

        # Must have at least one message
        if len(chosen) == 0 or len(rejected) == 0:
            return False

        # Must end with assistant message
        if (
            chosen[-1].get("role") != "assistant"
            or rejected[-1].get("role") != "assistant"
        ):
            return False

        # Responses must not be empty
        if (
            not chosen[-1].get("content", "").strip()
            or not rejected[-1].get("content", "").strip()
        ):
            return False

        return True

    logging.info("Before filtering dataset size...")
    logging.info(f"Train size before filtering: {len(train_dataset)}")
    logging.info(f"Eval size before filtering: {len(eval_dataset)}")
    logging.info(f"Dataset example: {train_dataset[0]}")

    train_dataset = train_dataset.filter(ok)
    eval_dataset = eval_dataset.filter(ok)

    max_p = max_prompt_length
    max_c = max_completion_length

    def length_ok(ex):
        """Check length after applying chat template.

        Since DPOTrainer will apply the template, we need to simulate that
        to check the final tokenized length.
        """
        try:
            chosen = ex["chosen"]
            rejected = ex["rejected"]

            # Get prompt (all messages except last)
            prompt_messages = chosen[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize to check lengths
            p_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]

            # Response is the difference between full text and prompt
            # For simplicity, just tokenize the last message content
            c_response = chosen[-1]["content"]
            r_response = rejected[-1]["content"]

            c_ids = tokenizer(c_response, add_special_tokens=False)["input_ids"]
            r_ids = tokenizer(r_response, add_special_tokens=False)["input_ids"]

            return len(p_ids) <= max_p and len(c_ids) <= max_c and len(r_ids) <= max_c
        except Exception:
            # If template application fails, filter out
            return False

    train_dataset = train_dataset.filter(length_ok)
    eval_dataset = eval_dataset.filter(length_ok)

    # (Optional) filter out very short replies
    train_dataset = train_dataset.filter(
        lambda ex: len(ex["chosen"][-1]["content"]) > 10
        and len(ex["rejected"][-1]["content"]) > 10
    )
    eval_dataset = eval_dataset.filter(
        lambda ex: len(ex["chosen"][-1]["content"]) > 0
        and len(ex["rejected"][-1]["content"]) > 0
    )

    # Decrease dataset sizes
    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if eval_size:
        eval_dataset = eval_dataset.select(range(min(eval_size, len(eval_dataset))))

    logging.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def _stable_hash(obj: Any) -> str:
    try:
        payload = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        payload = str(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_git_info() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return out.stdout.decode().strip()
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _run(["git", "status", "--porcelain"]),
        "is_dirty": bool(_run(["git", "status", "--porcelain"]))
        if _run(["git", "rev-parse", "--git-dir"])
        else None,
    }


def _collect_hparams(
    cfg,
    *,
    dpo_args,
    experiment_args,
    tokenizer,
    quant_cfg,
    target_modules: List[str],
    train_size: int,
    eval_size: int,
) -> Dict[str, Any]:
    # versions
    try:
        import transformers as _tf

        transformers_ver = getattr(_tf, "__version__", None)
    except Exception:
        transformers_ver = None
    try:
        import peft as _peft

        peft_ver = getattr(_peft, "__version__", None)
    except Exception:
        peft_ver = None
    try:
        import trl as _trl

        trl_ver = getattr(_trl, "__version__", None)
    except Exception:
        trl_ver = None
    try:
        import datasets as _ds

        datasets_ver = getattr(_ds, "__version__", None)
    except Exception:
        datasets_ver = None

    lora = experiment_args.lora

    topk_cfg = {
        "r": int(lora.r),
        "k": int(lora.k),
        "k_final": int(getattr(lora, "k_final", lora.k) or lora.k),
        "temperature": float(getattr(lora, "temperature", 1.0)),
        "temperature_final": float(
            getattr(lora, "temperature_final", 0.1 * getattr(lora, "temperature", 1.0))
        ),
        "temperature_schedule": getattr(lora, "temperature_schedule", "linear"),
        "k_schedule": getattr(lora, "k_schedule", "constant"),
        "target_modules": list(target_modules),
        "alpha": float(getattr(lora, "alpha", getattr(lora, "lora_alpha", 16))),
        "dropout": float(getattr(lora, "dropout", 0.05)),
    }

    quant = None
    try:
        if quant_cfg is not None:
            # BitsAndBytesConfig is not trivially serializable, extract key fields when present
            quant = {
                k: getattr(quant_cfg, k)
                for k in [
                    "load_in_4bit",
                    "load_in_8bit",
                    "bnb_4bit_compute_dtype",
                    "bnb_4bit_use_double_quant",
                    "bnb_4bit_quant_type",
                    "llm_int8_threshold",
                    "llm_int8_enable_fp32_cpu_offload",
                ]
                if hasattr(quant_cfg, k)
            }
    except Exception:
        pass

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count(),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "pytorch": torch.__version__,
        "transformers": transformers_ver,
        "peft": peft_ver,
        "trl": trl_ver,
        "datasets": datasets_ver,
    }

    tok_info = {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": len(tokenizer.get_vocab())
        if hasattr(tokenizer, "get_vocab")
        else None,
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "special_tokens_map": tokenizer.special_tokens_map
        if hasattr(tokenizer, "special_tokens_map")
        else None,
    }

    base_model_path = cfg.training.base_sft_merged_model.checkpoint_dir

    hparams = {
        "experiment_name": getattr(cfg, "experiment_name", None),
        "task": "dpo_topk_lora",
        "model": {
            "base": base_model_path,
            "model_name": cfg.training.model.model_name,
        },
        "dataset": {
            "name": cfg.training.dpo_dataset.name,
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "max_prompt_length": int(getattr(dpo_args, "max_prompt_length", 0)),
            "max_completion_length": int(getattr(dpo_args, "max_completion_length", 0)),
        },
        "dpo": {
            "beta": float(dpo_args.beta),
            "learning_rate": float(dpo_args.learning_rate),
            "max_steps": int(dpo_args.max_steps),
            "per_device_train_batch_size": int(dpo_args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(dpo_args.gradient_accumulation_steps),
            "warmup_ratio": float(dpo_args.warmup_ratio),
            "eval_steps": int(dpo_args.eval_steps),
            "save_steps": int(dpo_args.save_steps),
        },
        "lora_topk": topk_cfg,
        "quantization": quant,
        "tokenizer": tok_info,
        "logger": getattr(cfg, "logger", None).__dict__
        if hasattr(getattr(cfg, "logger", None), "__dict__")
        else str(getattr(cfg, "logger", None)),
        "env": env,
        "git": _get_git_info(),
        "seeds": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "torch_seed": None,  # set by caller if needed
        },
    }
    return hparams


def _make_run_dir(
    base_dir: str, model_name: str, tag: str, hparams: Dict[str, Any]
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = _stable_hash(hparams)[:8]
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name)
    run_dir = os.path.join(base_dir, f"{safe_model}_{tag}_{ts}_{fp}")
    os.makedirs(run_dir, exist_ok=True)

    # Persist metadata
    try:
        with open(os.path.join(run_dir, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2, default=str)
    except Exception as e:
        logging.warning(f"Failed to write hparams.json: {e}")

    # Try to persist cfg as YAML if OmegaConf is available
    try:
        from omegaconf import OmegaConf

        try:
            cfg_dict = OmegaConf.to_container(hparams.get("cfg", {}), resolve=True)
        except Exception:
            cfg_dict = None
        if cfg_dict:
            with open(os.path.join(run_dir, "cfg.yaml"), "w") as f:
                import yaml

                yaml.safe_dump(cfg_dict, f, sort_keys=False)
    except Exception:
        pass

    return run_dir


def prepare_civil_comments_datasets(
    *,
    tokenizer,
    max_prompt_length=512,
    max_completion_length=16,
    train_size=None,
    eval_size=100,
    label_field: str = "toxicity",
    threshold: float = 0.5,
    approval_token: str = "APPROVE",
    rejection_token: str = "REJECT",
):
    """
    Load and prepare google/civil_comments (or civil_comments) as DPO pairs.
    For each comment, the prompt asks for a single-token moderation decision.
    The preferred completion is APPROVE for non-toxic and REJECT for toxic.

    Columns expected:
      - text column: one of ["text", "comment_text"] (auto-detected)
      - label field: default "toxicity" (float in [0,1])
    """
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    eos = tokenizer.eos_token or ""

    # Robust dataset load (namespace may vary on HF)
    try:
        base = load_dataset("google/civil_comments", cache_dir=cache_dir)
    except Exception:
        base = load_dataset("civil_comments", cache_dir=cache_dir)

    # Pick splits
    train_split = "train" if "train" in base else list(base.keys())[0]
    eval_split = (
        "validation"
        if "validation" in base
        else ("test" if "test" in base else train_split)
    )

    train_ds = base[train_split]
    eval_ds = base[eval_split]

    # Detect text column
    text_col = None
    for cand in ["text", "comment_text", "comment"]:
        if cand in train_ds.column_names:
            text_col = cand
            break
    if text_col is None:
        raise ValueError(f"Could not find a text column in {train_ds.column_names}")

    if label_field not in train_ds.column_names:
        raise ValueError(
            f"Label field '{label_field}' not found. Available: {train_ds.column_names}"
        )

    ASSISTANT = "Assistant:"
    INSTRUCTIONS = (
        "You are a content-moderation assistant. "
        f"Read the user comment and answer with a single token: {approval_token} if it's civil/acceptable, "
        f"or {rejection_token} if it violates toxicity guidelines.\n\n"
    )

    def make_pairs(batch):
        prompts, chosens, rejecteds = [], [], []
        texts = batch[text_col]
        labels = batch[label_field]

        for t, y in zip(texts, labels):
            if t is None:
                continue
            try:
                tox = float(y)
            except Exception:
                continue

            is_toxic = tox >= threshold
            chosen = rejection_token if is_toxic else approval_token
            rejected = approval_token if is_toxic else rejection_token

            prompt = f"{INSTRUCTIONS}Comment:\n{t}\n\n{ASSISTANT}"

            # ensure EOS termination
            ch = (chosen + " " + eos).strip() if eos else chosen
            rj = (rejected + " " + eos).strip() if eos else rejected

            prompts.append(prompt)
            chosens.append(ch)
            rejecteds.append(rj)

        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    # Map to DPO columns
    train_ds = train_ds.map(
        make_pairs,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Formatting Civil Comments train for DPO",
    )
    eval_ds = eval_ds.map(
        make_pairs,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Formatting Civil Comments eval for DPO",
    )

    # Basic sanity filters
    def ok(ex):
        p = ex["prompt"]
        return (
            p.rstrip().endswith(ASSISTANT)
            and len(ex["chosen"]) > 0
            and len(ex["rejected"]) > 0
        )

    train_ds = train_ds.filter(ok)
    eval_ds = eval_ds.filter(ok)

    # Length constraints (prompt/completions)
    max_p = max_prompt_length
    max_c = max_completion_length

    def length_ok(ex):
        p_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(ex["chosen"], add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(ex["rejected"], add_special_tokens=False)["input_ids"]
        return len(p_ids) <= max_p and len(c_ids) <= max_c and len(r_ids) <= max_c

    train_ds = train_ds.filter(length_ok)
    eval_ds = eval_ds.filter(length_ok)

    # Optional downsampling
    if train_size:
        train_ds = train_ds.select(range(min(train_size, len(train_ds))))
    if eval_size:
        eval_ds = eval_ds.select(range(min(eval_size, len(eval_ds))))

    logging.info(
        f"Civil Comments -> Train size: {len(train_ds)}, Eval size: {len(eval_ds)}"
    )
    return train_ds, eval_ds


def run_dpo(cfg, quant_cfg):
    dpo_args = cfg.training.dpo
    experiment_args = cfg.training.dpo_experiment
    init_distributed()
    world_size = get_world_size()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    if device == "mps":
        quant_cfg = None

    # Load tokenizer
    logging.info(
        f"Loading tokenizer from {cfg.training.base_sft_merged_model.checkpoint_dir}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.training.base_sft_merged_model.checkpoint_dir
    )

    # Load policy model
    logging.info("Loading policy model...")
    _device_map = _resolve_device_map_for_ddp() if device == "cuda" else None
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    attn_impl = "sdpa" if device == "cuda" else "eager"
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.training.base_sft_merged_model.checkpoint_dir,
        torch_dtype=torch_dtype,
        device_map=_device_map,  # per-process placement; avoid auto-sharding under DDP
        trust_remote_code=True,
        quantization_config=quant_cfg,
        attn_implementation=attn_impl,  # lower memory than eager when available
    )
    if quant_cfg is not None:
        policy_model = prepare_model_for_kbit_training(policy_model)
    if device == "mps":
        policy_model.to(device)
    # Respect config flag for gradient checkpointing to reduce activation memory
    try:
        if getattr(cfg.training.dpo, "gradient_checkpointing", False):
            policy_model.gradient_checkpointing_enable()
            if hasattr(policy_model, "enable_input_require_grads"):
                policy_model.enable_input_require_grads()
            policy_model.config.use_cache = False
    except Exception as e:
        logging.warning(f"Could not enable gradient checkpointing: {e}")
    new_tokens = ensure_chat_template_and_special_tokens(
        tokenizer,
        policy_model,
        cfg.training.base_sft_merged_model.model_it_name,
    )
    eot_token, eot_token_id = configure_eos_eot(tokenizer, policy_model)

    # Log the configuration
    logging.info(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    logging.info(f"EOS token ID(s): {policy_model.generation_config.eos_token_id}")

    # Load reference model
    logging.info("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.training.base_sft_merged_model.checkpoint_dir,
        torch_dtype=torch_dtype,
        device_map=_device_map,  # match policy placement per process
        trust_remote_code=True,
        quantization_config=quant_cfg,
        attn_implementation=attn_impl,
    )
    if device == "mps":
        ref_model.to(device)
    # Avoid cache growth on the ref model as well
    try:
        ref_model.config.use_cache = False
    except Exception:
        pass

    # Sanity logs about device placement and memory state
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        logging.info(
            f"Policy first param device: {next(policy_model.parameters()).device}; Ref first param device: {next(ref_model.parameters()).device}; current_device={dev}"
        )
        mem = torch.cuda.memory_allocated()
        rsv = torch.cuda.memory_reserved()
        logging.info(
            f"CUDA mem at start: allocated={mem / 1e9:.2f} GB, reserved={rsv / 1e9:.2f} GB"
        )

    ref_model.generation_config.eos_token_id = (
        policy_model.generation_config.eos_token_id
    )

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if new_tokens:
        ref_model.resize_token_embeddings(len(tokenizer))
        logging.info("Added %d extra special tokens to ref model", len(new_tokens))

    train_dataset, eval_dataset = load_dpo_datasets_from_cfg(
        cfg, tokenizer=tokenizer, dpo_args=dpo_args
    )

    # Configure LoRA
    target_modules = list(experiment_args.lora.target_modules)
    logging.info(f"Target modules: {len(target_modules)} modules")

    lora_config = LoraConfig(
        r=experiment_args.lora.r,
        lora_alpha=experiment_args.lora.alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(policy_model, lora_config)
    model.config.use_cache = False

    initialised = 0
    for mod in model.modules():
        if isinstance(mod, LoraLayer):
            for lora_B in mod.lora_B.values():
                if hasattr(lora_B, "weight"):
                    nn.init.normal_(lora_B.weight, mean=0.0, std=1e-3)
                    initialised += 1
    logging.info(f"Initialized {initialised} LoRA layers with normal distribution")

    # Inject TopK wrappers (aligned with SFT pattern)
    logging.info("🔥 Injecting TopKLoRALinearSTE wrappers...")
    replaced, _ = wrap_topk_lora_modules(
        model,
        k=experiment_args.lora.k,
        temperature=experiment_args.lora.temperature,
        temperature_schedule=experiment_args.lora.temperature_schedule,
        k_schedule=experiment_args.lora.k_schedule,
        k_final=experiment_args.lora.k_final,
        temperature_final=getattr(experiment_args.lora, "temperature_final", None),
        is_topk_experiment=experiment_args.lora.get("top_k_experiment", False),
        set_train=True,
    )
    logging.info(f"✅ Injected TopK STE wrappers in {replaced} layers")
    model.print_trainable_parameters()

    # Configure gradients for TopK training
    enable_topk_lora_grads(model)
    logging.info("Model after TopK injection and gradient setup:")
    count_params(model)

    # Build structured hparams and output_dir
    hparams = _collect_hparams(
        cfg,
        dpo_args=dpo_args,
        experiment_args=experiment_args,
        tokenizer=tokenizer,
        quant_cfg=quant_cfg,
        target_modules=target_modules,
        train_size=len(train_dataset),
        eval_size=len(eval_dataset),
    )
    output_dir = _make_run_dir(
        cfg.training.dump_path,
        cfg.training.model.model_name,
        tag="topk_dpo",
        hparams=hparams,
    )
    logging.info(f"Run artifacts will be saved under: {output_dir}")

    if is_main_process():
        # Mark this as the latest run via symlink (best-effort)
        try:
            latest_link = os.path.join(cfg.training.dump_path, "latest")
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                try:
                    os.remove(latest_link)
                except Exception as e:
                    logging.warning(f"Could not remove existing 'latest' link '{latest_link}': {e}")
            os.symlink(output_dir, latest_link)
        except Exception as e:
            logging.warning(f"Could not create 'latest' symlink: {e}")

        save_cfg_yaml(output_dir, cfg)
        capture_env_snapshot(output_dir)

    # Human-readable summary
    try:
        summary = []
        summary.append(f"model: {cfg.training.model.model_name}")
        summary.append(f"base: {cfg.training.base_sft_merged_model.checkpoint_dir}")
        summary.append(
            f"dataset: {cfg.training.dpo_dataset.name} (train={len(train_dataset)}, eval={len(eval_dataset)})"
        )
        summary.append(
            f"lora_topk: r={experiment_args.lora.r}, k={experiment_args.lora.k}, k_final={getattr(experiment_args.lora, 'k_final', experiment_args.lora.k)}, "
            f"alpha={getattr(experiment_args.lora, 'alpha', getattr(experiment_args.lora, 'lora_alpha', 16))}, temp={getattr(experiment_args.lora, 'temperature', 1.0)}"
        )
        summary.append(
            f"dpo: beta={dpo_args.beta}, lr={dpo_args.learning_rate}, steps={dpo_args.max_steps}, bs={dpo_args.per_device_train_batch_size}x{dpo_args.gradient_accumulation_steps}"
        )
        if is_main_process():
            save_summary(output_dir, summary)
    except Exception as e:
        logging.warning(f"Failed to build summary README.txt: {e}")

    run_name = getattr(cfg, "experiment_name", None) or os.path.basename(output_dir)
    if is_main_process():
        maybe_update_wandb_config(cfg.logger, hparams, run_name)

    # DPO configuration
    ddp_backend = "nccl" if world_size > 1 and device == "cuda" else None
    ddp_find_unused = False if world_size > 1 else None
    dpo_config = DPOConfig(
        output_dir=output_dir,
        # num_train_epochs=args.epochs,
        reference_free=False,
        per_device_train_batch_size=dpo_args.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_args.gradient_accumulation_steps,
        learning_rate=dpo_args.learning_rate,
        max_steps=dpo_args.max_steps,
        beta=dpo_args.beta,
        lr_scheduler_type="cosine",
        warmup_ratio=dpo_args.warmup_ratio,
        logging_steps=5,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=dpo_args.eval_steps,
        save_steps=dpo_args.save_steps,
        bf16=True if device == "cuda" else False,
        report_to=cfg.logger.report_to,
        run_name=cfg.experiment_name,
        remove_unused_columns=False,
        max_grad_norm=0.5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/accuracies",
        greater_is_better=True,
        save_total_limit=3,
        ddp_backend=ddp_backend,
        ddp_find_unused_parameters=ddp_find_unused,
    )

    # Persist training args for reproducibility
    if is_main_process():
        try:
            with open(os.path.join(output_dir, "dpo_config.json"), "w") as f:
                json.dump(
                    dpo_config.to_dict()
                    if hasattr(dpo_config, "to_dict")
                    else dpo_config.__dict__,
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            logging.warning(f"Failed to write dpo_config.json: {e}")

    # collator = None

    class GradNormLogger(TrainerCallback):
        def __init__(self, every=100):
            self.every = every

        def on_gradient_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.every != 0 or model is None:
                return
            tot = 0.0
            cnt = 0
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                # with ZeRO stage 2, p.grad is a shard; still fine for a sanity number
                g = p.grad
                try:
                    val = float(g.norm().detach().cpu())
                except Exception:
                    continue
                tot += val
                cnt += 1
            if cnt:
                logging.info(f"[gradnorm] mean={tot / cnt:.4f} over {cnt} params")

    # Regularization config (aligned with SFT pattern)
    reg_cfg = {
        "log_every": 100,  # Reduced logging frequency for speed
        "L_DECORR": 1e-4,  # decorrelate latents
        "L_MASS": 1e-3,  # enforce soft mass ~= k
        "L_ENTROPY": 0.0,  # encourage sharp gates (disabled by default)
        "L_ORTHO_A": 1e-4,  # orthogonality on A (rows ~ latents)
        "L_ORTHO_B": 1e-4,  # orthogonality on B (columns ~ latents)
        # Reduced from 4 to 20 for better speed (only for z_plus_ortho mode)
        "ORTHO_EVERY": 20,
        # CRITICAL: Decorrelation is expensive (r×r matrix), only do every 10 steps
        "DECORR_EVERY": 10,
        "MASS_EVERY": 2,  # Mass is cheap but still skip some steps
        "schedule_decorr": True,
        "schedule_mass": True,
        "schedule_ent": True,
        "schedule_ortho": True,
        "sched_start": 0.0,
        "sched_end": 0.15,
        "sched_type": "cubic",
    }

    # Prepare callbacks
    callbacks = [
        MemoryClearCallback(),
        TopKProgressCallback(),
    ]

    # Add dead latent logging if enabled in config
    if getattr(experiment_args.lora, "log_dead_latents", False):
        from src.models import DeadLatentsLoggerCallback

        dead_latents_log_every = getattr(
            experiment_args.lora, "dead_latents_log_every", 500
        )
        callbacks.append(DeadLatentsLoggerCallback(log_every=dead_latents_log_every))
        logging.info(
            f"📊 Added DeadLatentsLoggerCallback (log_every={dead_latents_log_every})"
        )

    # Verify dataset format before passing to DPOTrainer
    logging.info("\n=== Dataset format going into DPOTrainer ===")
    logging.info(f"Columns: {train_dataset.column_names}")
    logging.info(f"First example type - chosen: {type(train_dataset[0]['chosen'])}")
    logging.info(f"First example type - rejected: {type(train_dataset[0]['rejected'])}")
    if isinstance(train_dataset[0]["chosen"], list):
        logging.info("Format: Message lists ✅")
        logging.info(f"Sample chosen messages: {train_dataset[0]['chosen'][:2]}")
        logging.info(f"Sample rejected messages: {train_dataset[0]['rejected'][:2]}")
    else:
        logging.info("Format: Text strings ⚠️")
        logging.info(f"Sample chosen: {train_dataset[0]['chosen'][:200]}")

    # Create trainer with TopK-aware regularization
    trainer = EnhancedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
        processing_class=tokenizer,
        callbacks=callbacks,
        reg_cfg=reg_cfg,
        reg_mode="off",  # Enable TopK-aware regularization
    )

    print(trainer.train_dataset[0])

    # Train
    logging.info("Starting training...")
    trainer.train()

    # No unwrapping needed! TopKLoRALinearSTE.state_dict() handles transparency
    # The wrapper delegates to lora_module automatically during save_model()
    logging.info("Saving model (TopK wrappers are transparent to PEFT save)...")

    # Save final model
    final_path = os.path.join(output_dir, "final_adapter")

    # Primary save via trainer
    if trainer.is_world_process_zero():
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        # Also explicit PEFT save for compatibility
        trainer.model.save_pretrained(final_path)

        # Explicit adapter state dict (belt and suspenders approach from SFT)
        adapter_state_dict = get_peft_model_state_dict(
            model, state_dict=model.state_dict(), adapter_name="default"
        )
        torch.save(adapter_state_dict, f"{final_path}/adapter_model.bin")

        # Save as safetensors (preferred format)
        from safetensors.torch import load_file, save_file

        save_file(adapter_state_dict, f"{final_path}/adapter_model.safetensors")

        logging.info(f"✅ Adapter saved to: {final_path}")

        # print dataset used
        # Verification: print saved keys
        saved = load_file(f"{final_path}/adapter_model.safetensors")
        print("Saved keys sample:")
        for key in list(saved.keys())[:5]:
            print(f"  {key}")

        logging.info(f"Model saved to {final_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final model saved to: {final_path}")
    print("\nConfiguration summary:")
    print(
        f"  - LoRA: r={experiment_args.lora.r}, k={experiment_args.lora.k} (sparsity={(1 - experiment_args.lora.k / experiment_args.lora.r) * 100:.1f}%)"
    )
    print(f"  - Soft masking with temperature={experiment_args.lora.temperature}")
    print(f"  - DPO beta={dpo_args.beta}, lr={dpo_args.learning_rate}")
    print("=" * 60)
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Eval dataset size: {len(eval_dataset)}")
    # print the name of the dataset
    logging.info(f"Dataset used: {cfg.training.dpo_dataset.name}")

    if cfg.logger.wandb_mode != "disabled" and trainer.is_world_process_zero():
        wandb.finish()

from trl import DPOTrainer
from typing import Optional, Dict, Any
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
from typing import List, Optional, Dict, Any
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from src.models import TopKLoRALinearSTE, _soft_topk_mass
from trl import DPOTrainer, DPOConfig
from torch.optim import AdamW
import wandb

# NEW: utilities for structured run tracking
import json
import hashlib
from datetime import datetime
import platform
import subprocess

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

L_DECORR = 1e-4   # decorrelate latents (small)
L_MASS = 1e-3   # enforce soft mass ~= k
L_ENTROPY = 0.0    # encourage sharp gates (set >0 if needed)
L_ORTHO_A = 1e-4   # orthogonality strength on A (rows ~ latents)
L_ORTHO_B = 1e-4   # orthogonality strength on B (columns ~ latents)
ORTHO_EVERY = 4    # compute every step; set to 2/4 to reduce overhead


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
            if isinstance(module, TopKLoRALinearSTE):
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
            if isinstance(module, TopKLoRALinearSTE) and name in self.activation_trackers:
                if module._last_z is not None:
                    with torch.no_grad():
                        # Get activation magnitudes
                        z_abs = module._last_z
                        # Average over batch and sequence dimensions
                        avg_activations = z_abs.mean(dim=(0, 1)).cpu()

                        # Update tracker
                        tracker = self.activation_trackers[name]
                        tracker['total_activations'] += avg_activations
                        tracker['activation_counts'] += (
                            avg_activations > 0.01).long()
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
            avg_activations = tracker['total_activations'] / \
                tracker['samples_seen']
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

    def __init__(self, *args, reg_cfg: Optional[Dict[str, Any]] = None, reg_mode: str = "z_only", **kwargs):
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
            "ORTHO_EVERY": 1,
            "sched_type": "cubic",  # "linear" | "quad" | "cubic"
            "sched_start": 0.0,    # fraction of training (0..1)
            "sched_end": 0.30,     # fraction of training (0..1)
            "schedule_decorr": True,
            "schedule_mass":   True,
            "schedule_ent":    True,
            "schedule_ortho":  True,
            "log_every": 50,
        }
        if reg_cfg:
            self.reg_cfg.update(reg_cfg)

    # ---------- scheduling helpers ----------
    def _sched_scalar(self, p: float) -> float:
        s0 = float(self.reg_cfg["sched_start"])
        s1 = float(self.reg_cfg["sched_end"])
        if s1 <= s0:
            return 1.0   # always on
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
            return t ** 3
        return t ** 2  # quad default

    # DS-safe: only build term if it will actually contribute
    def _active(self, L: float, scheduled_flag: bool, w_sched: float) -> bool:
        if L <= 0.0:
            return False
        if not scheduled_flag:
            return True
        return w_sched > 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # base DPO loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
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
                        self.log({
                            f"{name}.k": st["k"],
                            f"{name}.tau": st["tau"],
                            f"{name}.frac_active_vs_target": st["frac_active_vs_target"],
                        })
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
                need_decorr = self._active(L_DECORR,  self.reg_cfg.get(
                    "schedule_decorr", True), w_sched)
                need_mass = self._active(L_MASS,    self.reg_cfg.get(
                    "schedule_mass",   True), w_sched)
                need_ent = self._active(L_ENTROPY, self.reg_cfg.get(
                    "schedule_ent",    True), w_sched)
                need_orthoA = (self.reg_mode == "z_plus_ortho") and ORTHO_EVERY > 0 and (step % ORTHO_EVERY == 0) \
                    and self._active(L_ORTHO_A, self.reg_cfg.get("schedule_ortho", True), w_sched)
                need_orthoB = (self.reg_mode == "z_plus_ortho") and ORTHO_EVERY > 0 and (step % ORTHO_EVERY == 0) \
                    and self._active(L_ORTHO_B, self.reg_cfg.get("schedule_ortho", True), w_sched)

                # If every term is scheduled & weight == 0, hard-skip this module (DS-safe)
                all_sched = (
                    self.reg_cfg.get("schedule_decorr", True) and
                    self.reg_cfg.get("schedule_mass",   True) and
                    self.reg_cfg.get("schedule_ent",    True) and
                    (self.reg_mode != "z_plus_ortho" or self.reg_cfg.get(
                        "schedule_ortho", True))
                )
                if all_sched and (not (need_decorr or need_mass or need_ent or need_orthoA or need_orthoB)):
                    if do_log:
                        acc["reg/sched_w"] += w_sched
                        n_layers += 1
                    continue

                # Prepare gates once (used by mass/entropy and usage balancing)
                k_now = m._current_k()
                tau = m._tau()
                g_soft_live = getattr(m, "_g_soft_live", None)
                g_soft = g_soft_live if g_soft_live is not None else _soft_topk_mass(
                    z_live, k_now, tau)

                # -------- z-based regs (always allowed in "z_only" and "z_plus_ortho") --------
                if need_decorr:
                    Z = z_live.reshape(-1, z_live.size(-1)).float()
                    Z = Z - Z.mean(dim=0, keepdim=True)
                    C = (Z.T @ Z) / (Z.size(0) + 1e-6)
                    off = C - torch.diag(torch.diag(C))
                    r_decorr = (off ** 2).mean().to(loss.dtype)
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
                        r_ent = -(g_soft.clamp_min(1e-8) *
                                  g_soft.clamp_min(1e-8).log()).sum(dim=-1).mean()
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
                    r_oa = (GA_off ** 2).mean().to(loss.dtype)
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
                    r_ob = (GB_off ** 2).mean().to(loss.dtype)
                    if self.reg_cfg.get("schedule_ortho", True):
                        r_ob = r_ob * r_ob.new_tensor(w_sched)
                    reg = reg + L_ORTHO_B * r_ob
                    if do_log:
                        acc["reg/ortho_B"] += float(r_ob.detach().cpu())

                if do_log:
                    acc["reg/sched_w"] += w_sched
                    n_layers += 1

                L1 = 1e-5
                reg = reg + L1 * z_live.abs().mean()
                usage = g_soft.mean(dim=(0, 1))                 # [r]
                cov = ((usage - usage.mean())**2).mean()
                reg = reg + 1e-4 * cov

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


class MemoryClearCallback(TrainerCallback):
    """Memory management callback"""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.gradient_accumulation_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()


def prepare_hh_rlhf_datasets(
    max_length=1024,
    train_size=None,
    eval_size=100,
    tokenizer=None,
    max_prompt_length=512,
    max_completion_length=512
):
    """Load and prepare Anthropic/hh-rlhf for reference-free DPO."""
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    eos = tokenizer.eos_token

    ASSISTANT = "Assistant:"

    def split_reply(text: str):
        i = text.rfind(ASSISTANT)
        if i == -1:
            raise ValueError("No 'Assistant:' in example.")
        prompt = text[: i + len(ASSISTANT)]
        reply = text[i + len(ASSISTANT):].strip()
        return prompt, reply

    def format_hh(samples):
        prompts, chosens, rejecteds = [], [], []
        for c, r in zip(samples["chosen"], samples["rejected"]):
            p_c, ch = split_reply(c)
            p_r, rj = split_reply(r)
            # normalize whitespace before comparing
            if p_c.strip() != p_r.strip() or not ch or not rj:
                continue
            # ensure EOS termination
            if eos and not ch.endswith(eos):
                ch = ch + " " + eos
            if eos and not rj.endswith(eos):
                rj = rj + " " + eos
            p_c = re.sub(r"^\s*\n*", "", p_c)
            prompts.append(p_c)
            chosens.append(ch)
            rejecteds.append(rj)
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

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
        p = ex["prompt"]
        return (
            p.rstrip().endswith("Assistant:") and
            ("Assistant:" not in ex["chosen"]) and
            ("Assistant:" not in ex["rejected"]) and
            len(ex["chosen"]) > 0 and len(ex["rejected"]) > 0
        )
    train_dataset = train_dataset.filter(ok)
    eval_dataset = eval_dataset.filter(ok)

    max_p = max_prompt_length
    max_c = max_completion_length

    def length_ok(ex):
        p_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(ex["chosen"], add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(ex["rejected"], add_special_tokens=False)[
            "input_ids"]
        return len(p_ids) <= max_p and len(c_ids) <= max_c and len(r_ids) <= max_c

    train_dataset = train_dataset.filter(length_ok)
    eval_dataset = eval_dataset.filter(length_ok)

    # (Optional) filter out very short replies
    train_dataset = train_dataset.filter(
        lambda ex: len(ex["chosen"]) > 10 and len(
            ex["rejected"]) > 10
    )
    eval_dataset = eval_dataset.filter(
        lambda ex: len(ex["chosen"]) > 0 and len(
            ex["rejected"]) > 0
    )

    # Decrease dataset sizes
    if train_size:
        train_dataset = train_dataset.select(
            range(min(train_size, len(train_dataset))))
    if eval_size:
        eval_dataset = eval_dataset.select(
            range(min(eval_size, len(eval_dataset))))

    logging.info(
        f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
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
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out.stdout.decode().strip()
        except Exception:
            return None
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _run(["git", "status", "--porcelain"]),
        "is_dirty": bool(_run(["git", "status", "--porcelain"])) if _run(["git", "rev-parse", "--git-dir"]) else None,
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
        "temperature_final": float(getattr(lora, "temperature_final", 0.1 * getattr(lora, "temperature", 1.0))),
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
                    "load_in_4bit", "load_in_8bit", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant",
                    "bnb_4bit_quant_type", "llm_int8_threshold", "llm_int8_enable_fp32_cpu_offload",
                ] if hasattr(quant_cfg, k)
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
        "vocab_size": len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else None,
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "special_tokens_map": tokenizer.special_tokens_map if hasattr(tokenizer, "special_tokens_map") else None,
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
        "logger": getattr(cfg, "logger", None).__dict__ if hasattr(getattr(cfg, "logger", None), "__dict__") else str(getattr(cfg, "logger", None)),
        "env": env,
        "git": _get_git_info(),
        "seeds": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "torch_seed": None,  # set by caller if needed
        },
    }
    return hparams


def _make_run_dir(base_dir: str, model_name: str, tag: str, hparams: Dict[str, Any]) -> str:
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
            cfg_dict = OmegaConf.to_container(
                hparams.get("cfg", {}), resolve=True)
        except Exception:
            cfg_dict = None
        if cfg_dict:
            with open(os.path.join(run_dir, "cfg.yaml"), "w") as f:
                import yaml
                yaml.safe_dump(cfg_dict, f, sort_keys=False)
    except Exception:
        pass

    return run_dir


def run_dpo(cfg, quant_cfg):
    dpo_args = cfg.training.dpo
    experiment_args = cfg.training.dpo_experiment

    # Load tokenizer
    logging.info(
        f"Loading tokenizer from {cfg.training.base_sft_merged_model.checkpoint_dir}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.training.base_sft_merged_model.checkpoint_dir
    )

    # Load policy model
    logging.info("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.training.base_sft_merged_model.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_cfg,
        attn_implementation='eager'
    )
    policy_model = prepare_model_for_kbit_training(policy_model)
    extra = False

    if not getattr(tokenizer, "chat_template", None):
        logging.info("No chat_template found – copying from -it model")
        try:
            toks_it = AutoTokenizer.from_pretrained(
                cfg.training.base_sft_merged_model.model_it_name,
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
                policy_model.generation_config.eos_token_id.append(
                    eot_token_id)
        else:
            prev_eos = policy_model.generation_config.eos_token_id
            policy_model.generation_config.eos_token_id = [
                prev_eos, eot_token_id]
    else:
        policy_model.generation_config.eos_token_id = [
            tokenizer.eos_token_id, eot_token_id]

    # Log the configuration
    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS token ID(s): {policy_model.generation_config.eos_token_id}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load reference model
    logging.info("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.training.base_sft_merged_model.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_cfg,
        attn_implementation='eager'
    )
    ref_model.generation_config.eos_token_id = policy_model.generation_config.eos_token_id

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if extra and new_tokens:
        ref_model.resize_token_embeddings(len(tokenizer))
        logging.info(
            "Added %d extra special tokens to ref model",
            len(new_tokens)
        )

    if cfg.training.dpo_dataset.name == "hh-rlhf":
        train_dataset, eval_dataset = prepare_hh_rlhf_datasets(
            max_length=dpo_args.max_prompt_length,
            tokenizer=tokenizer, max_prompt_length=dpo_args.max_prompt_length,
            max_completion_length=dpo_args.max_completion_length,
        )
    else:
        raise NotImplementedError(
            f"Dataset {cfg.training.dpo_dataset.name} not implemented"
        )

    # Configure LoRA
    target_modules = list(experiment_args.lora.target_modules)
    logging.info(
        f"Target modules: {len(target_modules)} modules"
    )

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
    logging.info(
        f"Initialized {initialised} LoRA layers with normal distribution")

    # Inject TopK wrappers
    logging.info("Injecting TopKLoRALinearSTE wrappers...")
    replaced = 0
    for name, module in model.named_modules():
        if getattr(module, "lora_A", None) is None and False:
            continue
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        attr = name.split(".")[-1]
        setattr(
            parent, attr,
            TopKLoRALinearSTE(
                base=module,
                layer_name=name,
                k=experiment_args.lora.k,
                temperature=experiment_args.lora.temperature,
                temperature_schedule=experiment_args.lora.temperature_schedule,
                k_schedule=experiment_args.lora.k_schedule,
                k_final=experiment_args.lora.k_final,
                hard_eval=True,
                relu_latents=True,
                alpha_over_r=True,
                temperature_final=getattr(
                    experiment_args.lora, 'temperature_final', None),
            )
        )
        replaced += 1

    logging.info(f"Injected TopK STE wrappers in {replaced} layers")
    model.print_trainable_parameters()

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

    # Mark this as the latest run via symlink (best-effort)
    try:
        latest_link = os.path.join(cfg.training.dump_path, "latest")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            try:
                os.remove(latest_link)
            except Exception:
                pass
        os.symlink(output_dir, latest_link)
    except Exception as e:
        logging.warning(f"Could not create 'latest' symlink: {e}")

    # Save the full cfg as YAML for exact reproducibility (if OmegaConf available)
    try:
        from omegaconf import OmegaConf
        with open(os.path.join(output_dir, "cfg.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
    except Exception as e:
        logging.warning(f"Could not serialize cfg to YAML: {e}")

    # Capture environment snapshots
    try:
        env_dir = os.path.join(output_dir, "env")
        os.makedirs(env_dir, exist_ok=True)
        # pip freeze
        try:
            frz = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(os.path.join(env_dir, "requirements_freeze.txt"), "wb") as f:
                f.write(frz.stdout)
        except Exception:
            pass
        # nvidia-smi
        try:
            smi = subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(os.path.join(env_dir, "nvidia-smi.txt"), "wb") as f:
                f.write(smi.stdout or smi.stderr)
        except Exception:
            pass
    except Exception as e:
        logging.warning(f"Failed to capture environment info: {e}")

    # Human-readable summary
    try:
        summary = []
        summary.append(f"model: {cfg.training.model.model_name}")
        summary.append(
            f"base: {cfg.training.base_sft_merged_model.checkpoint_dir}")
        summary.append(
            f"dataset: {cfg.training.dpo_dataset.name} (train={len(train_dataset)}, eval={len(eval_dataset)})")
        summary.append(
            f"lora_topk: r={experiment_args.lora.r}, k={experiment_args.lora.k}, k_final={getattr(experiment_args.lora, 'k_final', experiment_args.lora.k)}, "
            f"alpha={getattr(experiment_args.lora, 'alpha', getattr(experiment_args.lora, 'lora_alpha', 16))}, temp={getattr(experiment_args.lora, 'temperature', 1.0)}"
        )
        summary.append(
            f"dpo: beta={dpo_args.beta}, lr={dpo_args.learning_rate}, steps={dpo_args.max_steps}, bs={dpo_args.per_device_train_batch_size}x{dpo_args.gradient_accumulation_steps}"
        )
        with open(os.path.join(output_dir, "README.txt"), "w") as f:
            f.write("\n".join(summary) + "\n")
    except Exception as e:
        logging.warning(f"Failed to write summary README.txt: {e}")

    # Optionally sync hparams into Weights & Biases config
    if getattr(cfg.logger, "report_to", None) and "wandb" in cfg.logger.report_to:
        try:
            if wandb.run is not None:
                wandb.config.update(hparams, allow_val_change=True)
                # Prefer the folder name as run_name if not provided
                if not getattr(cfg, "experiment_name", None):
                    wandb.run.name = os.path.basename(output_dir)
        except Exception as e:
            logging.warning(f"Could not update wandb config: {e}")

    # DPO configuration
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
        eval_steps=dpo_args.eval_steps,
        save_steps=dpo_args.save_steps,
        bf16=True,
        report_to=cfg.logger.report_to,
        run_name=cfg.experiment_name,
        remove_unused_columns=False,
        max_grad_norm=0.5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/accuracies",
        greater_is_better=True,
        save_total_limit=3,
    )

    # Persist training args for reproducibility
    try:
        with open(os.path.join(output_dir, "dpo_config.json"), "w") as f:
            json.dump(dpo_config.to_dict() if hasattr(dpo_config, "to_dict")
                      else dpo_config.__dict__, f, indent=2, default=str)
    except Exception as e:
        logging.warning(f"Failed to write dpo_config.json: {e}")

    collator = None

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
                logging.info(
                    f"[gradnorm] mean={tot/cnt:.4f} over {cnt} params")

    # Create trainer
    # trainer = EnhancedDPOTrainer(
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
        processing_class=tokenizer,
        callbacks=[
            MemoryClearCallback(),
            TopKProgressCallback(),
            # DeadLatentsLoggerCallback(log_every=5),
            # GradNormLogger(every=1),

            # DeadNeuronDetectionCallback(
            #     check_interval=args.dead_neuron_interval,
            #     num_samples=args.dead_neuron_samples,
            #     activation_threshold=0.001, # TODO: consider lowering to 0
            #     use_soft_detection=False
            # )
            # EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )

    # trainer.reg_mode = "z_only"
    # trainer.reg_cfg.update(dict(
    #     L_MASS=5e-3, L_DECORR=1e-4, L_ENTROPY=0.0,
    #     sched_start=0.0, sched_end=0.15,          # fast early ramp
    #     schedule_decorr=True, schedule_mass=True, schedule_ent=True,
    #     sched_type="cubic",
    #     log_every=50,
    # ))
    # trainer.reg_cfg.update(log_every=5)

    # trainer.reg_mode = "z_plus_ortho"
    # trainer.reg_cfg.update({
    #     "ORTHO_EVERY": 8,         # start sparse
    #     "sched_start": 0.001,      # don’t start at step 0
    #     "sched_end": 0.30,
    #     "schedule_ortho": True,
    # })

    # def attach_lora_grad_hooks(model):
    #     handles = []
    #     for name, p in model.named_parameters():
    #         if p.requires_grad and (name.endswith(".A") or name.endswith(".B") or
    #                                 "lora_A" in name or "lora_B" in name):
    #             def make_hook(n):
    #                 def _hook(grad):
    #                     print(f"{n} grad_norm={float(grad.norm())}")
    #                 return _hook
    #             handles.append(p.register_hook(make_hook(name)))
    #     return handles

    # # After building `model` and before `trainer.train()`:
    # handles = attach_lora_grad_hooks(model)

    # Train
    logging.info("Starting training...")
    trainer.train()

    # Unwrap TopK wrappers before saving
    logging.info("Unwrapping TopK wrappers...")
    unwrapped = 0
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinearSTE) and False:
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            attr = name.split(".")[-1]
            setattr(parent, attr, module.lora_module)
            unwrapped += 1

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
    # if hasattr(trainer.state, 'best_metric'):
    #    print(f"Best eval accuracy: {trainer.state.best_metric:.4f}")
    print("\nConfiguration summary:")
    print(
        f"  - LoRA: r={experiment_args.lora.r}, k={experiment_args.lora.k} (sparsity={(1-experiment_args.lora.k/experiment_args.lora.r)*100:.1f}%)")
    print(
        f"  - Soft masking with temperature={experiment_args.lora.temperature}")
    print(f"  - DPO beta={dpo_args.beta}, lr={dpo_args.learning_rate}")
    print("="*60)

    if cfg.logger.wandb_mode != "disabled":
        wandb.finish()

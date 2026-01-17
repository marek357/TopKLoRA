"""HH-RLHF-specific causal autointerp pipeline for TopKLoRA latents.

This module runs an end-to-end evaluation loop over HH-RLHF prompts:
prompt selection, latent stats collection, intervention calibration,
evidence pack generation, hypothesis creation, and verification scoring.
"""

from __future__ import annotations

import json
import logging
import os
import random
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from src.models import TopKLoRALinearSTE, _hard_topk_mask
from src.steering import FeatureSteeringContext, list_available_adapters
from src.utils import generate_completions_from_prompts, hh_string_to_messages


logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    """Select CUDA if available, else MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_dir(path: str) -> None:
    """Create the directory and parents if they do not already exist."""
    os.makedirs(path, exist_ok=True)


def _stable_hash(text: str) -> str:
    """Return a short stable hash for deriving IDs from prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    """Write a sequence of records to a JSONL file."""
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _render_prompt(tokenizer, prompt: str) -> str:
    """Render a user prompt using the chat template when available."""
    if getattr(tokenizer, "apply_chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            return prompt
    return prompt


def _extract_prompt_from_hh(text: str) -> str:
    """Extract the first user message from an HH-RLHF conversation string."""
    msgs = hh_string_to_messages(text)
    for msg in msgs:
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def _build_prompt_records(cfg) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load HH-RLHF prompts, assign buckets, and split into analysis/eval sets.

    Args:
        cfg: Hydra config with eval settings at `cfg.evals.causal_autointerp_framework`.

    Returns:
        A tuple of (analysis_prompts, eval_prompts), each a list of dicts with
        keys: prompt_id, prompt, bucket, meta.

    Raises:
        FileNotFoundError: If the dataset or split cannot be located.
        ValueError: If the dataset is malformed or split keys are invalid.
    """
    dataset_cfg = cfg.evals.causal_autointerp_framework.dataset
    prompts: List[Dict[str, Any]] = []

    for bucket in dataset_cfg.buckets:
        ds = load_dataset(
            dataset_cfg.name,
            data_dir=bucket.data_dir,
            split=bucket.split,
        )
        max_prompts = int(getattr(bucket, "max_prompts", -1))
        if max_prompts > 0:
            ds = ds.select(range(min(len(ds), max_prompts)))
        for ex in ds:
            prompt_text = ""
            if "prompt" in ex:
                prompt_text = ex["prompt"].strip()
            elif "chosen" in ex:
                prompt_text = _extract_prompt_from_hh(ex["chosen"])
            if not prompt_text:
                continue
            prompt_id = _stable_hash(prompt_text)
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "bucket": bucket.bucket,
                    "meta": {"split": bucket.split, "data_dir": bucket.data_dir},
                }
            )

    rng = random.Random(int(getattr(dataset_cfg.split, "seed", cfg.seed)))
    rng.shuffle(prompts)
    # analysis_frac determines the fraction of prompts used for analysis stages
    # (latent stats, calibration, evidence train). The remainder is held out
    # for evaluation stages (evidence eval + verification).
    analysis_frac = float(getattr(dataset_cfg.split, "analysis_frac", 0.8))
    split_idx = int(len(prompts) * analysis_frac)
    analysis_prompts = prompts[:split_idx]
    eval_prompts = prompts[split_idx:]
    return analysis_prompts, eval_prompts


def _list_topk_modules(model: torch.nn.Module) -> Dict[str, TopKLoRALinearSTE]:
    """Collect all TopKLoRALinearSTE modules keyed by full module name."""
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinearSTE):
            modules[name] = module
    return modules


def _build_latent_index(modules: Dict[str, TopKLoRALinearSTE]) -> List[Dict[str, Any]]:
    """Assign a global latent_id to each adapter feature index."""
    latent_index = []
    latent_id = 0
    for name, module in modules.items():
        for idx in range(module.r):
            latent_index.append(
                {
                    "latent_id": latent_id,
                    "adapter_name": name,
                    "feature_idx": idx,
                }
            )
            latent_id += 1
    return latent_index


def _extract_token_window(
    tokenizer, input_ids: torch.Tensor, pos: int, window: int
) -> str:
    """Return a token window around a position, marking the center token."""
    ids = input_ids.tolist()
    start = max(0, pos - window)
    end = min(len(ids), pos + window + 1)
    toks = [
        tokenizer.decode([tid], skip_special_tokens=False) for tid in ids[start:end]
    ]
    center = pos - start
    if 0 <= center < len(toks):
        toks[center] = f"<<{toks[center]}>>"
    return "".join(toks).replace("\n", "⏎")


def _collect_latent_stats(
    cfg,
    model,
    tokenizer,
    prompts: List[Dict[str, Any]],
    modules: Dict[str, TopKLoRALinearSTE],
    latent_index: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute per-latent activation statistics across analysis prompts.

    Args:
        cfg: Hydra config with `latents` settings (epsilon, quantiles, etc.).
        model: Loaded model with TopKLoRALinearSTE modules.
        tokenizer: Tokenizer used for prompt encoding.
        prompts: List of analysis prompt records.
        modules: Mapping of module name to TopKLoRALinearSTE.
        latent_index: Global latent index entries (latent_id, adapter_name, feature_idx).

    Returns:
        List of per-latent stats dictionaries with keys:
        latent_id, mu, sigma, p_active, quantiles, notes, adapter_name, feature_idx.

    Notes:
        For TopK layers, stats are computed on the *gated* activations using the
        hard TopK mask, and p_active counts prompts where the latent was gated on
        at least once (per-prompt activity, not per-token).

    Raises:
        RuntimeError: If forward pass fails (e.g., device mismatch).
    """
    device = next(model.parameters()).device
    lat_cfg = cfg.evals.causal_autointerp_framework.latents
    epsilon = float(getattr(lat_cfg, "activation_epsilon", 1e-6))
    quantile_samples = int(getattr(lat_cfg, "quantile_samples", 2000))
    aggregate = getattr(lat_cfg, "aggregate", "max")

    sums = np.zeros(len(latent_index), dtype=np.float64)
    sums_sq = np.zeros(len(latent_index), dtype=np.float64)
    counts = np.zeros(len(latent_index), dtype=np.int64)
    active_counts = np.zeros(len(latent_index), dtype=np.int64)
    samples: List[List[float]] = [[] for _ in range(len(latent_index))]

    adapter_offsets = {}
    for entry in latent_index:
        if entry["feature_idx"] == 0:
            adapter_offsets[entry["adapter_name"]] = entry["latent_id"]

    def push_sample(latent_id: int, value: float) -> None:
        lst = samples[latent_id]
        if len(lst) < quantile_samples:
            lst.append(value)
        else:
            idx = random.randint(0, len(lst) - 1)
            lst[idx] = value

    model.eval()
    for rec in tqdm(prompts, desc="Collecting latent stats"):
        prompt_text = rec["prompt"]
        rendered = _render_prompt(tokenizer, prompt_text)
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=int(getattr(lat_cfg, "max_length", 2048)),
        ).to(device)
        with torch.no_grad():
            _ = model(**enc)

        for name, module in modules.items():
            if module._last_z is None:
                continue
            z = module._last_z.detach()
            if z.ndim == 3:
                z = z[0]
            if z.ndim != 2:
                continue

            k_now = int(module._current_k())
            mask = _hard_topk_mask(z, k_now)
            z_eff = z * mask
            active_mask = mask.any(dim=0)

            if aggregate == "mean":
                agg = z_eff.mean(dim=0)
            else:
                agg = z_eff.abs().max(dim=0).values

            adapter_offset = adapter_offsets.get(name)
            if adapter_offset is None:
                continue
            for feature_idx in range(module.r):
                latent_id = adapter_offset + feature_idx
                val = float(agg[feature_idx].item())
                sums[latent_id] += val
                sums_sq[latent_id] += val * val
                counts[latent_id] += 1
                if bool(active_mask[feature_idx].item()):
                    active_counts[latent_id] += 1
                push_sample(latent_id, val)

    latent_stats: List[Dict[str, Any]] = []
    for entry in latent_index:
        latent_id = entry["latent_id"]
        mean = sums[latent_id] / max(counts[latent_id], 1)
        var = sums_sq[latent_id] / max(counts[latent_id], 1) - mean * mean
        sigma = float(max(var, 1e-12) ** 0.5)
        p_active = active_counts[latent_id] / max(counts[latent_id], 1)
        qs = {}
        if samples[latent_id]:
            values = np.array(samples[latent_id])
            for q in getattr(lat_cfg, "quantiles", [0.5, 0.9, 0.99]):
                qs[str(q)] = float(np.quantile(values, q))
        latent_stats.append(
            {
                "latent_id": latent_id,
                "mu": float(mean),
                "sigma": float(sigma),
                "p_active": float(p_active),
                "quantiles": qs,
                "notes": {"dead": int(p_active <= 0.0)},
                "adapter_name": entry["adapter_name"],
                "feature_idx": entry["feature_idx"],
            }
        )
    return latent_stats


def _build_feature_dict(
    latent_entry: Dict[str, Any], effect: str
) -> Dict[str, List[Tuple[int, str]]]:
    """Build a steering dict for a single latent and effect."""
    return {latent_entry["adapter_name"]: [(latent_entry["feature_idx"], effect)]}


def _compute_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Compute mean KL divergence KL(logits_a || logits_b)."""
    logp_a = F.log_softmax(logits_a, dim=-1)
    logp_b = F.log_softmax(logits_b, dim=-1)
    p_a = logp_a.exp()
    kl = (p_a * (logp_a - logp_b)).sum(dim=-1)
    return float(kl.mean().item())


def _kl_for_prompt(
    model,
    tokenizer,
    prompt: str,
    intervention: Optional[Dict[str, Any]],
    window_tokens: int,
    amplification: float = 1.0,
) -> float:
    """Compute KL between baseline and intervened logits for a prompt."""
    device = next(model.parameters()).device
    rendered = _render_prompt(tokenizer, prompt)
    enc = tokenizer(rendered, return_tensors="pt").to(device)

    with torch.no_grad():
        base_logits = model(**enc).logits

    if intervention is None:
        return 0.0

    feature_dict = intervention["feature_dict"]
    with FeatureSteeringContext(
        model, feature_dict, verbose=False, amplification=amplification
    ):
        with torch.no_grad():
            steered_logits = model(**enc).logits

    if window_tokens > 0:
        base_logits = base_logits[:, -window_tokens:]
        steered_logits = steered_logits[:, -window_tokens:]

    return _compute_kl(base_logits, steered_logits)


def _calibrate_interventions(
    cfg,
    model,
    tokenizer,
    latent_index: Dict[int, Dict[str, Any]],
    latent_entries: List[Dict[str, Any]],
    prompts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Calibrate interventions to match target KL levels.

    Args:
        cfg: Hydra config with `intervention` settings.
        model: Loaded model for forward passes.
        tokenizer: Tokenizer for prompt encoding.
        latent_index: Mapping from latent_id to latent metadata.
        latent_entries: Ordered list of latent entries to calibrate.
        prompts: Calibration prompt records.

    Returns:
        List of calibration records with target_kl and observed_kl for
        `zero_ablate` and `steer_with_alpha` interventions.

    Notes:
        zero_ablate:
            Force the latent activation to zero (ablation) during generation.
        steer_with_alpha:
            Force the latent to be strongly active by amplifying its activation
            by a calibrated scalar alpha (a steering strength chosen to hit a KL target).

    Raises:
        RuntimeError: If KL computation fails or model forward fails.
    """
    calib_cfg = cfg.evals.causal_autointerp_framework.intervention
    kl_targets = list(getattr(calib_cfg, "kl_targets", [0.01]))
    window_tokens = int(getattr(calib_cfg, "window_tokens", 32))
    amp_min = float(getattr(calib_cfg, "amp_min", 0.5))
    amp_max = float(getattr(calib_cfg, "amp_max", 8.0))
    amp_steps = int(getattr(calib_cfg, "amp_search_steps", 8))
    n_prompts = int(getattr(calib_cfg, "calibration_prompts", 16))

    calib_prompts = prompts[: min(n_prompts, len(prompts))]
    records: List[Dict[str, Any]] = []

    for entry in latent_entries:
        latent_id = entry["latent_id"]
        for target_kl in kl_targets:
            effect = "disable"
            feature_dict = _build_feature_dict(entry, effect)
            kl_vals = [
                _kl_for_prompt(
                    model,
                    tokenizer,
                    p["prompt"],
                    {"feature_dict": feature_dict, "effect": effect},
                    window_tokens,
                )
                for p in calib_prompts
            ]
            observed = float(sum(kl_vals) / len(kl_vals)) if kl_vals else 0.0
            records.append(
                {
                    "latent_id": latent_id,
                    "intervention_type": "zero_ablate",
                    "target_kl": float(target_kl),
                    "alpha": None,
                    "observed_kl": observed,
                }
            )

            effect = "enable"
            low, high = amp_min, amp_max
            best_amp = high
            best_kl = 0.0
            for _ in range(amp_steps):
                mid = (low + high) / 2.0
                feature_dict = _build_feature_dict(entry, effect)
                kl_vals = [
                    _kl_for_prompt(
                        model,
                        tokenizer,
                        p["prompt"],
                        {"feature_dict": feature_dict, "effect": effect},
                        window_tokens,
                        amplification=mid,
                    )
                    for p in calib_prompts
                ]
                observed = float(sum(kl_vals) / len(kl_vals)) if kl_vals else 0.0
                best_amp, best_kl = mid, observed
                if observed < target_kl:
                    low = mid
                else:
                    high = mid
            records.append(
                {
                    "latent_id": latent_id,
                    "intervention_type": "steer_with_alpha",
                    "target_kl": float(target_kl),
                    "alpha": float(best_amp),
                    "observed_kl": float(best_kl),
                }
            )

    return records


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    gen_cfg,
    intervention: Optional[Dict[str, Any]] = None,
    amplification: float = 1.0,
) -> str:
    """Generate a completion for a prompt, with optional latent intervention."""
    device = next(model.parameters()).device
    rendered = _render_prompt(tokenizer, prompt)
    gen_kwargs = dict(
        max_new_tokens=int(getattr(gen_cfg, "max_new_tokens", 128)),
        do_sample=bool(getattr(gen_cfg, "do_sample", True)),
        temperature=float(getattr(gen_cfg, "temperature", 0.7)),
        top_p=float(getattr(gen_cfg, "top_p", 0.95)),
        eos_token_id=getattr(model.generation_config, "eos_token_id", None),
    )

    if intervention is None:
        completions = generate_completions_from_prompts(
            model,
            tokenizer,
            [rendered],
            device=str(device),
            gen_kwargs=gen_kwargs,
        )
        return completions[0] if completions else ""
    else:
        enc = tokenizer(rendered, return_tensors="pt").to(device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        with FeatureSteeringContext(
            model,
            intervention["feature_dict"],
            verbose=False,
            amplification=amplification,
        ):
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)

    completion_ids = out[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _build_evidence_packs(
    cfg,
    model,
    tokenizer,
    prompts: List[Dict[str, Any]],
    latent_index: Dict[int, Dict[str, Any]],
    latent_entries: List[Dict[str, Any]],
    calibration: List[Dict[str, Any]],
    per_latent: int,
) -> List[Dict[str, Any]]:
    """Create evidence packs containing baseline and steered samples.

    Args:
        cfg: Hydra config with `evidence` and `generation` settings.
        model: Loaded model for text generation.
        tokenizer: Tokenizer for prompt rendering/decoding.
        prompts: Prompt records to sample from.
        latent_index: Mapping from latent_id to latent metadata.
        latent_entries: Ordered list of latent entries to process.
        calibration: Calibration records used to select amplitudes.
        per_latent: Number of prompts to sample per latent.

    Returns:
        List of evidence pack dictionaries with baseline and steered samples.

    Notes:
        Evidence packs use the calibrated `steer_with_alpha` intervention by default.
        The `zero_ablate` variant is used only during calibration.

    Raises:
        RuntimeError: If generation fails or model forward fails.
    """
    ev_cfg = cfg.evals.causal_autointerp_framework.evidence
    gen_cfg = cfg.evals.causal_autointerp_framework.generation
    target_kl = float(getattr(ev_cfg, "target_kl", 0.01))
    baseline_samples = int(getattr(gen_cfg, "baseline_samples", 3))
    calib_map = {}
    for rec in calibration:
        if (
            rec["target_kl"] == target_kl
            and rec["intervention_type"] == "steer_with_alpha"
        ):
            calib_map[rec["latent_id"]] = rec

    rng = random.Random(cfg.seed)
    evidence_records: List[Dict[str, Any]] = []

    for entry in latent_entries:
        latent_id = entry["latent_id"]
        if len(prompts) >= per_latent:
            selected = rng.sample(prompts, k=per_latent)
        else:
            selected = prompts

        calib = calib_map.get(latent_id)
        if not calib:
            continue

        feature_dict = _build_feature_dict(entry, "enable")
        amplification = float(calib["alpha"])

        for prompt_rec in selected:
            prompt = prompt_rec["prompt"]
            baseline = [
                _generate_text(model, tokenizer, prompt, gen_cfg)
                for _ in range(baseline_samples)
            ]
            steered = _generate_text(
                model,
                tokenizer,
                prompt,
                gen_cfg,
                intervention={"feature_dict": feature_dict},
                amplification=amplification,
            )

            evidence_records.append(
                {
                    "latent_id": latent_id,
                    "prompt_id": prompt_rec["prompt_id"],
                    "prompt": prompt,
                    "baseline_samples": baseline,
                    "steered_sample": steered,
                    "control_samples": {},
                    "intervention_meta": {
                        "type": "steer_with_alpha",
                        "alpha": amplification,
                        "target_kl": target_kl,
                        "window": {
                            "t_start": 0,
                            "t_end": int(getattr(ev_cfg, "window_tokens", 64)),
                        },
                    },
                }
            )

    return evidence_records


def _summarize_verification(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize verifier accuracy and hypothesis lift per latent."""
    by_latent: Dict[int, Dict[str, List[int]]] = {}
    for rec in results:
        latent_id = rec["latent_id"]
        by_latent.setdefault(latent_id, {"blind": [], "hyp": []})
        by_latent[latent_id]["blind"].append(int(rec["blind_correct"]))
        by_latent[latent_id]["hyp"].append(int(rec["hyp_correct"]))

    summary = {"latents": {}, "overall": {}}
    all_blind = []
    all_hyp = []
    for latent_id, vals in by_latent.items():
        blind_acc = (
            float(sum(vals["blind"]) / len(vals["blind"])) if vals["blind"] else 0.0
        )
        hyp_acc = float(sum(vals["hyp"]) / len(vals["hyp"])) if vals["hyp"] else 0.0
        summary["latents"][str(latent_id)] = {
            "A_blind": blind_acc,
            "A_hyp": hyp_acc,
            "lift": hyp_acc - blind_acc,
        }
        all_blind.extend(vals["blind"])
        all_hyp.extend(vals["hyp"])

    summary["overall"] = {
        "A_blind": float(sum(all_blind) / len(all_blind)) if all_blind else 0.0,
        "A_hyp": float(sum(all_hyp) / len(all_hyp)) if all_hyp else 0.0,
        "lift": (
            float(sum(all_hyp) / len(all_hyp)) - float(sum(all_blind) / len(all_blind))
        )
        if all_hyp
        else 0.0,
    }
    return summary


def _build_explainer_prompt(
    evidence_packs: List[Dict[str, Any]],
    effects_menu: List[str],
    include_antipredictions: bool,
) -> str:
    raise NotImplementedError


# TODO: Needs to redo this, get some inspiration from literature
# def _build_explainer_prompt(
#     evidence_packs: List[Dict[str, Any]],
#     effects_menu: List[str],
#     include_antipredictions: bool,
# ) -> str:
#     """Build the explainer prompt that requests structured hypotheses."""
#     lines = [
#         "You are an interpretability researcher.",
#         "You will be given evidence packs from interventions on a single latent.",
#         "Return a JSON object with keys: latent_id, hypothesis_text, predicted_effects, scope, anti_predictions.",
#         "predicted_effects must be a subset of the allowed menu.",
#         "scope must include buckets (list) and expected_strength (small|medium|large).",
#         "anti_predictions should list at least one thing that should NOT change.",
#         "Allowed predicted_effects menu:",
#         ", ".join(effects_menu),
#         "",
#         "Evidence packs:",
#     ]
#     for pack in evidence_packs:
#         lines.append(f"latent_id: {pack['latent_id']}")
#         lines.append(f"prompt: {pack['prompt']}")
#         for i, sample in enumerate(pack["baseline_samples"], 1):
#             lines.append(f"baseline_{i}: {sample}")
#         lines.append(f"steered: {pack['steered_sample']}")
#         lines.append("---")
#     if not include_antipredictions:
#         lines.append("If unsure, keep anti_predictions empty.")
#     return "\n".join(lines)


# TODO: Needs to redo this, get some inspiration from literature
def _build_verifier_prompt(
    hypothesis: Optional[Dict[str, Any]],
    candidates: List[str],
    blind: bool,
) -> str:
    raise NotImplementedError


# def _build_verifier_prompt(
#     hypothesis: Optional[Dict[str, Any]],
#     candidates: List[str],
#     blind: bool,
# ) -> str:
#     """Build a verifier prompt for blind or hypothesis-aware detection."""
#     lines = [
#         "You are given multiple candidate responses.",
#         "Exactly one response was produced with an internal latent intervention.",
#         "Your task: choose which candidate is most likely the intervened sample.",
#         "Respond with only the number (1..N).",
#     ]
#     if not blind and hypothesis is not None:
#         lines.append("")
#         lines.append(f"Hypothesis: {hypothesis.get('hypothesis_text', '')}")
#         effects = hypothesis.get("predicted_effects", [])
#         if effects:
#             lines.append("Predicted effects: " + ", ".join(effects))
#     lines.append("")
#     for idx, cand in enumerate(candidates, 1):
#         lines.append(f"[{idx}] {cand}")
#     return "\n".join(lines)


def _parse_int(text: str, n: int) -> Optional[int]:
    """Extract a 1..n choice from a model response string."""
    for tok in text.strip().split():
        if tok.isdigit():
            val = int(tok)
            if 1 <= val <= n:
                return val
    return None


def _call_openai_json(
    client: OpenAI, model: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    """Call the OpenAI Responses API and return raw text output."""
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return response.output_text


def _generate_hypotheses(
    cfg,
    evidence_packs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate hypotheses from evidence packs using the explainer model.

    Args:
        cfg: Hydra config with `llm.explainer` settings.
        evidence_packs: Evidence packs for hypothesis generation.

    Returns:
        List of hypothesis dicts, one per latent_id.

    Raises:
        RuntimeError: If the OpenAI API call fails.
        ValueError: If the response cannot be parsed to JSON (handled gracefully).
    """
    llm_cfg = cfg.evals.causal_autointerp_framework.llm.explainer
    if not llm_cfg.enabled:
        return []

    client = OpenAI()
    model = llm_cfg.model
    temperature = float(getattr(llm_cfg, "temperature", 0.2))
    max_tokens = int(getattr(llm_cfg, "max_tokens", 512))
    per_latent = int(getattr(llm_cfg, "evidence_per_latent", 8))
    effects_menu = list(getattr(llm_cfg, "effects_menu", []))
    include_antipredictions = bool(getattr(llm_cfg, "include_antipredictions", True))

    by_latent: Dict[int, List[Dict[str, Any]]] = {}
    for pack in evidence_packs:
        by_latent.setdefault(pack["latent_id"], []).append(pack)

    hypotheses: List[Dict[str, Any]] = []
    for latent_id, packs in by_latent.items():
        packs = packs[:per_latent]
        prompt = _build_explainer_prompt(packs, effects_menu, include_antipredictions)
        raw = _call_openai_json(client, model, prompt, temperature, max_tokens)
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {
                "latent_id": latent_id,
                "hypothesis_text": raw.strip(),
                "predicted_effects": [],
                "scope": {"buckets": [], "expected_strength": "small"},
                "anti_predictions": [],
            }
        if "latent_id" not in parsed:
            parsed["latent_id"] = latent_id
        hypotheses.append(parsed)
    return hypotheses


def _run_verification(
    cfg,
    evidence_packs: List[Dict[str, Any]],
    hypotheses: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run blind and hypothesis-informed verification on held-out evidence.

    Args:
        cfg: Hydra config with `llm.verifier` settings.
        evidence_packs: Held-out evidence packs for evaluation.
        hypotheses: Mapping from latent_id to hypothesis dict.

    Returns:
        List of verification result dicts with blind/hypothesis choices.

    Raises:
        RuntimeError: If the OpenAI API call fails.
    """
    llm_cfg = cfg.evals.causal_autointerp_framework.llm.verifier
    if not llm_cfg.enabled:
        return []
    client = OpenAI()
    model = llm_cfg.model
    temperature = float(getattr(llm_cfg, "temperature", 0.0))
    max_tokens = int(getattr(llm_cfg, "max_tokens", 128))

    results: List[Dict[str, Any]] = []
    rng = random.Random(cfg.seed)
    for pack in evidence_packs:
        candidates = pack["baseline_samples"] + [pack["steered_sample"]]
        indices = list(range(len(candidates)))
        rng.shuffle(indices)
        shuffled = [candidates[i] for i in indices]
        correct_idx = indices.index(len(candidates) - 1) + 1

        blind_prompt = _build_verifier_prompt(None, shuffled, blind=True)
        blind_raw = _call_openai_json(
            client, model, blind_prompt, temperature, max_tokens
        )
        blind_choice = _parse_int(blind_raw, len(shuffled))

        hyp = hypotheses.get(pack["latent_id"])
        hyp_prompt = _build_verifier_prompt(hyp, shuffled, blind=False)
        hyp_raw = _call_openai_json(client, model, hyp_prompt, temperature, max_tokens)
        hyp_choice = _parse_int(hyp_raw, len(shuffled))

        results.append(
            {
                "latent_id": pack["latent_id"],
                "prompt_id": pack["prompt_id"],
                "blind_choice": blind_choice,
                "hyp_choice": hyp_choice,
                "correct_choice": correct_idx,
                "blind_correct": int(blind_choice == correct_idx)
                if blind_choice
                else 0,
                "hyp_correct": int(hyp_choice == correct_idx) if hyp_choice else 0,
            }
        )
    return results


def run_autointerp_framework(cfg, model, tokenizer, wrapped_modules) -> None:
    """Run the HH-RLHF causal autointerp pipeline end-to-end.

    Args:
        cfg: Hydra config containing `causal_autointerp_framework` settings.
        model: Loaded model with TopKLoRALinearSTE modules.
        tokenizer: Tokenizer for prompt rendering and decoding.
        wrapped_modules: Unused placeholder for compatibility with evals API.

    Returns:
        None. Writes JSON/JSONL artifacts to `output_dir`.

    Raises:
        RuntimeError: If any stage fails (model forward, generation, API calls).
        FileNotFoundError: If required datasets are missing.
    """
    eval_cfg = cfg.evals.causal_autointerp_framework
    output_dir = eval_cfg.output_dir
    _ensure_dir(output_dir)

    device = _select_device()
    model.to(device)
    model.eval()

    logger.info("Listing available TopK adapters...")
    list_available_adapters(model, verbose=True)

    prompts_path = os.path.join(output_dir, "prompts.jsonl")
    latent_index_path = os.path.join(output_dir, "latent_index.json")
    latent_stats_path = os.path.join(output_dir, "latent_stats.jsonl")
    # Paths for downstream stages (calibration/evidence/hypothesis/verification)
    # are defined inside their respective stage blocks when enabled.

    if eval_cfg.stages.prompts:
        analysis_prompts, eval_prompts = _build_prompt_records(cfg)
        _write_jsonl(prompts_path, analysis_prompts + eval_prompts)
    else:
        all_prompts = _read_jsonl(prompts_path)
        analysis_prompts = all_prompts
        eval_prompts = all_prompts

    modules = _list_topk_modules(model)
    latent_index = _build_latent_index(modules)

    with open(latent_index_path, "w", encoding="utf-8") as f:
        json.dump(latent_index, f, indent=2)
    if eval_cfg.stages.latent_stats:
        latent_stats = _collect_latent_stats(
            cfg, model, tokenizer, analysis_prompts, modules, latent_index
        )

        _write_jsonl(latent_stats_path, latent_stats)

    # TODO: Checked until here and works.

    # if eval_cfg.stages.calibration:
    #     calibration = _calibrate_interventions(
    #         cfg,
    #         model,
    #         tokenizer,
    #         latent_index_map,
    #         latent_index,
    #         analysis_prompts,
    #     )
    #     _write_jsonl(calibration_path, calibration)
    # else:
    #     calibration = _read_jsonl(calibration_path)

    # if eval_cfg.stages.evidence_train:
    #     evidence_train = _build_evidence_packs(
    #         cfg,
    #         model,
    #         tokenizer,
    #         analysis_prompts,
    #         latent_index_map,
    #         latent_index,
    #         calibration,
    #         per_latent=int(getattr(eval_cfg.evidence, "train_per_latent", 10)),
    #     )
    #     _write_jsonl(evidence_train_path, evidence_train)

    # if eval_cfg.stages.evidence_eval:
    #     evidence_eval = _build_evidence_packs(
    #         cfg,
    #         model,
    #         tokenizer,
    #         eval_prompts,
    #         latent_index_map,
    #         latent_index,
    #         calibration,
    #         per_latent=int(getattr(eval_cfg.evidence, "eval_per_latent", 10)),
    #     )
    #     _write_jsonl(evidence_eval_path, evidence_eval)

    # if eval_cfg.stages.hypothesis:
    #     evidence_train = _read_jsonl(evidence_train_path)
    #     hypotheses = _generate_hypotheses(cfg, evidence_train)
    #     _write_jsonl(hypotheses_path, hypotheses)

    # if eval_cfg.stages.verification:
    #     evidence_eval = _read_jsonl(evidence_eval_path)
    #     hypotheses = _read_jsonl(hypotheses_path)
    #     hypothesis_map = {h["latent_id"]: h for h in hypotheses if "latent_id" in h}
    #     verification = _run_verification(cfg, evidence_eval, hypothesis_map)
    #     _write_jsonl(verification_path, verification)

    # if eval_cfg.stages.summary:
    #     verification_results = _read_jsonl(verification_path)
    #     summary = _summarize_verification(verification_results)
    #     with open(summary_path, "w", encoding="utf-8") as f:
    #         json.dump(summary, f, indent=2)

    # logger.info("Causal autointerp framework run complete.")

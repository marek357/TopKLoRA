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
import heapq
import math
from typing import Any, Dict, List, Tuple, Literal

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from src.models import TopKLoRALinearSTE, _hard_topk_mask
from src.steering import FeatureSteeringContext, list_available_adapters
from src.utils import hh_string_to_messages, generate_completions_from_prompts
from .causal_explainer import run_explainer
from .autointerp_utils import (
    _ensure_dir,
    _write_jsonl,
    _read_jsonl,
    build_latent_index,
)


logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    """Select CUDA if available, else MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _stable_hash(text: str) -> str:
    """Return a short stable hash for deriving IDs from prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def _build_prompt_records(cfg) -> List[Dict[str, Any]]:
    """Load HH-RLHF prompts and assign buckets.

    Args:
        cfg: Hydra config with eval settings at `cfg.evals.causal_autointerp_framework`.

    Returns:
        A list of dicts with keys: prompt_id, prompt, bucket, meta.

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
    return prompts


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


def _collect_latent_stats(
    cfg,
    model,
    tokenizer,
    prompts: List[Dict[str, Any]],
    modules: Dict[str, TopKLoRALinearSTE],
    latent_index: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute per-latent activation statistics across analysis prompts.

    Args:
        cfg: Hydra config with `latents` settings (epsilon, quantiles, etc.).
        model: Loaded model with TopKLoRALinearSTE modules.
        tokenizer: Tokenizer used for prompt encoding.
        prompts: List of analysis prompt records.
        modules: Mapping of module name to TopKLoRALinearSTE.
        latent_index: Global latent index entries (latent_id, adapter_name, feature_idx).

    Returns:
        A tuple of (latent_stats, top_prompts) where:
        - latent_stats: per-latent stats dictionaries
        - top_prompts: per-latent top prompt lists (prompt_id, score)

    Notes:
        For TopK layers, stats are computed on the *gated* activations using the
        hard TopK mask, and p_active counts prompts where the latent was gated on
        at least once (per-prompt activity, not per-token).

    Raises:
        RuntimeError: If forward pass fails (e.g., device mismatch).
    """
    device = next(model.parameters()).device
    lat_cfg = cfg.evals.causal_autointerp_framework.latents
    quantile_samples = int(getattr(lat_cfg, "quantile_samples", 2000))
    top_prompt_count = int(getattr(lat_cfg, "top_prompt_count", 100))

    sums = np.zeros(len(latent_index), dtype=np.float64)
    sums_sq = np.zeros(len(latent_index), dtype=np.float64)
    counts = np.zeros(len(latent_index), dtype=np.int64)
    active_counts = np.zeros(len(latent_index), dtype=np.int64)
    samples: List[List[float]] = [[] for _ in range(len(latent_index))]
    top_prompts: List[List[Tuple[float, str]]] = [[] for _ in range(len(latent_index))]

    adapter_offsets = {}
    for entry in latent_index:
        if entry["feature_idx"] == 0:
            adapter_offsets[entry["adapter_name"]] = entry["latent_id"]

    model.eval()
    for rec in tqdm(prompts, desc="Collecting latent stats"):
        prompt_text = rec["prompt"]
        prompt_id = rec.get("prompt_id", "")
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
            z = module._last_z.detach().cpu()
            if z.ndim == 3:
                z = z[0]
            if z.ndim != 2:
                continue
            k_now = int(module._current_k())
            mask = _hard_topk_mask(z, k_now)
            z_eff = z * mask
            active_mask = mask.any(dim=0)
            z_relu = F.relu(z_eff)

            # aggregate using max over tokens
            agg = z_relu.max(dim=0).values
            adapter_offset = adapter_offsets.get(name)
            if adapter_offset is None:
                continue
            start = adapter_offset
            end = adapter_offset + module.r

            agg_cpu = agg.numpy()
            active_cpu = active_mask.numpy().astype(np.int64)
            sums[start:end] += agg_cpu
            sums_sq[start:end] += agg_cpu * agg_cpu
            counts[start:end] += 1
            active_counts[start:end] += active_cpu

            if quantile_samples > 0:
                for feature_idx, val in enumerate(agg_cpu):
                    samples[start + feature_idx].append(float(val))
            if top_prompt_count > 0 and prompt_id:
                for feature_idx, val in enumerate(agg_cpu):
                    latent_id = start + feature_idx
                    heap = top_prompts[latent_id]
                    score = float(val)
                    if len(heap) < top_prompt_count:
                        heapq.heappush(heap, (score, prompt_id))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, prompt_id))

    latent_stats: List[Dict[str, Any]] = []
    top_prompt_records: List[Dict[str, Any]] = []
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
        if top_prompt_count > 0:
            heap = top_prompts[latent_id]
            top_sorted = sorted(heap, key=lambda x: x[0], reverse=True)
            top_prompt_records.append(
                {
                    "latent_id": latent_id,
                    "adapter_name": entry["adapter_name"],
                    "feature_idx": entry["feature_idx"],
                    "prompts": [
                        {
                            "prompt_id": pid,
                            "score": float(score),
                            "activation": float(score),
                        }
                        for score, pid in top_sorted
                    ],
                }
            )
    return latent_stats, top_prompt_records


def _build_feature_dict(
    latent_entry: Dict[str, Any], effect: str
) -> Dict[str, List[Tuple[int, str]]]:
    """Build a steering dict for a single latent and effect."""
    return {latent_entry["adapter_name"]: [(latent_entry["feature_idx"], effect)]}


def _require_nonempty_jsonl(path: str, kind: str) -> List[Dict[str, Any]]:
    """Load a required JSONL file and ensure it is present + non-empty."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing pre-cached {kind} file: {path}")
    records = _read_jsonl(path)
    if not records:
        raise ValueError(f"Pre-cached {kind} file is empty: {path}")
    return records


def _validate_precached_latent_records_strict(
    records: List[Dict[str, Any]],
    latent_index: List[Dict[str, Any]],
    record_name: str,
) -> List[Dict[str, Any]]:
    """Validate strict latent-id equality against current runtime latent index."""
    validated: List[Dict[str, Any]] = []
    max_latent_id = len(latent_index) - 1

    for rec_idx, rec in enumerate(records):
        if "latent_id" not in rec:
            raise ValueError(
                f"{record_name} record #{rec_idx} is missing 'latent_id'."
            )

        try:
            latent_id = int(rec["latent_id"])
        except (TypeError, ValueError):
            raise ValueError(
                f"{record_name} record #{rec_idx} has invalid latent_id={rec.get('latent_id')!r}."
            ) from None

        if latent_id < 0 or latent_id > max_latent_id:
            raise ValueError(
                f"{record_name} record #{rec_idx} has latent_id={latent_id}, "
                f"expected range [0, {max_latent_id}]."
            )

        expected = latent_index[latent_id]
        rec_adapter_name = rec.get("adapter_name")
        rec_feature_idx = rec.get("feature_idx")
        if rec_adapter_name is None or rec_feature_idx is None:
            raise ValueError(
                f"{record_name} record #{rec_idx} (latent_id={latent_id}) must include "
                "'adapter_name' and 'feature_idx' for strict validation."
            )

        try:
            rec_feature_idx_int = int(rec_feature_idx)
        except (TypeError, ValueError):
            raise ValueError(
                f"{record_name} record #{rec_idx} has invalid feature_idx={rec_feature_idx!r}."
            ) from None

        expected_adapter = expected["adapter_name"]
        expected_feature_idx = int(expected["feature_idx"])
        if (
            rec_adapter_name != expected_adapter
            or rec_feature_idx_int != expected_feature_idx
        ):
            raise ValueError(
                f"{record_name} latent mismatch at record #{rec_idx}, latent_id={latent_id}: "
                f"expected (adapter_name={expected_adapter!r}, feature_idx={expected_feature_idx}), "
                f"got (adapter_name={rec_adapter_name!r}, feature_idx={rec_feature_idx_int})."
            )

        updated = dict(rec)
        updated["latent_id"] = latent_id
        updated["feature_idx"] = rec_feature_idx_int
        validated.append(updated)

    return validated


def _select_latents(
    cfg,
    latent_index: List[Dict[str, Any]],
    latent_stats: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample latents from the p_active distribution.

    Returns:
        selected_latents: ordered list of latent entries to keep.
        selection_records: per-latent diagnostics and scores.
    """
    sel_cfg = cfg.evals.causal_autointerp_framework.latents.selection
    if not bool(getattr(sel_cfg, "enabled", True)):
        return latent_index, []

    stats_by_id = {rec["latent_id"]: rec for rec in latent_stats}
    seed = int(getattr(sel_cfg, "seed", cfg.seed))
    rng = np.random.default_rng(seed)

    p_active_min = float(getattr(sel_cfg, "p_active_min", 0.01))
    p_active_max = float(getattr(sel_cfg, "p_active_max", 1.0))
    dead_max = float(getattr(sel_cfg, "dead_p_active_max", 0.0))
    max_latents = int(getattr(sel_cfg, "max_latents", 128))

    candidate_entries: List[Dict[str, Any]] = []
    candidate_stats: List[Dict[str, Any]] = []
    for entry in latent_index:
        stats = stats_by_id.get(entry["latent_id"])
        if stats is None:
            continue
        p_active = float(stats.get("p_active", 0.0))
        if p_active <= dead_max:
            continue
        if p_active < p_active_min or p_active > p_active_max:
            continue
        candidate_entries.append(entry)
        candidate_stats.append(stats)

    if not candidate_entries:
        return [], []

    if len(candidate_entries) <= max_latents:
        top_indices = np.arange(len(candidate_entries))
    else:
        top_indices = rng.choice(
            len(candidate_entries), size=max_latents, replace=False
        )

    selected_entries = [candidate_entries[i] for i in top_indices]
    selection_records = []
    for i, entry in enumerate(candidate_entries):
        stats = stats_by_id.get(entry["latent_id"], {})
        selection_records.append(
            {
                "latent_id": entry["latent_id"],
                "adapter_name": entry["adapter_name"],
                "feature_idx": entry["feature_idx"],
                "p_active": float(stats.get("p_active", 0.0)),
                "selected": int(i in top_indices),
            }
        )

    selection_records.sort(key=lambda x: x["p_active"], reverse=True)
    return selected_entries, selection_records


def _compute_latent_alpha(
    latent_id: int,
    ev_cfg,
    latent_stats_by_id: Dict[int, Dict[str, Any]],
    top_prompts_by_id: Dict[int, Dict[str, Any]],
) -> float:
    """Resolve steering alpha for one latent under the configured alpha_mode."""
    alpha_mode = str(getattr(ev_cfg, "alpha_mode", "fixed")).strip().lower()

    if alpha_mode == "fixed":
        alpha = float(getattr(ev_cfg, "alpha"))
    elif alpha_mode == "mean_active":
        stats = latent_stats_by_id.get(latent_id)
        if stats is None:
            raise ValueError(
                f"alpha_mode=mean_active requires latent_stats for latent_id={latent_id}, "
                "but no record was found."
            )
        p_active = float(stats.get("p_active", 0.0))
        if p_active <= 0.0:
            raise ValueError(
                f"alpha_mode=mean_active failed for latent_id={latent_id}: "
                f"p_active must be > 0, got {p_active}."
            )
        alpha = float(stats.get("mu", 0.0)) / p_active
    elif alpha_mode == "mean_unconditional":
        stats = latent_stats_by_id.get(latent_id)
        if stats is None:
            raise ValueError(
                f"alpha_mode=mean_unconditional requires latent_stats for latent_id={latent_id}, "
                "but no record was found."
            )
        alpha = float(stats.get("mu", 0.0))
    elif alpha_mode == "top_prompt_mean":
        top_rec = top_prompts_by_id.get(latent_id)
        if top_rec is None:
            raise ValueError(
                f"alpha_mode=top_prompt_mean requires top_prompts for latent_id={latent_id}, "
                "but no record was found."
            )
        prompts = top_rec.get("prompts", [])
        if not prompts:
            raise ValueError(
                f"alpha_mode=top_prompt_mean failed for latent_id={latent_id}: "
                "top_prompts record has no prompts."
            )
        vals: List[float] = []
        for prompt_rec in prompts:
            if "activation" in prompt_rec:
                vals.append(float(prompt_rec["activation"]))
            elif "score" in prompt_rec:
                vals.append(float(prompt_rec["score"]))
        if not vals:
            raise ValueError(
                f"alpha_mode=top_prompt_mean failed for latent_id={latent_id}: "
                "no numeric activation/score values in prompts."
            )
        alpha = float(sum(vals) / len(vals))
    else:
        raise ValueError(
            f"Unsupported alpha_mode={alpha_mode!r}. "
            "Expected one of: fixed, mean_active, mean_unconditional, top_prompt_mean."
        )

    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(
            f"Computed alpha is invalid for latent_id={latent_id}, alpha_mode={alpha_mode!r}: "
            f"alpha={alpha}. Expected finite alpha > 0."
        )
    return alpha


def _build_evidence_packs(
    cfg,
    model,
    tokenizer,
    prompts: List[Dict[str, Any]],
    latent_entries: List[Dict[str, Any]],
    top_prompt_map: Dict[int, List[str]] = None,
    latent_stats_by_id: Dict[int, Dict[str, Any]] = None,
    top_prompts_by_id: Dict[int, Dict[str, Any]] = None,
    sample_top_prompts: bool = False,
    mode: Literal["explainer", "verifier"] = "explainer",
) -> List[Dict[str, Any]]:
    """Create evidence packs containing baseline and steered samples.

    Args:
        cfg: Hydra config with `evidence` and `generation` settings.
        model: Loaded model for text generation.
        tokenizer: Tokenizer for prompt rendering/decoding.
        prompts: Prompt records to sample from.
        latent_entries: Ordered list of latent entries to process.
        top_prompt_map: Mapping of latent_id -> ordered prompt_ids.
        latent_stats_by_id: Mapping of latent_id -> latent stats record.
        top_prompts_by_id: Mapping of latent_id -> top-prompts record.
        sample_top_prompts: Whether to sample prompts from top_prompt_map.
        mode: "explainer" or "verifier" mode for evidence pack formatting.

    Returns:
        List of evidence pack dictionaries with baseline and steered samples.

    Raises:
        RuntimeError: If generation fails or model forward fails.
    """
    if mode == "explainer":
        ev_cfg = cfg.evals.causal_autointerp_framework.evidence.explainer
    else:
        ev_cfg = cfg.evals.causal_autointerp_framework.evidence.verifier

    gen_cfg = ev_cfg.generation
    intervention_type = getattr(ev_cfg, "intervention_type")
    latent_stats_by_id = latent_stats_by_id or {}
    top_prompts_by_id = top_prompts_by_id or {}
    device = next(model.parameters()).device
    gen_kwargs = dict(
        max_new_tokens=int(getattr(gen_cfg, "max_new_tokens")),
        do_sample=bool(getattr(gen_cfg, "do_sample")),
        eos_token_id=getattr(model.generation_config, "eos_token_id"),
    )
    if (
        bool(getattr(gen_cfg, "do_sample"))
        and getattr(gen_cfg, "temperature") is not None
    ):
        gen_kwargs["temperature"] = float(getattr(gen_cfg, "temperature"))
    if bool(getattr(gen_cfg, "do_sample")) and getattr(gen_cfg, "top_p") is not None:
        gen_kwargs["top_p"] = float(getattr(gen_cfg, "top_p"))

    rng = random.Random(cfg.seed)
    evidence_records: List[Dict[str, Any]] = []
    prompt_by_id = {rec["prompt_id"]: rec for rec in prompts if rec.get("prompt_id")}

    for entry in tqdm(
        latent_entries,
        desc=f"Building evidence packs. {getattr(ev_cfg, 'per_latent')} sets per latent",
    ):
        latent_id = entry["latent_id"]
        if intervention_type == "steer_with_alpha":
            amplification = _compute_latent_alpha(
                latent_id=latent_id,
                ev_cfg=ev_cfg,
                latent_stats_by_id=latent_stats_by_id,
                top_prompts_by_id=top_prompts_by_id,
            )
        else:
            amplification = 1.0

        top_ids = []
        if top_prompt_map and latent_id in top_prompt_map:
            top_ids = [pid for pid in top_prompt_map[latent_id]]
        if not top_ids:
            top_ids = [rec["prompt_id"] for rec in prompts if rec.get("prompt_id")]

        selected_ids: List[str]
        if sample_top_prompts:
            sample_k = min(len(top_ids), int(getattr(ev_cfg, "per_latent")))
            selected_ids = rng.sample(top_ids, k=sample_k)
        elif top_prompt_map:
            selected_ids = top_ids[: getattr(ev_cfg, "per_latent")]
        else:
            selected_ids = top_ids[: int(getattr(ev_cfg, "per_latent"))]

        if not selected_ids:
            logger.warning(
                "No prompt ids available for latent_id=%s; skipping evidence pack generation.",
                latent_id,
            )
            continue

        missing_prompt_ids = [pid for pid in selected_ids if pid not in prompt_by_id]
        if missing_prompt_ids:
            logger.warning(
                "Missing %d/%d prompt ids for latent_id=%s from prompt map; "
                "continuing with available prompts.",
                len(missing_prompt_ids),
                len(selected_ids),
                latent_id,
            )
        selected = [prompt_by_id[pid] for pid in selected_ids if pid in prompt_by_id]
        if not selected:
            logger.warning(
                "All selected prompt ids were missing for latent_id=%s; skipping.",
                latent_id,
            )
            continue

        if intervention_type == "zero_ablate":
            feature_dict = _build_feature_dict(entry, "disable")
        else:
            feature_dict = _build_feature_dict(entry, "enable")
        rendered_prompts = [
            _render_prompt(tokenizer, rec["prompt"]) for rec in selected
        ]

        logging.debug(f"First three rendered prompts: {rendered_prompts[:3]}")
        baseline_count = int(getattr(ev_cfg, "baseline_samples"))
        baseline_lists = [[] for _ in selected]
        for _ in range(baseline_count):
            batch = generate_completions_from_prompts(
                model,
                tokenizer,
                rendered_prompts,
                max_length=int(getattr(gen_cfg, "max_new_tokens")),
                device=str(device),
                gen_kwargs=gen_kwargs,
            )
            for i, text in enumerate(batch):
                baseline_lists[i].append(text)

        enc = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(getattr(gen_cfg, "max_new_tokens")),
        ).to(device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # TODO: This steers all token positions, we may need to consider rewriting the
        # context manager and the hook to only target the last token/top activating token
        with FeatureSteeringContext(
            model,
            feature_dict,
            verbose=False,
            amplification=amplification,
        ):
            logging.debug(msg=f"feature_dict: {feature_dict}")
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
        input_lengths = enc["input_ids"].shape[1]
        steered_samples = []

        for seq in out:
            steered_samples.append(
                tokenizer.decode(
                    seq[input_lengths:], skip_special_tokens=True
                ).strip()  # truncate the prompt after padding.
            )

        for prompt_rec, baseline, steered in zip(
            selected, baseline_lists, steered_samples
        ):
            evidence_records.append(
                {
                    "latent_id": latent_id,
                    "adapter_name": entry["adapter_name"],
                    "feature_idx": entry["feature_idx"],
                    "prompt_id": prompt_rec["prompt_id"],
                    "prompt": prompt_rec["prompt"],
                    "baseline_samples": baseline,
                    "steered_sample": steered,
                    "intervention_meta": {
                        "type": intervention_type,
                        "alpha": None
                        if intervention_type == "zero_ablate"
                        else amplification,
                    },
                }
            )

    return evidence_records


def run_autointerp_framework(cfg, model, tokenizer) -> None:
    """Run the HH-RLHF causal autointerp pipeline end-to-end.

    Args:
        cfg: Hydra config containing `causal_autointerp_framework` settings.
        model: Loaded model with TopKLoRALinearSTE modules.
        tokenizer: Tokenizer for prompt rendering and decoding.

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
    top_prompts_path = os.path.join(output_dir, "top_prompts.jsonl")
    latent_selection_path = os.path.join(output_dir, "latent_selection.jsonl")
    evidence_explainer_path = os.path.join(output_dir, "evidence_explainer.jsonl")
    evidence_verifier_path = os.path.join(output_dir, "evidence_verifier.jsonl")

    precached_cfg = getattr(eval_cfg, "precached", None)
    use_precached = bool(
        getattr(precached_cfg, "enabled", False) if precached_cfg else False
    )
    precached_stats_dir = (
        str(getattr(precached_cfg, "stats_dir", "")).strip() if precached_cfg else ""
    )
    precached_recompute_selection = bool(
        getattr(precached_cfg, "recompute_selection_from_stats", False)
        if precached_cfg
        else False
    )

    if use_precached:
        logger.info(
            "Pre-cached mode enabled. Bypassing stages.prompts=%s, "
            "stages.latent_stats=%s, stages.latent_selection=%s.",
            bool(eval_cfg.stages.prompts),
            bool(eval_cfg.stages.latent_stats),
            bool(eval_cfg.stages.latent_selection),
        )
        if precached_recompute_selection:
            logger.info(
                "precached.recompute_selection_from_stats=true; latent selection "
                "will be recomputed from pre-cached latent_stats using "
                "latents.selection settings."
            )
        if not precached_stats_dir:
            raise ValueError(
                "precached.enabled is true but precached.stats_dir is not set."
            )
        if not os.path.isdir(precached_stats_dir):
            raise FileNotFoundError(
                f"Pre-cached stats_dir does not exist: {precached_stats_dir}"
            )
        prompts_precached_path = os.path.join(precached_stats_dir, "prompts.jsonl")
        all_prompts = _require_nonempty_jsonl(prompts_precached_path, "prompts")
    elif eval_cfg.stages.prompts:
        all_prompts = _build_prompt_records(cfg)
        _write_jsonl(prompts_path, all_prompts)
    else:
        all_prompts = _read_jsonl(prompts_path)

    modules = _list_topk_modules(model)
    latent_index, _ = build_latent_index(modules)

    with open(latent_index_path, "w", encoding="utf-8") as f:
        json.dump(latent_index, f, indent=2)

    if use_precached:
        latent_stats_precached_path = os.path.join(
            precached_stats_dir, "latent_stats.jsonl"
        )
        top_prompts_precached_path = os.path.join(precached_stats_dir, "top_prompts.jsonl")
        latent_stats = _require_nonempty_jsonl(latent_stats_precached_path, "latent_stats")
        top_prompts = _require_nonempty_jsonl(top_prompts_precached_path, "top_prompts")
        latent_stats = _validate_precached_latent_records_strict(
            latent_stats,
            latent_index,
            "latent_stats",
        )
        top_prompts = _validate_precached_latent_records_strict(
            top_prompts,
            latent_index,
            "top_prompts",
        )
    elif eval_cfg.stages.latent_stats:
        latent_stats, top_prompts = _collect_latent_stats(
            cfg, model, tokenizer, all_prompts, modules, latent_index
        )

        _write_jsonl(latent_stats_path, latent_stats)
        _write_jsonl(top_prompts_path, top_prompts)
    else:
        latent_stats = _read_jsonl(latent_stats_path)
        top_prompts = _read_jsonl(top_prompts_path)

    top_prompt_map = None
    if top_prompts:
        top_prompt_map = {
            rec["latent_id"]: [p["prompt_id"] for p in rec.get("prompts", [])]
            for rec in top_prompts
            if "latent_id" in rec
        }
    latent_stats_by_id = {
        int(rec["latent_id"]): rec for rec in latent_stats if "latent_id" in rec
    }
    top_prompts_by_id = {
        int(rec["latent_id"]): rec for rec in top_prompts if "latent_id" in rec
    }

    selected_latents = latent_index
    selection_records: List[Dict[str, Any]] = []
    if use_precached:
        if precached_recompute_selection:
            selected_latents, selection_records = _select_latents(
                cfg,
                latent_index,
                latent_stats,
            )
            if selection_records:
                _write_jsonl(latent_selection_path, selection_records)
            logger.info(
                "Recomputed latent selection from pre-cached latent_stats; "
                "selected %d latents.",
                len(selected_latents),
            )
        else:
            latent_selection_precached_path = os.path.join(
                precached_stats_dir, "latent_selection.jsonl"
            )
            selection_records = _require_nonempty_jsonl(
                latent_selection_precached_path,
                "latent_selection",
            )
            selection_records = _validate_precached_latent_records_strict(
                selection_records,
                latent_index,
                "latent_selection",
            )
            selected_latents = [
                entry
                for entry in selection_records
                if int(entry.get("selected", 0)) == 1
            ]
            logger.info(
                "Loaded %d selected latents from pre-cached selection file.",
                len(selected_latents),
            )
    elif eval_cfg.stages.latent_selection:
        selected_latents, selection_records = _select_latents(
            cfg,
            latent_index,
            latent_stats,
        )
        if selection_records:
            _write_jsonl(latent_selection_path, selection_records)
    else:
        selection_records = _read_jsonl(latent_selection_path)
        selected_latents = [
            entry for entry in selection_records if entry.get("selected") == 1
        ]
        logging.info(f"Loaded {len(selected_latents)} selected latents from file.")

    # Log selected latents per adapter
    if selected_latents:
        adapter_counts: Dict[str, int] = {}
        for entry in selected_latents:
            adapter_counts[entry["adapter_name"]] = (
                adapter_counts.get(entry["adapter_name"], 0) + 1
            )
        for adapter_name, count in sorted(adapter_counts.items()):
            logger.info(
                "Selected %d latents from adapter %s",
                count,
                adapter_name,
            )

    if eval_cfg.stages.evidence_explainer:
        evidence_train = _build_evidence_packs(
            cfg,
            model,
            tokenizer,
            all_prompts,
            selected_latents,
            top_prompt_map=top_prompt_map,
            latent_stats_by_id=latent_stats_by_id,
            top_prompts_by_id=top_prompts_by_id,
            sample_top_prompts=False,
            mode="explainer",
        )
        _write_jsonl(evidence_explainer_path, evidence_train)

    if eval_cfg.stages.evidence_verifier:
        evidence_eval = _build_evidence_packs(
            cfg,
            model,
            tokenizer,
            all_prompts,
            selected_latents,
            top_prompt_map=top_prompt_map,
            latent_stats_by_id=latent_stats_by_id,
            top_prompts_by_id=top_prompts_by_id,
            sample_top_prompts=True,
            mode="verifier",
        )
        _write_jsonl(evidence_verifier_path, evidence_eval)

    # Run explainer stage to generate hypotheses
    if eval_cfg.stages.hypothesis:
        logger.info("Running causal explainer to generate hypotheses...")

        # Get vLLM server URL from config or use default
        vllm_cfg = getattr(eval_cfg, "vllm", None)
        vllm_base_url = (
            getattr(vllm_cfg, "base_url", "http://localhost:8080/v1")
            if vllm_cfg
            else "http://localhost:8080/v1"
        )

        hypotheses = run_explainer(
            cfg,
            output_dir=output_dir,
            vllm_base_url=vllm_base_url,
        )
        logger.info(f"Generated {len(hypotheses)} hypotheses")

    if eval_cfg.stages.verification:
        raise NotImplementedError("Verification stage not yet implemented")

    if eval_cfg.stages.summary:
        raise NotImplementedError("Summary stage not yet implemented")

    logger.info("Causal autointerp framework run complete.")

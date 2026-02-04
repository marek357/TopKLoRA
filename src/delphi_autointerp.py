import asyncio
import dataclasses
import json
import os
import sys
import hashlib
import heapq
import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from delphi.clients import Offline, OpenRouter
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.explainers import ContrastiveExplainer, DefaultExplainer
from delphi.latents import LatentCache, LatentDataset
from delphi.pipeline import Pipeline, process_wrapper
from delphi.scorers import (
    DetectionScorer,
    FuzzingScorer,
    OpenAISimulator,
    SurprisalScorer,
)
from torch.utils.data import DataLoader
from src.autointerp_utils import _read_jsonl, _write_jsonl, build_latent_index
from src.utils import hh_string_to_messages, autointerp_violates_alternation
import logging
from dotenv import load_dotenv
from src.openai_client import OpenAIClient

# Add path for our improvements

device = "cuda" if torch.cuda.is_available() else "cpu"


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_prompt_from_example(example) -> str:
    if "prompt" in example and example["prompt"]:
        return example["prompt"].strip()
    if "text" in example and example["text"]:
        return example["text"].strip()
    for key in ("chosen", "rejected"):
        if key in example and example[key]:
            msgs = hh_string_to_messages(example[key])
            for msg in msgs:
                if msg.get("role") == "user":
                    return msg.get("content", "").strip()
    return ""


def _extract_first_user(msgs) -> str:
    for msg in msgs:
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def _extract_continuation_messages(example, choice, rng):
    selected = choice
    if selected is None:
        if "chosen" in example and "rejected" in example:
            selected = rng.choice(["chosen", "rejected"])

    if selected in ("chosen", "rejected") and selected in example:
        convo_text = example[selected]
        msgs = hh_string_to_messages(convo_text)
        return msgs, convo_text, selected

    return None, "", selected


def _stream_and_format_dataset(
    dataset,
    max_examples,
    rng,
    continuation_choice,
    dataset_split,
    dataset_config,
):
    inputs = []
    prompt_records = []
    incorrectly_formatted = 0
    skipped_empty_prompt = 0
    skipped_empty_continuation = 0

    for example in islice(dataset, max_examples):
        prompt_text = _extract_prompt_from_example(example)
        messages, continuation_text, continuation_source = (
            _extract_continuation_messages(example, continuation_choice, rng)
        )

        prompt_text = _extract_first_user(messages) or prompt_text
        if not prompt_text:
            skipped_empty_prompt += 1
            continue
        if not continuation_text and not messages:
            skipped_empty_continuation += 1
            continue
        if not continuation_text and messages:
            continuation_text = json.dumps(messages, ensure_ascii=False)

        if autointerp_violates_alternation(messages):
            incorrectly_formatted += 1
            continue

        prompt_id = _stable_hash(f"{prompt_text}\n\n{continuation_text}")
        record = {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "continuation": continuation_text,
            "continuation_source": continuation_source or "random",
            "meta": {
                "split": dataset_split,
                "config": dataset_config,
            },
        }
        record["meta"].update(
            {
                "num_messages": len(messages),
                "multiturn": len(messages) > 2,
            }
        )
        inputs.append({"input": messages})

        prompt_records.append(record)

    stats = {
        "incorrectly_formatted": incorrectly_formatted,
        "skipped_empty_prompt": skipped_empty_prompt,
        "skipped_empty_continuation": skipped_empty_continuation,
        "total_records": len(prompt_records),
    }
    return inputs, prompt_records, stats


def _collect_latent_stats_from_cache(
    cache,
    wrapped_modules,
    prompt_ids,
    exp_cfg,
):
    # Hyperparameters controlling top-prompt tracking
    top_prompt_count = int(getattr(exp_cfg, "top_prompt_count"))

    latent_index, adapter_offsets = build_latent_index(wrapped_modules)

    # Allocate aggregation buffers across the full latent space
    total_latents = len(latent_index)
    sums = np.zeros(total_latents, dtype=np.float64)
    sums_sq = np.zeros(total_latents, dtype=np.float64)
    active_counts = np.zeros(total_latents, dtype=np.int64)
    top_prompts = [[] for _ in range(total_latents)] if top_prompt_count > 0 else None

    # Aggregate per-sequence maxima into global statistics
    def _update_from_sequence(seq_idx, max_by_latent, adapter_offset):
        prompt_id = prompt_ids[seq_idx] if seq_idx < len(prompt_ids) else None
        for feature_idx, max_val in max_by_latent.items():
            latent_global = adapter_offset + feature_idx
            sums[latent_global] += max_val
            sums_sq[latent_global] += max_val * max_val
            active_counts[latent_global] += 1
            if top_prompts is not None and prompt_id:
                heap = top_prompts[latent_global]
                score = float(max_val)
                if len(heap) < top_prompt_count:
                    heapq.heappush(heap, (score, prompt_id))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, prompt_id))

    # Walk cached non-zero activations and compute per-sequence max for each latent
    for name, module in wrapped_modules.items():
        hookpoint = f"{name}.topk"
        if hookpoint not in cache.cache.latent_locations:
            continue
        locations_tensor = cache.cache.latent_locations[hookpoint]
        activations_tensor = cache.cache.latent_activations[hookpoint]
        if locations_tensor is None or activations_tensor is None:
            continue
        if locations_tensor.numel() == 0:
            continue
        adapter_offset = adapter_offsets[name]
        locations = locations_tensor.numpy()
        activations = activations_tensor.numpy()
        order = np.argsort(locations[:, 0])
        loc_sorted = locations[order]
        act_sorted = activations[order]
        current_seq = int(loc_sorted[0, 0])
        max_by_latent = {}
        for loc, act in zip(loc_sorted, act_sorted):
            seq_idx = int(loc[0])
            if seq_idx != current_seq:
                _update_from_sequence(current_seq, max_by_latent, adapter_offset)
                max_by_latent = {}
                current_seq = seq_idx
            feature_idx = int(loc[2])
            prev = max_by_latent.get(feature_idx)
            if prev is None or act > prev:
                max_by_latent[feature_idx] = act
        _update_from_sequence(current_seq, max_by_latent, adapter_offset)

    # Convert aggregated buffers into summary records
    counts = max(len(prompt_ids), 1)
    latent_stats = []
    top_prompt_records = []
    for entry in latent_index:
        latent_id = entry["latent_id"]
        mean = sums[latent_id] / counts
        var = sums_sq[latent_id] / counts - mean * mean
        sigma = float(max(var, 1e-12) ** 0.5)
        p_active = active_counts[latent_id] / counts
        latent_stats.append(
            {
                "latent_id": latent_id,
                "mu": float(mean),
                "sigma": float(sigma),
                "p_active": float(p_active),
                "notes": {"dead": int(p_active <= 0.0)},
                "adapter_name": entry["adapter_name"],
                "feature_idx": entry["feature_idx"],
            }
        )
        if top_prompts is not None:
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


def _select_latents_from_stats(cfg, latent_stats):
    sel_cfg = getattr(cfg.evals.auto_interp, "latent_selection")

    enabled = bool(getattr(sel_cfg, "enabled"))
    if not enabled:
        return [], []

    seed = int(getattr(cfg, "seed"))
    rng = np.random.default_rng(seed)

    p_active_min = float(getattr(sel_cfg, "p_active_min"))
    p_active_max = float(getattr(sel_cfg, "p_active_max"))
    max_latents = int(getattr(sel_cfg, "max_latents"))

    candidate_entries = []
    for stats in latent_stats:
        p_active = float(stats.get("p_active", 0.0))
        if p_active < p_active_min or p_active > p_active_max:
            continue
        candidate_entries.append(stats)

    if not candidate_entries:
        return [], []

    if len(candidate_entries) <= max_latents:
        top_indices = np.arange(len(candidate_entries))
    else:
        top_indices = rng.choice(
            len(candidate_entries), size=max_latents, replace=False
        )

    selection_records = []
    top_index_set = set(int(i) for i in top_indices.tolist())
    for i, entry in enumerate(candidate_entries):
        selection_records.append(
            {
                "latent_id": entry.get("latent_id"),
                "adapter_name": entry.get("adapter_name"),
                "feature_idx": entry.get("feature_idx"),
                "p_active": float(entry.get("p_active", 0.0)),
                "selected": int(i in top_index_set),
            }
        )

    selection_records.sort(key=lambda x: x["p_active"], reverse=True)
    selected_entries = [candidate_entries[i] for i in top_indices]
    return selected_entries, selection_records


def build_hookpoint_offsets(wrapped_modules):
    """
    Build a mapping from hookpoint names to their global latent ID offsets.
    Assigns unique global IDs to latents across all hookpoints.

    Args:
        wrapped_modules: Dictionary of module name -> TopKLoRALinearSTE module.

    Returns:
        tuple: (hookpoint_offsets, hookpoint_widths, total_latents)
            - hookpoint_offsets: dict mapping hookpoint name to global ID offset
            - hookpoint_widths: dict mapping hookpoint name to latent width
            - total_latents: total number of latents across all hookpoints
    """
    hookpoint_offsets = {}
    hookpoint_widths = {}
    offset = 0

    for name, module in wrapped_modules.items():
        hookpoint = f"{name}.topk"
        width = module.r
        hookpoint_offsets[hookpoint] = offset
        hookpoint_widths[hookpoint] = width
        offset += width

    total_latents = offset
    logging.info(
        f"Built hookpoint offsets: {len(hookpoint_offsets)} hookpoints, "
        f"{total_latents} total latents"
    )
    return hookpoint_offsets, hookpoint_widths, total_latents


def compute_co_occurrence_from_cache(cache, wrapped_modules):
    """
    Compute co-occurrence statistics from cached latent activations.

    For each position (batch_idx, seq_idx), finds all latents that fired
    and increments co-occurrence counts for each pair.

    Args:
        cache: LatentCache object with populated cache.latent_locations
        wrapped_modules: Dictionary of module name -> TopKLoRALinearSTE module

    Returns:
        tuple: (co_occurrence, hookpoint_meta)
            - co_occurrence: dict[int, dict[int, int]] where
                co_occurrence[latent_a][latent_b] = count of positions
                where both latents fired together
            - hookpoint_meta: dict with offsets, widths, total_latents
    """
    from collections import defaultdict
    from tqdm import tqdm

    hookpoint_offsets, hookpoint_widths, total_latents = build_hookpoint_offsets(
        wrapped_modules
    )

    # Group all active latents by position (batch_idx, seq_idx)
    position_to_latents = defaultdict(list)

    for hookpoint, locations in cache.cache.latent_locations.items():
        if locations is None or locations.numel() == 0:
            continue

        offset = hookpoint_offsets.get(hookpoint, 0)
        locations_np = locations.numpy()

        for i in range(locations_np.shape[0]):
            batch_idx = int(locations_np[i, 0])
            seq_idx = int(locations_np[i, 1])
            local_latent_idx = int(locations_np[i, 2])
            global_latent_id = offset + local_latent_idx

            position_to_latents[(batch_idx, seq_idx)].append(global_latent_id)

    # Compute co-occurrence counts
    co_occurrence = defaultdict(lambda: defaultdict(int))

    n_positions = len(position_to_latents)
    logging.info(f"Computing co-occurrence across {n_positions} positions...")

    for position, latent_ids in tqdm(
        position_to_latents.items(), desc="Computing co-occurrence"
    ):
        # For each pair of latents at this position, increment co-occurrence
        latent_ids_sorted = sorted(set(latent_ids))  # dedupe within position
        n = len(latent_ids_sorted)
        for i in range(n):
            latent_a = latent_ids_sorted[i]
            for j in range(i + 1, n):
                latent_b = latent_ids_sorted[j]
                # Store bidirectionally for easy lookup
                co_occurrence[latent_a][latent_b] += 1
                co_occurrence[latent_b][latent_a] += 1

    # Convert to regular dicts for serialization
    co_occurrence = {k: dict(v) for k, v in co_occurrence.items()}

    total_pairs = sum(len(v) for v in co_occurrence.values()) // 2
    logging.info(
        f"Co-occurrence computed: {len(co_occurrence)} latents with "
        f"{total_pairs} unique co-occurring pairs"
    )

    hookpoint_meta = {
        "offsets": hookpoint_offsets,
        "widths": hookpoint_widths,
        "total_latents": total_latents,
    }

    return co_occurrence, hookpoint_meta


def get_latent_info(global_latent_id, hookpoint_offsets, hookpoint_widths):
    """
    Get hookpoint and local index for a global latent ID.

    Args:
        global_latent_id: The global latent ID.
        hookpoint_offsets: dict mapping hookpoint name to global ID offset
        hookpoint_widths: dict mapping hookpoint name to latent width

    Returns:
        dict with 'hookpoint', 'local_idx', and 'global_id' keys.
    """
    for hookpoint, offset in hookpoint_offsets.items():
        width = hookpoint_widths.get(hookpoint, 0)
        if offset <= global_latent_id < offset + width:
            return {
                "hookpoint": hookpoint,
                "local_idx": global_latent_id - offset,
                "global_id": global_latent_id,
            }
    return {"hookpoint": None, "local_idx": None, "global_id": global_latent_id}


def save_co_occurrence(co_occurrence, hookpoint_meta, save_dir):
    """
    Save co-occurrence data to disk.

    Args:
        co_occurrence: dict[int, dict[int, int]] co-occurrence counts
        hookpoint_meta: dict with offsets, widths, total_latents
        save_dir: Path to directory where data will be saved
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save co-occurrence as JSON
    co_occurrence_path = save_dir / "co_occurrence.json"
    with open(co_occurrence_path, "w") as f:
        # Convert int keys to strings for JSON compatibility
        json_compatible = {
            str(k): {str(k2): v2 for k2, v2 in v.items()}
            for k, v in co_occurrence.items()
        }
        json.dump(json_compatible, f, indent=2)

    # Save hookpoint offsets for ID resolution
    offsets_path = save_dir / "hookpoint_offsets.json"
    with open(offsets_path, "w") as f:
        json.dump(hookpoint_meta, f, indent=2)

    logging.info(f"Saved co-occurrence data to {save_dir}")


class ChatTemplateCollator:
    def __init__(self, tokenizer, device, max_length=1024, add_generation_prompt=True):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.add_generation_prompt = add_generation_prompt

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # For generation tasks, left padding is typically better
        self.original_padding_side = tokenizer.padding_side
        self.tokenizer.padding_side = "left"

    def __call__(self, examples):
        texts = []
        for ex in examples:
            msgs = ex.get("input", ex.get("chosen", ex.get("rejected")))
            text = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
            )
            texts.append(text)

        # Efficient batch tokenization with optimized settings
        batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            # Add these for extra efficiency:
            return_attention_mask=True,
            return_token_type_ids=False,  # Not needed for most models
        ).to(self.device)  # Move directly to device

        return batch

    def __del__(self):
        # Restore original padding side
        if hasattr(self, "original_padding_side"):
            self.tokenizer.padding_side = self.original_padding_side


def save_explanation(result, model_str, explainer_type):
    latent_str = str(result.record.latent)

    # TODO: set the dirs through config
    safe = latent_str.replace(".", "_").replace(":", "_").replace(" ", "_")

    out_dir = f"autointerp/{model_str}/explanations/" + explainer_type
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{safe}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "explanation": result.explanation,
                # 'activating_sequences': result.activating_sequences,
                # 'non_activating_sequences': result.non_activating_sequences,
            },
            f,
            indent=2,
        )
    return result


def save_score(result, model_str, scorer):
    # TODO: set the dirs through config
    # 1) Build a safe filename from the latent
    latent_str = str(result.record.latent)  # e.g. "layers.5.self.topk:42"
    safe = latent_str.replace(".", "_").replace(":", "_").replace(" ", "_")

    # 2) Ensure output directory
    out_dir = f"autointerp/{model_str}/scores/{scorer}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{safe}.json")

    # 3) Serialize result.score
    score_obj = result.score

    if hasattr(score_obj, "to_json_string"):
        # HF ModelOutput
        text = score_obj.to_json_string()
    elif isinstance(score_obj, list):
        # List of dataclasses (e.g. SurprisalOutput)
        # Convert each element to dict
        dicts = [dataclasses.asdict(elem) for elem in score_obj]
        text = json.dumps(dicts, indent=2)
    elif isinstance(score_obj, dict):
        # Already a dict
        text = json.dumps(score_obj, indent=2)
    else:
        # Fallback to plain repr
        text = json.dumps({"score": score_obj}, indent=2)

    # 4) Write
    with open(path, "w") as f:
        f.write(text)

    return result


def delphi_collect_activations(cfg, model, tokenizer, wrapped_modules):
    print("starting activation collection")
    exp_cfg = cfg.evals.auto_interp.activation_collection

    if not hasattr(exp_cfg, "dataset_name"):
        raise ValueError(
            "cfg must have 'dataset_name' attribute for activation collection"
        )

    dataset_split = getattr(exp_cfg, "dataset_split", "train")
    dataset_config = getattr(exp_cfg, "dataset_config_name")
    flat_ds = load_dataset(
        exp_cfg.dataset_name,
        name=dataset_config,
        split=dataset_split,
        streaming=True,
    )

    continuation_choice = getattr(exp_cfg, "dataset_continuation")
    rng = random.Random(int(getattr(cfg, "seed", 42)))

    max_batches = getattr(exp_cfg, "max_batches")
    flat_ds, prompt_records, stats = _stream_and_format_dataset(
        flat_ds,
        max_batches,
        rng,
        continuation_choice,
        dataset_split,
        dataset_config,
    )
    if stats["total_records"]:
        logging.info(
            "Skipped %d/%d (%.2f%%) incorrectly formatted examples.",
            stats["incorrectly_formatted"],
            stats["total_records"],
            stats["incorrectly_formatted"] / stats["total_records"] * 100,
        )
    if stats["skipped_empty_prompt"]:
        logging.info(
            "Skipped %d examples with empty prompt.",
            stats["skipped_empty_prompt"],
        )
    if stats["skipped_empty_continuation"]:
        logging.info(
            "Skipped %d examples with empty continuation.",
            stats["skipped_empty_continuation"],
        )
    chat_collate = ChatTemplateCollator(
        tokenizer,
        device,
        max_length=getattr(exp_cfg, "seq_len"),
        add_generation_prompt=False,
    )

    loader = DataLoader(  # type: ignore[arg-type]
        flat_ds,
        batch_size=exp_cfg.batch_size,
        shuffle=False,
        collate_fn=chat_collate,
        drop_last=False,
    )

    n_tokens = int(getattr(exp_cfg, "n_tokens"))
    seq_len = int(getattr(exp_cfg, "seq_len"))
    n_seqs = (n_tokens + seq_len - 1) // seq_len

    rows = []
    for batch in loader:
        # batch["input_ids"]: Tensor[B, SEQ_LEN]
        arr = batch["input_ids"].detach().cpu().clone()  # shape (B, SEQ_LEN)
        for row in arr:
            rows.append(row)
            if len(rows) >= n_seqs:
                break
        if len(rows) >= n_seqs:
            break

    # shape (n_seqs, SEQ_LEN)
    tokens_array = torch.stack(rows[:n_seqs], dim=0)
    prompt_ids = [rec["prompt_id"] for rec in prompt_records][:n_seqs]

    topk_modules = {
        f"{name}.topk": module.topk for name, module in wrapped_modules.items()
    }

    # Temporarily enable TopK experiment mode so hooks see gated latents
    original_modes = {}
    for module in wrapped_modules.values():
        if hasattr(module, "is_topk_experiment"):
            original_modes[module] = module.is_topk_experiment
            module.is_topk_experiment = True

    cache = LatentCache(
        model=model,
        hookpoint_to_sparse_encode=topk_modules,
        batch_size=exp_cfg.batch_size,
        transcode=False,
    )

    try:
        cache.run(
            n_tokens=n_tokens,
            tokens=tokens_array,
        )
        print("Cache collection complete. Checking cache contents...")
        total_entries = 0
        for hookpoint, locations in cache.cache.latent_locations.items():
            num_entries = int(locations.shape[0]) if locations is not None else 0
            total_entries += num_entries
            print(f"  {hookpoint}: {num_entries} non-zero activations")
        if total_entries == 0:
            print("WARNING: No latent activations were recorded.")
        out_dir = Path(
            f"delphi_cache/{cfg.model.module_type.name}_k{cfg.model.k}_r{cfg.model.r}_layer{cfg.model.layer}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        cache.save_splits(n_splits=4, save_dir=out_dir)
        widths = {f"{name}.topk": wrapped_modules[name].r for name in wrapped_modules}

        for hookpoint in widths:
            # the directory is literally raw_dir / hookpoint
            hp_dir = out_dir / hookpoint
            hp_dir.mkdir(parents=True, exist_ok=True)

            config = {"hookpoint": hookpoint, "width": widths[hookpoint]}
            with open(hp_dir / "config.json", "w") as f:
                json.dump(config, f)

        stats_dir = out_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        prompts_path = stats_dir / "prompts.jsonl"
        latent_stats_path = stats_dir / "latent_stats.jsonl"
        top_prompts_path = stats_dir / "top_prompts.jsonl"
        _write_jsonl(str(prompts_path), prompt_records)

        latent_stats, top_prompt_records = _collect_latent_stats_from_cache(
            cache,
            wrapped_modules,
            prompt_ids,
            exp_cfg,
        )

        _write_jsonl(str(latent_stats_path), latent_stats)
        _write_jsonl(str(top_prompts_path), top_prompt_records)

        # Compute and save co-occurrence if enabled
        compute_co_occurrence = bool(getattr(exp_cfg, "compute_co_occurrence", False))
        if compute_co_occurrence:
            logging.info("Computing co-occurrence from cache...")
            co_occurrence, hookpoint_meta = compute_co_occurrence_from_cache(
                cache, wrapped_modules
            )
            save_co_occurrence(co_occurrence, hookpoint_meta, stats_dir)
    finally:
        for module, original in original_modes.items():
            module.is_topk_experiment = original

    if bool(cfg.evals.auto_interp.latent_selection.enabled):
        delphi_select_latents(cfg)


def delphi_select_latents(cfg):
    stats_dir = Path(
        f"delphi_cache/{cfg.model.module_type.name}_k{cfg.model.k}_r{cfg.model.r}_layer{cfg.model.layer}/stats"
    )
    if not os.path.exists(stats_dir):
        raise ValueError(f"Stats dir not found: {stats_dir}")

    latent_stats_path = stats_dir / "latent_stats.jsonl"
    if not os.path.exists(latent_stats_path):
        raise ValueError(f"Latent stats file not found: {latent_stats_path}")

    latent_stats = _read_jsonl(latent_stats_path)
    selected_latents, selection_records = _select_latents_from_stats(
        cfg,
        latent_stats,
    )
    latent_selection_path = getattr(
        cfg, "latent_selection_path", str(stats_dir / "latent_selection.jsonl")
    )
    if selection_records:
        _write_jsonl(str(latent_selection_path), selection_records)
        print(
            f"Selected {len(selected_latents)} latents; wrote selection to {latent_selection_path}"
        )


def delphi_score(cfg, model, tokenizer, wrapped_modules):
    cfg_exp = cfg.evals.auto_interp.delphi_scoring
    # Create model-specific identifier string based on config
    model_str = f"{cfg.model.module_type.name}_k{cfg.model.k}_r{cfg.model.r}_layer{cfg.model.layer}"

    topk_modules = [
        f"{name}.topk"
        for name, _ in wrapped_modules.items()
        # if "q_proj" not in name  # filter out query projections -- these have already been analyzed
    ]

    logging.info(f"Initial topk modules: {topk_modules}")
    topk_modules = [elem for elem in topk_modules if str(cfg.model.layer) in elem]
    logging.info(f"Filtered to layer {cfg.model.layer} modules: {topk_modules}")

    model.cpu()
    del model
    del wrapped_modules

    selection_path = f"delphi_cache/{model_str}/stats/latent_selection.jsonl"

    # Load interpretability rankings and get priority latents
    selection_records = _read_jsonl(selection_path)
    priority_latents = {}
    selected_latents_counter = 0
    for entry in selection_records:
        if entry.get("selected") == 1:
            # make sure to add ".topk" suffix
            priority_latents.setdefault(entry["adapter_name"] + ".topk", []).append(
                entry["feature_idx"]
            )
            selected_latents_counter += 1

    topk_modules = [name for name in topk_modules if name in priority_latents]
    logging.info(
        f"Loaded {selected_latents_counter} priority latents for {len(topk_modules)} modules from {selection_path}"
    )

    logging.info(f"Topk modules after filtering: {topk_modules}")
    # 1) Load the raw cache you saved
    dataset = LatentDataset(
        raw_dir=Path(
            f"delphi_cache/{cfg.model.module_type.name}_k{cfg.model.k}_r{cfg.model.r}_layer{cfg.model.layer}"
        ),
        modules=topk_modules,
        latents={
            # Focus on most interpretable latents only
            name: torch.tensor(priority_latents[name], dtype=torch.long)
            for name in topk_modules
        },
        tokenizer=tokenizer,
        # TODO: Figure out what is the optimal config here.
        sampler_cfg=SamplerConfig(
            n_examples_train=50,  # Increased training examples for better analysis
            n_examples_test=40,  # More test examples for robust evaluation
            n_quantiles=10,  # Standard quantile analysis
            train_type="mix",  # Mixed sampling for diverse training examples
            test_type="quantiles",  # Quantile-based testing
            ratio_top=0.3,  # Focus on top 30% activations
        ),
        # TODO: Figure out how these may possibly improve explanations
        constructor_cfg=ConstructorConfig(
            # Enhanced contrastive analysis for better interpretability
            # faiss_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            # faiss_embedding_cache_enabled=True,
            # faiss_embedding_cache_dir=".embedding_cache",
            # example_ctx_len=32,      # Context length for examples
            # min_examples=200,  # Minimum examples for robust analysis
            # n_non_activating=20,     # Non-activating examples for contrast
            # center_examples=True,    # Center examples for better analysis
            # non_activating_source="FAISS",  # Use FAISS for better negative examples
            # neighbours_type="co-occurrence"  # Co-occurrence based neighbors
        ),
    )

    scoring_cfg = getattr(cfg_exp, "scoring_client", None)
    provider = (
        str(getattr(scoring_cfg, "provider", "offline")).strip().lower()
        if scoring_cfg is not None
        else "offline"
    )

    if provider == "offline":
        # GPU Memory Management Configuration
        num_gpus = 1
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set PyTorch CUDA memory management for fragmentation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            num_gpus = len(visible_devices.split(","))

            logging.info(
                f"🔧 Using GPUs: {visible_devices} (multi-GPU with tensor parallelism)"
            )

        explainer_config = scoring_cfg.offline_config

        client = Offline(
            explainer_config.model_name,
            num_gpus=num_gpus,
            max_model_len=explainer_config.max_model_len,  # smaller KV → faster & safer
            max_memory=explainer_config.max_memory,  # GB per GPU
            prefix_caching=explainer_config.prefix_caching,
            batch_size=explainer_config.batch_size,
            enforce_eager=explainer_config.enforce_eager,  # allow CUDA graphs
            number_tokens_to_generate=explainer_config.number_tokens_to_generate,
            # max_num_batched_tokens=3072,
        )

        # Add device attribute for SurprisalScorer compatibility
        client.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        if torch.cuda.is_available():
            logging.info(
                "✅ Model loaded successfully with multi-GPU tensor parallelism!"
            )
            logging.info(
                f"   - GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')} (tensor parallelism)"
            )
        else:
            logging.info(f"✅ Model loaded successfully on {client.device}!")
    elif provider == "openai":
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment."
            )

        explainer_config = scoring_cfg.openai_config
        openai_model = getattr(explainer_config, "openai_model", "gpt-4o-mini")
        openai_base_url = getattr(explainer_config, "openai_base_url", None)
        openai_temperature = float(getattr(explainer_config, "openai_temperature", 0.0))
        openai_max_tokens = int(getattr(explainer_config, "openai_max_tokens", 3000))
        openai_timeout = float(getattr(explainer_config, "openai_timeout", 60.0))
        cost_monitor_enabled = bool(
            getattr(explainer_config, "cost_monitor_enabled", False)
        )
        cost_monitor_every_n_requests = int(
            getattr(explainer_config, "cost_monitor_every_n_requests", 10)
        )
        input_cost_per_1m = getattr(explainer_config, "input_cost_per_1m", None)
        output_cost_per_1m = getattr(explainer_config, "output_cost_per_1m", None)
        if input_cost_per_1m is not None:
            input_cost_per_1m = float(input_cost_per_1m)
        if output_cost_per_1m is not None:
            output_cost_per_1m = float(output_cost_per_1m)

        client = OpenAIClient(
            openai_model,
            api_key=api_key,
            base_url=openai_base_url,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            timeout=openai_timeout,
            tokenizer=tokenizer,
            cost_monitor_enabled=cost_monitor_enabled,
            cost_monitor_every_n_requests=cost_monitor_every_n_requests,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )

        client.device = torch.device("cpu")
        logging.info(f"✅ Using OpenAI API model: {openai_model}")
    else:
        raise ValueError(
            f"Unknown scoring client provider '{provider}'. Use 'offline' or 'openai'."
        )

    if not cfg_exp.use_openai_simulator:
        # TODO may have to modify to adapt for cot (not originally implemented)
        # explainer = DefaultExplainer(client, cot=True)
        explainer = DefaultExplainer(client, cot=False)
        explainer_pipe = process_wrapper(
            explainer,
            postprocess=lambda x: save_explanation(x, model_str, "enhanced_default"),
        )

        detection_scorer = DetectionScorer(
            client, tokenizer=tokenizer, n_examples_shown=5
        )

        # Enhanced pipeline with multiple scoring methods
        logging.info(
            f"Running enhanced interpretability analysis on {len(priority_latents)} latents"
        )

        # Multi-stage pipeline
        # Capture model_str in closure for the async function
        _model_str = model_str

        async def comprehensive_scoring(explained):
            """Run both detection and surprisal scoring."""
            rec = explained.record
            rec.explanation = explained.explanation
            rec.extra_examples = rec.not_active

            # Run detection scoring
            try:
                det_result = await detection_scorer(rec)
                save_score(det_result, _model_str, "enhanced_detection")
            except Exception as e:
                logging.error(f"Detection scoring failed for {rec.latent}: {e}")

            return explained

        comprehensive_pipe = process_wrapper(comprehensive_scoring)

        # 5) Run the enhanced pipeline
        pipeline = Pipeline(
            dataset,
            explainer_pipe,
            comprehensive_pipe,
        )
    else:
        simulator = OpenAISimulator(
            client,
            tokenizer=tokenizer,  # use the same tokenizer as your dataset
        )

        # 3. Wrap it in a process pipe (optional preprocess/postprocess callbacks)
        def sim_preprocess(result):
            # Convert record+interpretation into simulator input
            return result

        sim_pipe = process_wrapper(
            simulator,
            preprocess=sim_preprocess,
            postprocess=lambda x: save_score(x, model_str, "OpenAISimulator"),
        )

        # 4. Build and run the pipeline
        pipeline = Pipeline(
            dataset,  # loads feature records & contexts
            sim_pipe,  # runs simulation scoring in one stage
        )

    max_concurrent = int(getattr(cfg_exp, "max_concurrent", 1))

    asyncio.run(pipeline.run(max_concurrent=max_concurrent))

    logging.info(
        f"✅ Pipeline completed with max_concurrent={max_concurrent} (memory-safe)"
    )
    # Generate summary after analysis
    logging.info(f"\n{'=' * 60}")
    logging.info("ENHANCED INTERPRETABILITY ANALYSIS COMPLETE")
    logging.info("Results saved to:")
    logging.info(
        f"  - Explanations: autointerp/{model_str}/explanations/enhanced_default/"
    )
    logging.info(
        f"  - Detection scores: autointerp/{model_str}/scores/enhanced_detection/"
    )
    logging.info(f"{'=' * 60}\n")

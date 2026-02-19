import asyncio
import dataclasses
import json
import os
import hashlib
import heapq
import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from delphi.clients import Offline
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.explainers import DefaultExplainer
from delphi.latents import LatentDataset
from delphi.pipeline import Pipeline, process_wrapper
from delphi.scorers import (
    DetectionScorer,
    OpenAISimulator,
)
from torch.utils.data import DataLoader
from .autointerp_utils import _read_jsonl, _write_jsonl, build_latent_index
from src.utils import hh_string_to_messages, autointerp_violates_alternation
import logging
from .openai_client import OpenAIClient
from .streaming_latent_cache import make_latent_cache
from tqdm import tqdm

# Add path for our improvements

device = "cuda" if torch.cuda.is_available() else "cpu"


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_prompt_from_example(example) -> str:
    # Handle LMSYS format with "conversation" field
    if "conversation" in example and example["conversation"]:
        msgs = example["conversation"]
        if isinstance(msgs, list):
            for msg in msgs:
                if msg.get("role") == "user":
                    return msg.get("content", "").strip()
    # Fallback to other formats
    if "prompt" in example and example["prompt"]:
        return example["prompt"].strip()
    if "text" in example and example["text"]:
        return example["text"].strip()
    return ""


def _extract_first_user(msgs) -> str:
    for msg in msgs:
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def _extract_continuation_messages(example, choice, rng):
    # Handle LMSYS format with "conversation" field
    if "conversation" in example and example["conversation"]:
        msgs = example["conversation"]
        if isinstance(msgs, list) and len(msgs) > 0:
            # Convert to JSON string for continuation_text
            convo_text = json.dumps(msgs, ensure_ascii=False)
            return msgs, convo_text, "conversation"

    # Fallback to HH-RLHF format if available
    selected = choice
    if selected is None:
        if "chosen" in example and "rejected" in example:
            selected = rng.choice(["chosen", "rejected"])

    if selected in ("chosen", "rejected") and selected in example:
        convo_text = example[selected]
        msgs = hh_string_to_messages(convo_text)
        return msgs, convo_text, selected

    return None, "", "none"


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

    for example in tqdm(islice(dataset, max_examples)):
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

    def _vectorized_per_seq_max(locations_np, activations_np, adapter_offset):
        """Vectorised: compute per-(sequence, feature) max and update stats."""
        if locations_np.shape[0] == 0:
            return

        seq_ids = locations_np[:, 0].astype(np.int64)
        feat_ids = locations_np[:, 2].astype(np.int64)
        act_vals = activations_np.ravel().astype(np.float64)

        global_ids = feat_ids + adapter_offset

        # Build composite key = seq_id * total_latents + global_id
        # to reduce to per-(seq, latent) max in one vectorised pass
        composite = seq_ids * total_latents + global_ids

        # np.maximum.at is O(n) — no sorting needed
        # First pass: find the max activation for each (seq, latent) pair
        unique_composites, inverse = np.unique(composite, return_inverse=True)
        max_vals = np.full(len(unique_composites), -np.inf, dtype=np.float64)
        np.maximum.at(max_vals, inverse, act_vals)

        # Recover seq_idx and global_latent_id from composite keys
        u_seq = unique_composites // total_latents
        u_global = unique_composites % total_latents

        # Bulk-update sums, sums_sq, active_counts
        np.add.at(sums, u_global, max_vals)
        np.add.at(sums_sq, u_global, max_vals * max_vals)
        np.add.at(active_counts, u_global, 1)

        # Top-prompt tracking (only needed for non-dead latents with prompt_ids)
        if top_prompts is not None and len(prompt_ids) > 0:
            for i in range(len(unique_composites)):
                s_idx = int(u_seq[i])
                g_id = int(u_global[i])
                score = float(max_vals[i])
                prompt_id = prompt_ids[s_idx] if s_idx < len(prompt_ids) else None
                if prompt_id is None:
                    continue
                heap = top_prompts[g_id]
                if len(heap) < top_prompt_count:
                    heapq.heappush(heap, (score, prompt_id))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, prompt_id))

    # Walk cached non-zero activations and compute per-sequence max for each latent
    for name, module in wrapped_modules.items():
        hookpoint = f"{name}.topk"
        adapter_offset = adapter_offsets[name]

        # Prefer materialized tensors; fall back to per-batch buffers (streaming mode)
        locations_tensor = cache.cache.latent_locations.get(hookpoint)
        activations_tensor = cache.cache.latent_activations.get(hookpoint)

        use_batches = False
        loc_batches = None
        act_batches = None

        if (locations_tensor is None or activations_tensor is None) and hasattr(
            cache.cache, "latent_locations_batches"
        ):
            loc_batches = cache.cache.latent_locations_batches.get(hookpoint, [])
            act_batches = cache.cache.latent_activations_batches.get(hookpoint, [])
            if loc_batches and act_batches:
                use_batches = True

        if use_batches:
            # Process batches incrementally — vectorised per batch.
            # Each batch is processed independently which is safe because
            # _vectorized_per_seq_max uses np.maximum.at / np.add.at which
            # correctly accumulate across multiple calls for the same seq/latent.
            #
            # NOTE: a sequence appearing in multiple batches will have its max
            # computed separately per batch, then both maxima contribute to sums.
            # This is the same semantics as the original code which sorted globally
            # and only flushed on seq_idx change — *provided* each batch contains
            # complete sequences. The upstream InMemoryCache.add() processes one
            # model-forward batch at a time with all its token positions, so each
            # batch already contains all positions for the sequences it covers.
            logging.info(
                f"  Processing {len(loc_batches)} batches for {hookpoint} (vectorised)..."
            )
            for batch_idx, (loc_batch, act_batch) in enumerate(
                zip(loc_batches, act_batches)
            ):
                if loc_batch.numel() == 0:
                    continue
                _vectorized_per_seq_max(
                    loc_batch.cpu().numpy(),
                    act_batch.cpu().numpy(),
                    adapter_offset,
                )
                if (batch_idx + 1) % 500 == 0:
                    logging.info(
                        f"    ... processed {batch_idx + 1}/{len(loc_batches)} batches"
                    )
        else:
            # Original path: use materialized tensors
            if locations_tensor is None or activations_tensor is None:
                continue
            if locations_tensor.numel() == 0:
                continue
            _vectorized_per_seq_max(
                locations_tensor.numpy(),
                activations_tensor.numpy(),
                adapter_offset,
            )

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

    Uses vectorised NumPy/scipy operations instead of Python loops for
    significant speed-up on large caches.  Processes data in chunks to
    avoid OOM on very large caches (billions of activations).

    Args:
        cache: LatentCache object with populated cache.latent_locations
        wrapped_modules: Dictionary of module name -> TopKLoRALinearSTE module

    Returns:
        tuple: (co_occurrence_coo, hookpoint_meta)
            - co_occurrence_coo: dict with keys 'row', 'col', 'count'
                (int32 numpy arrays, upper-triangle only, row < col)
            - hookpoint_meta: dict with offsets, widths, total_latents
    """
    import gc

    from scipy import sparse

    hookpoint_offsets, hookpoint_widths, total_latents = build_hookpoint_offsets(
        wrapped_modules
    )

    # ------------------------------------------------------------------
    # Helper: iterate (position_key, global_latent_id) pairs without
    # materializing everything into one giant array.
    # ------------------------------------------------------------------
    def _iter_location_chunks():
        """Yield (loc_np,) chunks — one per materialized tensor or batch."""
        resolved = cache.cache.latent_locations
        if resolved:
            for hookpoint, locations in resolved.items():
                if locations is None or locations.numel() == 0:
                    continue
                yield hookpoint, locations.numpy()
        elif hasattr(cache.cache, "latent_locations_batches"):
            for hookpoint, batches in cache.cache.latent_locations_batches.items():
                if not batches:
                    continue
                for batch in batches:
                    if batch.numel() == 0:
                        continue
                    yield hookpoint, batch.cpu().numpy()

    # Find max seq_idx for composite key stride
    max_seq = 0
    for _, loc_np in _iter_location_chunks():
        chunk_max = int(loc_np[:, 1].max())
        if chunk_max > max_seq:
            max_seq = chunk_max
    stride = max_seq + 1

    # ------------------------------------------------------------------
    # Build the global position-key mapping incrementally.
    # We need a consistent dense row index for each unique (batch, seq)
    # position across all chunks.  Do a two-pass approach:
    #   Pass 1: collect unique position keys (just int64 scalars, not rows)
    #   Pass 2: build sparse indicator columns chunk-by-chunk
    # ------------------------------------------------------------------
    logging.info("Co-occurrence pass 1: collecting unique position keys...")
    unique_keys_set: set[int] = set()
    total_activations = 0

    for _, loc_np in _iter_location_chunks():
        keys = loc_np[:, 0].astype(np.int64) * stride + loc_np[:, 1].astype(np.int64)
        unique_keys_set.update(keys.tolist())
        total_activations += len(keys)

    if not unique_keys_set:
        logging.warning("No latent activations found for co-occurrence.")
        hookpoint_meta = {
            "offsets": hookpoint_offsets,
            "widths": hookpoint_widths,
            "total_latents": total_latents,
        }
        return {}, hookpoint_meta

    # Build key→row mapping
    sorted_keys = np.array(sorted(unique_keys_set), dtype=np.int64)
    key_to_row = {int(k): i for i, k in enumerate(sorted_keys)}
    n_positions = len(sorted_keys)
    del unique_keys_set, sorted_keys
    gc.collect()

    logging.info(
        f"Co-occurrence pass 2: building sparse indicator "
        f"({n_positions:,} positions × {total_latents} latents "
        f"from {total_activations:,} activations)..."
    )

    # ------------------------------------------------------------------
    # Pass 2: accumulate the sparse indicator matrix chunk by chunk.
    # We accumulate directly into the co-occurrence matrix (latents × latents)
    # to avoid ever materializing the full (positions × latents) indicator.
    # For each chunk we build a small indicator, compute chunk_co = ind.T @ ind
    # and add it to a running total.
    # ------------------------------------------------------------------
    co_matrix = None  # will be (total_latents × total_latents) sparse

    # Process in larger chunks to amortize scipy overhead.  Group batches
    # until we hit a target chunk size.
    CHUNK_TARGET = 50_000_000  # ~50M activations per chunk ≈ 400MB RAM

    chunk_rows = []
    chunk_cols = []
    chunk_n = 0
    chunks_processed = 0

    def _flush_chunk():
        nonlocal co_matrix, chunk_rows, chunk_cols, chunk_n, chunks_processed
        if chunk_n == 0:
            return
        row_arr = np.concatenate(chunk_rows)
        col_arr = np.concatenate(chunk_cols)
        chunk_rows.clear()
        chunk_cols.clear()

        indicator = sparse.csr_matrix(
            (np.ones(len(row_arr), dtype=np.float32), (row_arr, col_arr)),
            shape=(n_positions, total_latents),
        )
        indicator.data = np.minimum(indicator.data, 1.0)

        chunk_co = (indicator.T @ indicator).tocsr()
        del indicator, row_arr, col_arr
        gc.collect()

        if co_matrix is None:
            co_matrix = chunk_co
        else:
            co_matrix = co_matrix + chunk_co
        del chunk_co
        gc.collect()

        chunks_processed += 1
        chunk_n = 0
        if chunks_processed % 5 == 0:
            logging.info(f"  ... processed {chunks_processed} co-occurrence chunks")

    for hookpoint, loc_np in _iter_location_chunks():
        offset = hookpoint_offsets.get(hookpoint, 0)

        keys = loc_np[:, 0].astype(np.int64) * stride + loc_np[:, 1].astype(np.int64)
        rows = np.array([key_to_row[int(k)] for k in keys], dtype=np.int32)
        cols = (loc_np[:, 2].astype(np.int64) + offset).astype(np.int32)

        chunk_rows.append(rows)
        chunk_cols.append(cols)
        chunk_n += len(rows)

        if chunk_n >= CHUNK_TARGET:
            _flush_chunk()

    _flush_chunk()

    if co_matrix is None:
        logging.warning("No co-occurrence data computed.")
        hookpoint_meta = {
            "offsets": hookpoint_offsets,
            "widths": hookpoint_widths,
            "total_latents": total_latents,
        }
        return {}, hookpoint_meta

    # Extract upper triangle
    co_coo = co_matrix.tocoo()
    del co_matrix
    gc.collect()

    mask = co_coo.row < co_coo.col
    rows_ut = co_coo.row[mask].astype(np.int32)
    cols_ut = co_coo.col[mask].astype(np.int32)
    vals_ut = co_coo.data[mask].astype(np.int32)
    del co_coo, mask
    gc.collect()

    total_pairs = len(vals_ut)
    n_latents_with_pairs = (
        len(np.unique(np.concatenate([rows_ut, cols_ut]))) if total_pairs > 0 else 0
    )
    logging.info(
        f"Co-occurrence computed: {n_latents_with_pairs} latents with "
        f"{total_pairs} unique co-occurring pairs"
    )

    co_occurrence_coo = {
        "row": rows_ut,
        "col": cols_ut,
        "count": vals_ut,
    }

    hookpoint_meta = {
        "offsets": hookpoint_offsets,
        "widths": hookpoint_widths,
        "total_latents": total_latents,
    }

    return co_occurrence_coo, hookpoint_meta


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


def save_co_occurrence(co_occurrence_coo, hookpoint_meta, save_dir):
    """
    Save co-occurrence data to disk as a compressed .npz file.

    The .npz stores three int32 arrays (row, col, count) representing the
    upper triangle of the symmetric co-occurrence matrix.  This is
    typically 50-100× smaller than the previous indented-JSON format.

    Args:
        co_occurrence_coo: dict with 'row', 'col', 'count' int32 arrays
            (upper-triangle only, row < col)
        hookpoint_meta: dict with offsets, widths, total_latents
        save_dir: Path to directory where data will be saved
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save co-occurrence as compressed numpy arrays
    co_occurrence_path = save_dir / "co_occurrence.npz"
    np.savez_compressed(
        co_occurrence_path,
        row=co_occurrence_coo["row"],
        col=co_occurrence_coo["col"],
        count=co_occurrence_coo["count"],
    )

    # Save hookpoint offsets for ID resolution
    offsets_path = save_dir / "hookpoint_offsets.json"
    with open(offsets_path, "w") as f:
        json.dump(hookpoint_meta, f, indent=2)

    nbytes = co_occurrence_path.stat().st_size
    logging.info(
        f"Saved co-occurrence data to {save_dir} "
        f"({len(co_occurrence_coo['row'])} pairs, {nbytes / 1e6:.1f} MB)"
    )


def load_co_occurrence(save_dir, as_sparse=False):
    """
    Load co-occurrence data from disk.

    Args:
        save_dir: Path to directory containing co_occurrence.npz and
            hookpoint_offsets.json.
        as_sparse: If True, return a scipy.sparse.coo_matrix.
            If False (default), return a nested dict
            ``{latent_a: {latent_b: count, ...}, ...}`` (bidirectional).

    Returns:
        tuple: (co_occurrence, hookpoint_meta)
    """
    save_dir = Path(save_dir)

    with open(save_dir / "hookpoint_offsets.json") as f:
        hookpoint_meta = json.load(f)

    data = np.load(save_dir / "co_occurrence.npz")
    rows = data["row"]
    cols = data["col"]
    counts = data["count"]

    if as_sparse:
        from scipy import sparse

        total = hookpoint_meta["total_latents"]
        # Build symmetric matrix from upper triangle
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])
        all_vals = np.concatenate([counts, counts])
        co_matrix = sparse.coo_matrix(
            (all_vals, (all_rows, all_cols)),
            shape=(total, total),
        )
        return co_matrix, hookpoint_meta

    # Build bidirectional nested dict
    co_occurrence = {}
    for r, c, v in zip(rows, cols, counts):
        r_int, c_int, v_int = int(r), int(c), int(v)
        co_occurrence.setdefault(r_int, {})[c_int] = v_int
        co_occurrence.setdefault(c_int, {})[r_int] = v_int
    return co_occurrence, hookpoint_meta


class ChatTemplateCollator:
    def __init__(self, tokenizer, device, max_length=1024, add_generation_prompt=True):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.add_generation_prompt = add_generation_prompt

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # For activation collection, use right padding to keep token positions consistent
        self.original_padding_side = tokenizer.padding_side
        self.tokenizer.padding_side = "right"

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
    logging.info("starting activation collection")
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

    # For lmsys-chat-1m, keep only English, non-redacted conversations.
    # Because the dataset is streamed we cannot know the total size upfront,
    # so we count accepted/rejected samples inside the filter callback.
    if "lmsys-chat" in exp_cfg.dataset_name:
        _filter_counts = {"before": 0, "after": 0}

        def _lmsys_filter(x):
            _filter_counts["before"] += 1
            keep = x.get("language") == "English" and x.get("redacted") is False
            if keep:
                _filter_counts["after"] += 1
            return keep

        flat_ds = flat_ds.filter(_lmsys_filter)
    else:
        _filter_counts = None

    continuation_choice = getattr(exp_cfg, "dataset_continuation")
    rng = random.Random(int(getattr(cfg, "seed", 42)))

    max_prompts = getattr(exp_cfg, "max_prompts")
    flat_ds, prompt_records, stats = _stream_and_format_dataset(
        flat_ds,
        max_prompts,
        rng,
        continuation_choice,
        dataset_split,
        dataset_config,
    )

    if _filter_counts is not None and _filter_counts["before"] > 0:
        pct = _filter_counts["after"] / _filter_counts["before"] * 100
        logging.info(
            "lmsys-chat filter: %d / %d samples kept (%.1f%%) "
            "[English=True, redacted=False]",
            _filter_counts["after"],
            _filter_counts["before"],
            pct,
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

    # Print example samples with applied chat template
    logging.info("\nExample samples with chat template applied:")
    for i in range(min(3, len(flat_ds))):
        formatted_text = tokenizer.apply_chat_template(
            flat_ds[i].get("input", []),
            tokenize=False,
            add_generation_prompt=False,
        )
        logging.info(f"\n--- Example {i + 1} ---\n{formatted_text}\n")

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

    streaming_cache = bool(getattr(exp_cfg, "streaming_cache", False))
    retry_enabled = bool(getattr(exp_cfg, "retry_on_oom", False))
    batch_size_min = int(getattr(exp_cfg, "batch_size_min", 1))
    backoff = float(getattr(exp_cfg, "batch_backoff", 0.5))
    max_retries = int(getattr(exp_cfg, "batch_max_retries", 2))

    current_batch_size = int(getattr(exp_cfg, "batch_size"))
    attempt = 0

    while True:
        cache = make_latent_cache(
            model=model,
            hookpoint_to_sparse_encode=topk_modules,
            batch_size=current_batch_size,
            transcode=False,
            streaming=streaming_cache,
        )

        try:
            cache.run(
                n_tokens=n_tokens,
                tokens=tokens_array,
            )
            break
        except RuntimeError as e:
            oom = "out of memory" in str(e).lower()
            if not (retry_enabled and oom):
                raise

            attempt += 1
            if attempt > max_retries or current_batch_size <= batch_size_min:
                logging.error(
                    "OOM even after retries (attempt %d, batch_size=%d). Re-raising.",
                    attempt,
                    current_batch_size,
                )
                raise

            next_bs = max(batch_size_min, max(1, int(current_batch_size * backoff)))
            logging.warning(
                "CUDA OOM at batch_size=%d; retrying with batch_size=%d (attempt %d/%d)",
                current_batch_size,
                next_bs,
                attempt,
                max_retries,
            )
            current_batch_size = next_bs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    try:
        logging.info("Cache collection complete. Checking cache contents...")
        total_entries = 0

        # In streaming mode the data lives in *_batches dicts, not the
        # materialized latent_locations / latent_activations dicts.
        # Count entries WITHOUT concatenating to avoid OOM on large caches.
        _loc_src = cache.cache.latent_locations
        if _loc_src:
            for hookpoint, locations in _loc_src.items():
                num_entries = int(locations.shape[0]) if locations is not None else 0
                total_entries += num_entries
                logging.info(f"  {hookpoint}: {num_entries} non-zero activations")
        elif hasattr(cache.cache, "latent_locations_batches"):
            for hp, batches in cache.cache.latent_locations_batches.items():
                if not batches:
                    continue
                num_entries = sum(b.shape[0] for b in batches)
                total_entries += num_entries
                logging.info(f"  {hp}: {num_entries} non-zero activations")
        if total_entries == 0:
            logging.warning("No latent activations were recorded.")
        out_dir = Path(
            f"delphi_cache/{cfg.model.module_type.name}_k{cfg.model.k}_r{cfg.model.r}_reg{cfg.model.reg}_layer{cfg.model.layer}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        cache.save_splits(n_splits=84, save_dir=out_dir)
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
        f"delphi_cache/{cfg.model.module_type.name}_k{cfg.model.k}_r{cfg.model.r}_reg{cfg.model.reg}_layer{cfg.model.layer}/stats"
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
        logging.info(
            f"Selected {len(selected_latents)} latents; wrote selection to {latent_selection_path}"
        )


def delphi_score(cfg, model, tokenizer, wrapped_modules):
    cfg_exp = cfg.evals.auto_interp.delphi_scoring
    # Create model-specific identifier string based on config
    model_str = f"{cfg.model.module_type.name}_k{cfg.model.k}_r{cfg.model.r}_reg{cfg.model.reg}_layer{cfg.model.layer}"

    topk_modules = [f"{name}.topk" for name, _ in wrapped_modules.items()]

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
            f"delphi_cache/{cfg.model.module_type.name}_k{cfg.model.k}_r{cfg.model.r}_reg{cfg.model.reg}_layer{cfg.model.layer}"
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
        explainer = DefaultExplainer(client, cot=True)
        # explainer = DefaultExplainer(client)
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
    if cfg_exp.use_openai_simulator:
        logging.info(
            f"  - Simulation scores: autointerp/{model_str}/scores/OpenAISimulator/"
        )
    else:
        logging.info(
            f"  - Explanations: autointerp/{model_str}/explanations/enhanced_default/"
        )
        logging.info(
            f"  - Detection scores: autointerp/{model_str}/scores/enhanced_detection/"
        )
    logging.info(f"{'=' * 60}\n")

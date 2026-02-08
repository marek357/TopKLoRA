import argparse
import heapq
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from tqdm import tqdm


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _merge_top(heap, score, pid, k):
    """Maintain a fixed-size min-heap of (score, pid)."""

    if k <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, (score, pid))
    elif score > heap[0][0]:
        heapq.heapreplace(heap, (score, pid))


def _merge_heap_dict(dest, src, k, offset: int = 0):
    for latent_idx, heap in src.items():
        dest_heap = dest[offset + latent_idx]
        for score, pid in heap:
            _merge_top(dest_heap, score, pid, k)


def _process_split(args):
    """Process a single split file; designed for executor map."""

    split, width, top_prompt_count, prompt_ids, verbose_split, verbose_activations = args
    data = load_file(split)
    locs = data["locations"]
    acts = data["activations"]

    # Split filenames are "<start>_<end>.safetensors". The latent indices inside
    # have been rebased to 0 during save_splits, so we must add `start` back to
    # recover the original hookpoint-local index.
    split_name = Path(split).stem  # e.g. "2048_4095"
    split_start = int(split_name.split("_")[0])

    if locs.shape[0] == 0:
        return (
            np.zeros(width, dtype=np.float64),
            np.zeros(width, dtype=np.float64),
            np.zeros(width, dtype=np.int64),
            defaultdict(list),
        )

    seq = locs[:, 0].astype(np.int64)
    latent_idx = locs[:, 2].astype(np.int64) + split_start
    key = seq * width + latent_idx

    order = np.argsort(key)
    key = key[order]
    acts_sorted = acts[order]

    unique_keys, first_idx = np.unique(key, return_index=True)
    max_per_key = np.maximum.reduceat(acts_sorted, first_idx)

    seq_for_key = unique_keys // width
    latent_for_key = unique_keys % width

    sums = np.zeros(width, dtype=np.float64)
    sums_sq = np.zeros(width, dtype=np.float64)
    active_counts = np.zeros(width, dtype=np.int64)
    top_prompts = defaultdict(list)

    unique_latents = np.unique(latent_for_key)
    latent_iter = unique_latents
    if verbose_split:
        latent_iter = tqdm(
            unique_latents,
            desc=f"latents::{split.name}",
            leave=False,
            total=len(unique_latents),
        )

    act_bar = None
    if verbose_activations:
        act_bar = tqdm(
            total=len(unique_keys),
            desc=f"acts::{split.name}",
            leave=False,
        )

    for local_latent in latent_iter:
        mask = latent_for_key == local_latent
        vals = max_per_key[mask]
        seqs = seq_for_key[mask]

        if act_bar is not None:
            act_bar.update(len(vals))

        sums[local_latent] += vals.sum()
        sums_sq[local_latent] += (vals * vals).sum()
        active_counts[local_latent] += len(vals)

        if prompt_ids:
            heap = top_prompts[local_latent]
            for val, seq_idx in zip(vals.tolist(), seqs.tolist()):
                if seq_idx >= len(prompt_ids):
                    continue
                pid = prompt_ids[seq_idx]
                if pid is None:
                    continue
                _merge_top(heap, val, pid, top_prompt_count)

    if act_bar is not None:
        act_bar.close()

    return sums, sums_sq, active_counts, top_prompts


def build_offsets(raw_dir: Path):
    hookpoint_offsets = {}
    hookpoint_widths = {}
    offset = 0
    for hook_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        cfg_path = hook_dir / "config.json"
        if not cfg_path.exists():
            continue
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        width = int(cfg.get("width", 0))
        hookpoint = cfg.get("hookpoint", hook_dir.name)
        hookpoint_offsets[hookpoint] = offset
        hookpoint_widths[hookpoint] = width
        offset += width
    return hookpoint_offsets, hookpoint_widths, offset


def process_hookpoint(
    hook_dir: Path,
    offset: int,
    width: int,
    top_prompt_count: int,
    prompt_ids,
    sums,
    sums_sq,
    active_counts,
    top_prompts,
    workers: int,
    show_split_progress: bool,
    progress_updater=None,
    verbose_split: bool = False,
    verbose_activations: bool = False,
):
    # Per-split verbose bars only render reliably when running splits in the main
    # process. When using a process pool, child-process tqdms won't appear. So we
    # enable the inner tqdm only when workers <= 1.
    enable_verbose_split = verbose_split and workers <= 1
    enable_verbose_activations = verbose_activations and workers <= 1

    split_files = sorted(hook_dir.glob("*.safetensors"))
    if not split_files:
        return

    args = [
        (
            split,
            width,
            top_prompt_count,
            prompt_ids,
            enable_verbose_split,
            enable_verbose_activations,
        )
        for split in split_files
    ]

    if workers > 1:
        # Use spawn-safe executor; fork is fine on Linux but spawn avoids surprises.
        with ProcessPoolExecutor(max_workers=workers) as ex:
            iterator = ex.map(_process_split, args)
            for split, result in tqdm(
                zip(split_files, iterator),
                total=len(split_files),
                desc=f"splits::{hook_dir.name}",
                leave=False,
                disable=not show_split_progress,
            ):
                split_sums, split_sums_sq, split_counts, split_top_prompts = result
                start = offset
                end = offset + width
                sums[start:end] += split_sums
                sums_sq[start:end] += split_sums_sq
                active_counts[start:end] += split_counts
                if prompt_ids:
                    _merge_heap_dict(
                        top_prompts, split_top_prompts, top_prompt_count, offset
                    )
                if progress_updater:
                    progress_updater(1)
    else:
        for split, result in tqdm(
            zip(split_files, map(_process_split, args)),
            total=len(split_files),
            desc=f"splits::{hook_dir.name}",
            leave=False,
            disable=not show_split_progress,
        ):
            split_sums, split_sums_sq, split_counts, split_top_prompts = result
            start = offset
            end = offset + width
            sums[start:end] += split_sums
            sums_sq[start:end] += split_sums_sq
            active_counts[start:end] += split_counts
            if prompt_ids:
                _merge_heap_dict(top_prompts, split_top_prompts, top_prompt_count, offset)
            if progress_updater:
                progress_updater(1)


def recompute(
    raw_dir: Path,
    stats_dir: Path,
    top_prompt_count: int,
    workers: int,
    hookpoint_workers: int,
    verbose_split: bool,
    verbose_activations: bool,
):
    prompts = read_jsonl(stats_dir / "prompts.jsonl")
    prompt_ids = [rec.get("prompt_id") for rec in prompts]

    hook_offsets, hook_widths, total_latents = build_offsets(raw_dir)
    if total_latents == 0:
        raise ValueError(f"No hookpoints found under {raw_dir}")

    sums = np.zeros(total_latents, dtype=np.float64)
    sums_sq = np.zeros(total_latents, dtype=np.float64)
    active_counts = np.zeros(total_latents, dtype=np.int64)
    top_prompts = [[] for _ in range(total_latents)]

    hook_items = list(hook_offsets.items())
    total_splits = sum(
        len(list((raw_dir / hookpoint).glob("*.safetensors")))
        for hookpoint, _ in hook_items
    )

    with tqdm(total=total_splits, desc="All splits") as global_split_pbar:
        updater = global_split_pbar.update if total_splits > 0 else None

        if hookpoint_workers > 1:
            show_split_progress = True  # allow overlapping tqdms; may be noisy
            with ThreadPoolExecutor(max_workers=hookpoint_workers) as ex:
                futures = []
                for idx, (hookpoint, offset) in enumerate(hook_items):
                    hook_dir = raw_dir / hookpoint
                    futures.append(
                        ex.submit(
                            process_hookpoint,
                            hook_dir,
                            offset,
                            hook_widths[hookpoint],
                            top_prompt_count,
                            prompt_ids,
                            sums,
                            sums_sq,
                            active_counts,
                            top_prompts,
                            workers,
                            show_split_progress,
                            updater,
                            verbose_split,
                            verbose_activations,
                        )
                    )
                for _ in tqdm(futures, desc="Processing hookpoints"):
                    _.result()
        else:
            for hookpoint, offset in tqdm(hook_items, desc="Processing hookpoints"):
                hook_dir = raw_dir / hookpoint
                process_hookpoint(
                    hook_dir,
                    offset,
                    hook_widths[hookpoint],
                    top_prompt_count,
                    prompt_ids,
                    sums,
                    sums_sq,
                    active_counts,
                    top_prompts,
                    workers,
                    True,
                    updater,
                    verbose_split,
                    verbose_activations,
                )

    counts = max(len(prompt_ids), 1)
    latent_stats = []
    top_prompt_records = []
    for hookpoint, offset in hook_offsets.items():
        width = hook_widths[hookpoint]
        for local_idx in range(width):
            latent_id = offset + local_idx
            mean = sums[latent_id] / counts
            var = sums_sq[latent_id] / counts - mean * mean
            sigma = float(max(var, 1e-12) ** 0.5)
            p_active = active_counts[latent_id] / counts
            latent_stats.append(
                {
                    "latent_id": latent_id,
                    "mu": float(mean),
                    "sigma": sigma,
                    "p_active": float(p_active),
                    "notes": {"dead": int(p_active <= 0.0)},
                    "adapter_name": hookpoint.replace(".topk", ""),
                    "feature_idx": local_idx,
                }
            )
            heap = top_prompts[latent_id]
            if heap:
                heap.sort(key=lambda x: x[0], reverse=True)
                top_prompt_records.append(
                    {
                        "latent_id": latent_id,
                        "adapter_name": hookpoint.replace(".topk", ""),
                        "feature_idx": local_idx,
                        "prompts": [
                            {"prompt_id": pid, "score": score, "activation": score}
                            for score, pid in heap[:top_prompt_count]
                        ],
                    }
                )

    write_jsonl(stats_dir / "latent_stats.jsonl", latent_stats)
    write_jsonl(stats_dir / "top_prompts.jsonl", top_prompt_records)
    print(
        f"Wrote {len(latent_stats)} latent stats to {stats_dir / 'latent_stats.jsonl'}"
    )
    print(
        f"Wrote {len(top_prompt_records)} top-prompts records to {stats_dir / 'top_prompts.jsonl'}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Recompute latent stats from cached splits without rerunning collection."
    )
    ap.add_argument(
        "raw_dir", type=Path, help="Path to delphi_cache/... hookpoint directory"
    )
    ap.add_argument(
        "--stats-dir",
        type=Path,
        default=None,
        help="Directory containing prompts.jsonl; defaults to raw_dir/stats",
    )
    ap.add_argument(
        "--top-prompt-count",
        type=int,
        default=1000,
        help="How many top prompts to keep per latent",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        help="Parallel workers per hookpoint (use 1 to disable multiprocessing)",
    )
    ap.add_argument(
        "--hookpoint-workers",
        type=int,
        default=1,
        help="Parallel hookpoints (uses threads; set >1 to enable)",
    )
    ap.add_argument(
        "--verbose-split-progress",
        action="store_true",
        help="Show per-split latent progress bars (can be noisy with many workers)",
    )
    ap.add_argument(
        "--verbose-activations-progress",
        action="store_true",
        help="Show per-split activation progress bars (only when workers <= 1)",
    )
    args = ap.parse_args()

    raw_dir = args.raw_dir
    stats_dir = args.stats_dir or (raw_dir / "stats")
    recompute(
        raw_dir,
        stats_dir,
        args.top_prompt_count,
        args.workers,
        args.hookpoint_workers,
        args.verbose_split_progress,
        args.verbose_activations_progress,
    )


if __name__ == "__main__":
    main()

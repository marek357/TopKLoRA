import argparse
import json
from collections import defaultdict
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
):
    split_files = sorted(hook_dir.glob("*.safetensors"))
    if not split_files:
        return

    for split in tqdm(split_files, desc=f"splits::{hook_dir.name}", leave=False):
        data = load_file(split)
        locs = data["locations"]
        acts = data["activations"]

        if locs.shape[0] == 0:
            continue

        seq = locs[:, 0].astype(np.int64)
        latent_idx = locs[:, 2].astype(np.int64)
        key = seq * width + latent_idx

        order = np.argsort(key)
        key = key[order]
        acts_sorted = acts[order]

        unique_keys, first_idx = np.unique(key, return_index=True)
        max_per_key = np.maximum.reduceat(acts_sorted, first_idx)

        seq_for_key = unique_keys // width
        latent_for_key = unique_keys % width

        unique_latents = np.unique(latent_for_key)
        for local_latent in tqdm(
            unique_latents,
            desc=f"latents::{hook_dir.name}",
            leave=False,
        ):
            mask = latent_for_key == local_latent
            vals = max_per_key[mask]
            seqs = seq_for_key[mask]
            latent_global = offset + local_latent

            sums[latent_global] += vals.sum()
            sums_sq[latent_global] += (vals * vals).sum()
            active_counts[latent_global] += len(vals)

            if prompt_ids:
                heap = top_prompts[latent_global]
                for val, seq_idx in zip(vals.tolist(), seqs.tolist()):
                    if seq_idx >= len(prompt_ids):
                        continue
                    pid = prompt_ids[seq_idx]
                    if pid is None:
                        continue
                    if len(heap) < top_prompt_count:
                        heap.append((val, pid))
                    else:
                        heap.sort(key=lambda x: x[0])
                        if val > heap[0][0]:
                            heap[0] = (val, pid)


def recompute(raw_dir: Path, stats_dir: Path, top_prompt_count: int):
    prompts = read_jsonl(stats_dir / "prompts.jsonl")
    prompt_ids = [rec.get("prompt_id") for rec in prompts]

    hook_offsets, hook_widths, total_latents = build_offsets(raw_dir)
    if total_latents == 0:
        raise ValueError(f"No hookpoints found under {raw_dir}")

    sums = np.zeros(total_latents, dtype=np.float64)
    sums_sq = np.zeros(total_latents, dtype=np.float64)
    active_counts = np.zeros(total_latents, dtype=np.int64)
    top_prompts = [[] for _ in range(total_latents)]

    for hookpoint, offset in tqdm(hook_offsets.items(), desc="Processing hookpoints"):
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
    args = ap.parse_args()

    raw_dir = args.raw_dir
    stats_dir = args.stats_dir or (raw_dir / "stats")
    recompute(raw_dir, stats_dir, args.top_prompt_count)


if __name__ == "__main__":
    main()

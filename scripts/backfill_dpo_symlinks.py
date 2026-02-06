#!/usr/bin/env python3
"""Backfill human-friendly symlinks for past DPO runs.

Scans a base directory containing hashed run folders (e.g.,
models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_YYYYMMDD_hash)
and creates symlinks named like:

    dpo_r4096_k512_kf32_mlp_L18 -> google-gemma-2-2b_topk_dpo_...

Usage:
    python scripts/backfill_dpo_symlinks.py \
        --base models/dpo/google/gemma-2-2b

Notes:
- Only reads hparams.json to infer r/k/k_final/module_type/layer/experiment_name.
- Existing symlinks with the same name are overwritten.
"""
import argparse
import json
import os
import re
from pathlib import Path


def infer_module_type(target_modules):
    if not target_modules:
        return "unknown"
    has_mlp = any("mlp." in t for t in target_modules)
    has_attn = any("self_attn" in t or "attn." in t for t in target_modules)
    if has_mlp and has_attn:
        return "mlp_attn"
    if has_mlp:
        return "mlp"
    if has_attn:
        return "attn"
    return "unknown"


def infer_layer(target_modules):
    if not target_modules:
        return None
    for t in target_modules:
        m = re.search(r"layers\.(\d+)\.", t)
        if m:
            return int(m.group(1))
    return None


def build_name(hparams):
    # Try to use logged experiment_name if it already contains r/k/kf/layer info
    exp_name = hparams.get("experiment_name")

    lt = hparams.get("lora_topk", {}) or {}
    r = lt.get("r")
    k = lt.get("k")
    kf = lt.get("k_final") or k
    target_modules = lt.get("target_modules") or []
    module_type = infer_module_type(target_modules)
    layer = infer_layer(target_modules)

    # Fallbacks
    if r is None or k is None:
        return exp_name or None
    if layer is None:
        layer = "?"

    return f"dpo_r{r}_k{k}_kf{kf}_{module_type}_L{layer}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base dir with hashed run folders")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    if not base.is_dir():
        print(f"Base path not found: {base}")
        return

    created = 0
    skipped = 0

    for child in sorted(base.iterdir()):
        if child.name == "latest":
            continue
        # Skip existing symlinks so we only treat real run directories as sources
        if child.is_symlink():
            continue
        if not child.is_dir():
            continue
        hparams_path = child / "hparams.json"
        if not hparams_path.exists():
            continue
        try:
            hparams = json.loads(hparams_path.read_text())
        except Exception as e:
            print(f"[skip] {child.name}: failed to read hparams.json ({e})")
            continue

        name = build_name(hparams)
        if not name:
            print(f"[skip] {child.name}: could not infer name")
            continue

        link_path = base / name
        if link_path.exists() or link_path.is_symlink():
            try:
                link_path.unlink()
            except Exception as e:
                print(f"[warn] could not remove existing {link_path}: {e}")
        try:
            link_path.symlink_to(child, target_is_directory=True)
            created += 1
        except Exception as e:
            print(f"[skip] {child.name}: symlink failed ({e})")
            skipped += 1

    print(f"Done. Created {created} symlinks, skipped {skipped}.")


if __name__ == "__main__":
    main()

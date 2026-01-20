#!/usr/bin/env python3
"""Build human-readable symlinks for trained adapters.

Finds every `final_adapter` under a source root (defaults to
`models/dpo/google/gemma-2-2b`) and creates a symlink at a structured path:
`{dest_root}/{module_type}/k{k}/r{r}/layer{layer}` pointing to that directory.
This can be rerun safely; existing correct links are left untouched.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_SOURCE = "models/dpo/google/gemma-2-2b"
DEFAULT_DEST = "models/dpo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE,
        help="Root directory containing hashed run folders (searches recursively for final_adapter).",
    )
    parser.add_argument(
        "--dest-root",
        default=DEFAULT_DEST,
        help="Base directory where structured symlinks will be created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned links without creating them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing mismatched symlinks. Existing real directories are never removed automatically.",
    )
    parser.add_argument(
        "--strategy",
        choices=["newest", "oldest"],
        default="newest",
        help="When multiple runs map to the same target path, pick the newest (default) or oldest run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def find_final_adapters(source_root: Path) -> Iterable[Tuple[Path, Path]]:
    seen = set()
    for final_dir in source_root.rglob("final_adapter"):
        if not final_dir.is_dir():
            continue
        run_dir = final_dir.parent.resolve()
        if run_dir in seen:
            continue
        seen.add(run_dir)
        yield run_dir, final_dir.resolve()


def load_hparams(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "hparams.json"
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logging.warning("Skipping %s (missing hparams.json)", run_dir)
        return None
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to read %s: %s", path, exc)
        return None


def infer_layer(target_modules: Sequence[str], fallback: Optional[int]) -> Optional[int]:
    for module in target_modules:
        match = re.search(r"layers\.(\d+)", module)
        if match:
            return int(match.group(1))
    return fallback


def infer_module_type(target_modules: Sequence[str]) -> Optional[str]:
    names = set()
    for module in target_modules:
        match = re.search(r"layers\.\d+\.([^.]+)", module)
        if match:
            part = match.group(1)
            names.add({"self_attn": "attn", "mlp": "mlp"}.get(part, part))
    if not names:
        return None
    if len(names) == 1:
        return names.pop()
    return "_".join(sorted(names))


def extract_adapter_meta(hparams: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lora_cfg = hparams.get("lora_topk") or {}
    r = lora_cfg.get("r")
    k = lora_cfg.get("k_final", lora_cfg.get("k"))
    targets = lora_cfg.get("target_modules") or []
    layer = infer_layer(targets, lora_cfg.get("layer"))
    module_type = infer_module_type(targets)

    if r is None or k is None or layer is None or module_type is None:
        logging.warning(
            "Missing metadata (r=%s, k=%s, layer=%s, module_type=%s)", r, k, layer, module_type
        )
        return None

    return {
        "r": int(r),
        "k": int(k),
        "layer": int(layer),
        "module_type": str(module_type),
    }


def build_dest_path(meta: Dict[str, Any], dest_root: Path) -> Path:
    return dest_root / meta["module_type"] / f"k{meta['k']}" / f"r{meta['r']}" / f"layer{meta['layer']}"


def ensure_symlink(src: Path, dest: Path, *, dry_run: bool, force: bool) -> str:
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink():
            if dest.resolve() == src.resolve():
                logging.info("Already linked: %s -> %s", dest, src)
                return "kept"
            if force:
                if not dry_run:
                    dest.unlink()
            else:
                logging.warning(
                    "Destination exists and differs: %s (use --force to replace symlinks)", dest)
                return "skipped"
        else:
            if dest.is_dir() and not any(dest.iterdir()) and force:
                if not dry_run:
                    dest.rmdir()
            else:
                logging.warning(
                    "Destination exists and is not a symlink: %s", dest)
                return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        try:
            os.symlink(src, dest)
        except FileExistsError:
            return "skipped"
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to link %s -> %s: %s", dest, src, exc)
            return "error"
    logging.info("Linked: %s -> %s", dest, src)
    return "created"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()

    created = kept = skipped = errors = 0

    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")

    grouped: Dict[Path, List[Dict[str, Any]]] = {}
    for run_dir, final_dir in find_final_adapters(source_root):
        hparams = load_hparams(run_dir)
        if not hparams:
            errors += 1
            continue

        meta = extract_adapter_meta(hparams)
        if not meta:
            errors += 1
            continue

        dest = build_dest_path(meta, dest_root)
        grouped.setdefault(dest, []).append(
            {
                "run_dir": run_dir,
                "final_dir": final_dir,
                "mtime": final_dir.stat().st_mtime,
            }
        )

    for dest, cands in grouped.items():
        if not cands:
            continue

        if args.strategy == "newest":
            chosen = max(cands, key=lambda c: c["mtime"])
        else:
            chosen = min(cands, key=lambda c: c["mtime"])

        if len(cands) > 1:
            logging.info(
                "Multiple runs map to %s; picking %s via %s strategy",
                dest,
                chosen["run_dir"].name,
                args.strategy,
            )

        result = ensure_symlink(
            chosen["final_dir"], dest, dry_run=args.dry_run, force=args.force
        )
        if result == "created":
            created += 1
        elif result == "kept":
            kept += 1
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1

    logging.info(
        "Done. created=%d kept=%d skipped=%d errors=%d (dry_run=%s)",
        created,
        kept,
        skipped,
        errors,
        args.dry_run,
    )


if __name__ == "__main__":
    main()

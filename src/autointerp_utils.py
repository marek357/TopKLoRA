"""Utility functions for autointerp framework."""

from typing import Any, Dict, Iterable, List, Tuple
import os
import json

from src.models import TopKLoRALinearSTE


def _ensure_dir(path: str) -> None:
    """Create the directory and parents if they do not already exist."""
    os.makedirs(path, exist_ok=True)


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


def build_latent_index(
    modules: Dict[str, TopKLoRALinearSTE],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Assign a global latent_id to each adapter feature index."""
    latent_index = []
    adapter_offset = {}
    latent_id = 0
    for name, module in modules.items():
        adapter_offset[name] = latent_id
        for idx in range(module.r):
            latent_index.append(
                {
                    "latent_id": latent_id,
                    "adapter_name": name,
                    "feature_idx": idx,
                }
            )
            latent_id += 1
    return latent_index, adapter_offset

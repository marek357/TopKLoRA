#!/usr/bin/env python3
"""Plot histogram of p_active across all features from latent_stats.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _collect_p_active(records: Iterable[dict]) -> List[float]:
    values: List[float] = []
    for rec in records:
        if "p_active" in rec:
            values.append(float(rec["p_active"]))
    return values


def _dead_proportion(records: Iterable[dict]) -> float:
    total = 0
    dead = 0
    for rec in records:
        if "p_active" not in rec:
            continue
        total += 1
        if "notes" in rec and isinstance(rec["notes"], dict):
            is_dead = bool(rec["notes"].get("dead", 0))
        else:
            is_dead = float(rec.get("p_active", 0.0)) <= 0.0
        dead += int(is_dead)
    if total == 0:
        return 0.0
    return dead / total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histogram of p_active across all features."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to latent_stats.jsonl",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (png/pdf/svg). If omitted, shows plot.",
    )
    args = parser.parse_args()

    records = list(_read_jsonl(args.input))
    values = _collect_p_active(records)
    dead_prop = _dead_proportion(records)

    if not values:
        raise SystemExit("No p_active values found in input file.")

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=args.bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    plt.xlabel("p_active")
    plt.ylabel("Count")
    plt.title("p_active Histogram")
    plt.text(
        0.98,
        0.95,
        f"dead proportion = {dead_prop:.2%}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

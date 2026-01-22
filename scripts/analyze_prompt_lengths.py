#!/usr/bin/env python3
"""Analyze prompt length statistics from a prompts.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import List


def _read_texts(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            continuation = record.get("continuation")
            if not isinstance(prompt, str):
                continue
            if isinstance(continuation, str) and continuation:
                texts.append(f"{prompt}\n\n{continuation}")
            else:
                texts.append(prompt)
    return texts


def _percentile(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _summary_stats(lengths: List[int]) -> dict:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": float(mean(lengths)),
        "median": float(median(lengths)),
        "p90": _percentile(lengths, 90),
        "p95": _percentile(lengths, 95),
    }


def _print_stats(label: str, stats: dict) -> None:
    print(f"{label}:")
    print(f"  count:  {stats['count']}")
    print(f"  min:    {stats['min']}")
    print(f"  max:    {stats['max']}")
    print(f"  mean:   {stats['mean']:.2f}")
    print(f"  median: {stats['median']:.2f}")
    print(f"  p90:    {stats['p90']:.2f}")
    print(f"  p95:    {stats['p95']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze prompt lengths from a prompts.jsonl file, including continuation when available."
        )
    )
    parser.add_argument(
        "prompts_path",
        type=Path,
        help="Path to prompts.jsonl",
    )
    args = parser.parse_args()

    texts = _read_texts(args.prompts_path)
    char_lengths = [len(text) for text in texts]
    word_lengths = [len(text.split()) for text in texts]

    _print_stats("Character length", _summary_stats(char_lengths))
    _print_stats("Word length", _summary_stats(word_lengths))


if __name__ == "__main__":
    main()

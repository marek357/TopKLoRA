"""
Script to analyze conversation datasets.
Supports:
- lmsys/lmsys-chat-1m: Counts English samples that are not redacted
- Anthropic/hh-rlhf: Analyzes chosen/rejected conversations

Computes token statistics using the gemma-2-2b-it tokenizer.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.utils import hh_string_to_messages


def analyze_lmsys(tokenizer):
    """Analyze the lmsys/lmsys-chat-1m dataset."""
    print("Loading lmsys/lmsys-chat-1m dataset...")
    dataset = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
    )

    total_samples = len(dataset)
    print(f"Total samples: {total_samples:,}")

    # Print first two examples after applying chat template
    print("\n" + "=" * 70)
    print("FIRST TWO EXAMPLES (after applying chat template)")
    print("=" * 70)
    for i in range(2):
        sample = dataset[i]
        conversation = sample.get("conversation", [])
        print(f"\n--- Example {i + 1} ---")
        print(f"Language: {sample.get('language')}")
        print(f"Redacted: {sample.get('redacted')}")
        print(f"Model: {sample.get('model')}")
        print(f"\nRaw conversation:\n{conversation}")
        try:
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            print(f"\nAfter chat template:\n{text}")
        except Exception as e:
            print(f"\nFailed to apply chat template: {e}")
        print("-" * 70)

    # Count English and non-redacted samples
    english_count = 0
    non_redacted_count = 0
    english_non_redacted_count = 0

    # Token statistics
    all_token_lengths = []
    english_non_redacted_token_lengths = []

    print("Analyzing samples...")
    for sample in tqdm(dataset, desc="Processing"):
        is_english = sample.get("language") == "English"
        is_not_redacted = not sample.get("redacted", False)

        if is_english:
            english_count += 1
        if is_not_redacted:
            non_redacted_count += 1

        # Apply chat template and tokenize
        conversation = sample.get("conversation", [])
        try:
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(text, add_special_tokens=False)
            num_tokens = len(tokens)
        except Exception:
            # Skip samples that fail to process
            num_tokens = 0

        all_token_lengths.append(num_tokens)

        if is_english and is_not_redacted:
            english_non_redacted_count += 1
            english_non_redacted_token_lengths.append(num_tokens)

    # Calculate ratios
    english_ratio = english_count / total_samples
    non_redacted_ratio = non_redacted_count / total_samples
    english_non_redacted_ratio = english_non_redacted_count / total_samples

    # Calculate token statistics for all samples
    all_tokens_arr = np.array(all_token_lengths)
    total_tokens_all = np.sum(all_tokens_arr)
    mean_tokens_all = np.mean(all_tokens_arr)
    std_tokens_all = np.std(all_tokens_arr)
    median_tokens_all = np.median(all_tokens_arr)
    min_tokens_all = np.min(all_tokens_arr)
    max_tokens_all = np.max(all_tokens_arr)

    # Calculate token statistics for English non-redacted samples
    en_nr_tokens_arr = np.array(english_non_redacted_token_lengths)
    total_tokens_en_nr = np.sum(en_nr_tokens_arr)
    mean_tokens_en_nr = np.mean(en_nr_tokens_arr)
    std_tokens_en_nr = np.std(en_nr_tokens_arr)
    median_tokens_en_nr = np.median(en_nr_tokens_arr)
    min_tokens_en_nr = np.min(en_nr_tokens_arr)
    max_tokens_en_nr = np.max(en_nr_tokens_arr)

    # Calculate percentiles for distribution
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    all_percentile_values = np.percentile(all_tokens_arr, percentiles)
    en_nr_percentile_values = np.percentile(en_nr_tokens_arr, percentiles)

    # Print results
    print("\n" + "=" * 70)
    print("SAMPLE COUNTS")
    print("=" * 70)
    print(f"Total samples:                    {total_samples:>12,}")
    print(
        f"English samples:                  {english_count:>12,} ({english_ratio:.2%})"
    )
    print(
        f"Non-redacted samples:             {non_redacted_count:>12,} ({non_redacted_ratio:.2%})"
    )
    print(
        f"English AND non-redacted:         {english_non_redacted_count:>12,} ({english_non_redacted_ratio:.2%})"
    )

    print("\n" + "=" * 70)
    print("TOKEN STATISTICS (ALL SAMPLES)")
    print("=" * 70)
    print(f"Total tokens:                     {total_tokens_all:>15,}")
    print(f"Mean tokens per sample:           {mean_tokens_all:>15,.1f}")
    print(f"Std dev:                          {std_tokens_all:>15,.1f}")
    print(f"Median tokens per sample:         {median_tokens_all:>15,.1f}")
    print(f"Min tokens:                       {min_tokens_all:>15,}")
    print(f"Max tokens:                       {max_tokens_all:>15,}")

    print("\n" + "=" * 70)
    print("TOKEN STATISTICS (ENGLISH & NON-REDACTED ONLY)")
    print("=" * 70)
    print(f"Total tokens:                     {total_tokens_en_nr:>15,}")
    print(f"Mean tokens per sample:           {mean_tokens_en_nr:>15,.1f}")
    print(f"Std dev:                          {std_tokens_en_nr:>15,.1f}")
    print(f"Median tokens per sample:         {median_tokens_en_nr:>15,.1f}")
    print(f"Min tokens:                       {min_tokens_en_nr:>15,}")
    print(f"Max tokens:                       {max_tokens_en_nr:>15,}")

    print("\n" + "=" * 70)
    print("TOKEN LENGTH DISTRIBUTION (PERCENTILES)")
    print("=" * 70)
    print(f"{'Percentile':<12} {'All Samples':>15} {'English Non-Redacted':>22}")
    print("-" * 70)
    for p, val_all, val_en_nr in zip(
        percentiles, all_percentile_values, en_nr_percentile_values
    ):
        print(f"P{p:<11} {val_all:>15,.0f} {val_en_nr:>22,.0f}")
    print("=" * 70)


def analyze_hh_rlhf(tokenizer, split="train"):
    """Analyze the Anthropic/hh-rlhf dataset."""
    print(f"Loading Anthropic/hh-rlhf dataset (split={split})...")
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        split=split,
    )

    total_samples = len(dataset)
    print(f"Total samples: {total_samples:,}")

    # Print first two examples
    print("\n" + "=" * 70)
    print("FIRST TWO EXAMPLES")
    print("=" * 70)
    for i in range(min(2, total_samples)):
        sample = dataset[i]
        print(f"\n--- Example {i + 1} ---")

        for key in ["chosen", "rejected"]:
            if key in sample and sample[key]:
                try:
                    msgs = hh_string_to_messages(sample[key])
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    )
                    print(f"\n{key.upper()} (after apply_chat_template):")
                    print(text)
                except Exception as e:
                    print(f"\n{key.upper()}: Failed to process: {e}")
        print("-" * 70)

    # Token statistics
    chosen_token_lengths = []
    rejected_token_lengths = []
    chosen_msg_counts = []
    rejected_msg_counts = []
    parse_failures = 0

    print("Analyzing samples...")
    for sample in tqdm(dataset, desc="Processing"):
        for key, token_list, msg_list in [
            ("chosen", chosen_token_lengths, chosen_msg_counts),
            ("rejected", rejected_token_lengths, rejected_msg_counts),
        ]:
            if key not in sample or not sample[key]:
                continue

            try:
                msgs = hh_string_to_messages(sample[key])
                msg_list.append(len(msgs))

                # Apply chat template and tokenize
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_list.append(len(tokens))
            except Exception:
                parse_failures += 1
                token_list.append(0)
                msg_list.append(0)

    if parse_failures > 0:
        print(f"\nWarning: {parse_failures} parse failures")

    # Print results
    print("\n" + "=" * 70)
    print("SAMPLE COUNTS")
    print("=" * 70)
    print(f"Total samples:                    {total_samples:>12,}")
    print(f"Chosen conversations:             {len(chosen_token_lengths):>12,}")
    print(f"Rejected conversations:           {len(rejected_token_lengths):>12,}")

    for name, token_lengths, msg_counts in [
        ("CHOSEN", chosen_token_lengths, chosen_msg_counts),
        ("REJECTED", rejected_token_lengths, rejected_msg_counts),
    ]:
        if not token_lengths:
            continue

        tokens_arr = np.array(token_lengths)
        msgs_arr = np.array(msg_counts)

        print(f"\n{'=' * 70}")
        print(f"TOKEN STATISTICS ({name} CONVERSATIONS)")
        print("=" * 70)
        print(f"Total tokens:                     {np.sum(tokens_arr):>15,}")
        print(f"Mean tokens per sample:           {np.mean(tokens_arr):>15,.1f}")
        print(f"Std dev:                          {np.std(tokens_arr):>15,.1f}")
        print(f"Median tokens per sample:         {np.median(tokens_arr):>15,.1f}")
        print(f"Min tokens:                       {np.min(tokens_arr):>15,}")
        print(f"Max tokens:                       {np.max(tokens_arr):>15,}")

        print(f"\nMESSAGE COUNTS ({name}):")
        print(f"Mean messages per sample:         {np.mean(msgs_arr):>15,.1f}")
        print(f"Median messages per sample:       {np.median(msgs_arr):>15,.1f}")
        print(f"Min messages:                     {np.min(msgs_arr):>15,}")
        print(f"Max messages:                     {np.max(msgs_arr):>15,}")

    # Combined percentiles
    all_tokens = chosen_token_lengths + rejected_token_lengths
    if all_tokens:
        all_tokens_arr = np.array(all_tokens)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(all_tokens_arr, percentiles)

        print(f"\n{'=' * 70}")
        print("TOKEN LENGTH DISTRIBUTION (ALL CONVERSATIONS)")
        print("=" * 70)
        print(f"{'Percentile':<12} {'Token Count':>15}")
        print("-" * 30)
        for p, val in zip(percentiles, percentile_values):
            print(f"P{p:<11} {val:>15,.0f}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze conversation datasets for token statistics."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["lmsys", "hh-rlhf"],
        default="lmsys",
        help="Dataset to analyze: 'lmsys' for lmsys/lmsys-chat-1m, 'hh-rlhf' for Anthropic/hh-rlhf",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/gemma-2-2b-it",
        help="Tokenizer to use for token counting (default: google/gemma-2-2b-it)",
    )
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading {args.tokenizer} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dataset == "lmsys":
        analyze_lmsys(tokenizer)
    elif args.dataset == "hh-rlhf":
        analyze_hh_rlhf(tokenizer, split=args.split)


if __name__ == "__main__":
    main()

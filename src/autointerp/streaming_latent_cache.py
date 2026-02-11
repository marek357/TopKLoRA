import gc
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from delphi.latents.cache import LatentCache
from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedModel


class StreamingLatentCache(LatentCache):
    """A drop-in LatentCache variant that avoids giant concatenations.

    When ``streaming=True`` it keeps per-batch buffers and writes splits
    directly from those buffers, bypassing the huge ``torch.cat`` in the
    upstream implementation that can OOM on very large runs.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        hookpoint_to_sparse_encode: dict[str, Callable],
        batch_size: int,
        transcode: bool = False,
        filters: dict[str, Float[Tensor, "indices"]] | None = None,
        log_path: Path | None = None,
        streaming: bool = False,
    ):
        super().__init__(
            model=model,
            hookpoint_to_sparse_encode=hookpoint_to_sparse_encode,
            batch_size=batch_size,
            transcode=transcode,
            filters=filters,
            log_path=log_path,
        )
        self.streaming = streaming

    def run(self, n_tokens: int, tokens: Tensor):
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)
        if total_batches == 0:
            logging.info("No token batches to process; skipping caching")
            return

        tokens_per_batch = token_batches[0].numel()
        from delphi.latents.collect_activations import collect_activations

        with torch.no_grad():
            from tqdm import tqdm

            with tqdm(total=total_batches, desc="Caching latents") as pbar:
                for batch_number, batch in enumerate(token_batches):
                    total_tokens += tokens_per_batch

                    with collect_activations(
                        self.model,
                        list(self.hookpoint_to_sparse_encode.keys()),
                        self.transcode,
                    ) as activations:
                        self.model(batch.to(self.model.device))

                        for hookpoint, latents in activations.items():
                            sae_latents = self.hookpoint_to_sparse_encode[hookpoint](
                                latents
                            )
                            self.cache.add(sae_latents, batch, batch_number, hookpoint)
                            firing_counts = (sae_latents > 0).sum((0, 1))
                            if self.width is None:
                                self.width = sae_latents.shape[2]

                            if hookpoint not in self.hookpoint_firing_counts:
                                self.hookpoint_firing_counts[hookpoint] = (
                                    firing_counts.cpu()
                                )
                            else:
                                self.hookpoint_firing_counts[hookpoint] += (
                                    firing_counts.cpu()
                                )

                    pbar.update(1)
                    pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        logging.info(f"Total tokens processed: {total_tokens:,}")
        if not self.streaming:
            # Original behavior: materialize single large tensors
            self.cache.save()
        self.save_firing_counts()

    def save_splits(
        self,
        n_splits: int,
        save_dir: Path,
        save_tokens: bool = True,
        max_split_bytes: int = 2 * 1024**3,  # 2 GiB target per split file
    ):
        if not self.streaming:
            return super().save_splits(
                n_splits=n_splits, save_dir=save_dir, save_tokens=save_tokens
            )

        assert self.width is not None, "Width must be set before saving splits"

        from safetensors.numpy import save_file

        for module_path in list(self.cache.latent_locations_batches.keys()):
            locations_batches = self.cache.latent_locations_batches[module_path]
            activations_batches = self.cache.latent_activations_batches[module_path]
            tokens_batches = self.cache.tokens_batches[module_path]

            module_dir = save_dir / module_path
            module_dir.mkdir(parents=True, exist_ok=True)

            # Write tokens once to a shared file instead of duplicating in
            # every split.  Concatenate one batch at a time to cap peak memory.
            if save_tokens and tokens_batches:
                tokens_file = module_dir / "tokens.safetensors"
                token_arrays = [b.cpu().numpy() for b in tokens_batches]
                tokens_np = np.concatenate(token_arrays, axis=0)
                save_file({"tokens": tokens_np}, tokens_file)
                del token_arrays, tokens_np
                gc.collect()

            # Compute total number of examples (rows across all batches)
            total_examples = sum(loc.shape[0] for loc in locations_batches)
            if total_examples == 0:
                continue

            # Estimate bytes per row to auto-compute a safe number of splits.
            # locations: typically int64 × ncols;  activations: typically float32 × 1
            sample_loc = locations_batches[0]
            sample_act = activations_batches[0]
            loc_row_bytes = (
                sample_loc.element_size() * sample_loc.shape[1]
                if sample_loc.dim() > 1
                else sample_loc.element_size()
            )
            act_row_bytes = (
                sample_act.element_size() * sample_act.shape[1]
                if sample_act.dim() > 1
                else sample_act.element_size()
            )
            bytes_per_row = loc_row_bytes + act_row_bytes
            # safetensors _tobytes() needs ~2× the array size (array + serialized copy)
            effective_bytes_per_row = bytes_per_row * 2

            total_bytes_est = total_examples * effective_bytes_per_row
            min_splits_for_memory = max(
                1, int(np.ceil(total_bytes_est / max_split_bytes))
            )
            actual_splits = max(n_splits, min_splits_for_memory)

            if actual_splits > n_splits:
                logging.info(
                    f"  {module_path}: auto-increased splits from {n_splits} to "
                    f"{actual_splits} ({total_examples:,} rows × {bytes_per_row} B/row "
                    f"≈ {total_bytes_est / 1024**3:.1f} GiB, target {max_split_bytes / 1024**3:.1f} GiB/split)"
                )

            examples_per_split = max(1, total_examples // actual_splits)

            split_num = 0
            cumulative_examples = 0
            split_loc_parts: list[np.ndarray] = []
            split_act_parts: list[np.ndarray] = []

            def _flush_split():
                nonlocal split_num, split_loc_parts, split_act_parts
                if not split_loc_parts:
                    return
                masked_locations = np.concatenate(split_loc_parts, axis=0)
                masked_activations = np.concatenate(split_act_parts, axis=0)
                split_loc_parts.clear()
                split_act_parts.clear()

                output_file = module_dir / f"split_{split_num:04d}.safetensors"
                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }
                save_file(split_data, output_file)
                del masked_locations, masked_activations, split_data
                gc.collect()
                split_num += 1

            for loc_batch, act_batch in zip(locations_batches, activations_batches):
                batch_size = loc_batch.shape[0]
                batch_start = 0

                while batch_start < batch_size:
                    # How many examples can we take from this batch before hitting the limit?
                    space_in_split = examples_per_split - (
                        cumulative_examples % examples_per_split
                    )
                    batch_end = min(batch_start + space_in_split, batch_size)

                    split_loc_parts.append(
                        loc_batch[batch_start:batch_end].cpu().numpy()
                    )
                    split_act_parts.append(
                        act_batch[batch_start:batch_end].cpu().numpy()
                    )

                    cumulative_examples += batch_end - batch_start
                    batch_start = batch_end

                    # If we've hit the split size, save and reset
                    if (
                        cumulative_examples % examples_per_split == 0
                        and split_loc_parts
                    ):
                        _flush_split()

            # Save any remaining examples
            _flush_split()


def make_latent_cache(
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    batch_size: int,
    transcode: bool = False,
    filters: dict[str, Float[Tensor, "indices"]] | None = None,
    log_path: Path | None = None,
    streaming: bool = False,
) -> LatentCache:
    """Factory that returns a streaming-aware cache."""
    return StreamingLatentCache(
        model=model,
        hookpoint_to_sparse_encode=hookpoint_to_sparse_encode,
        batch_size=batch_size,
        transcode=transcode,
        filters=filters,
        log_path=log_path,
        streaming=streaming,
    )

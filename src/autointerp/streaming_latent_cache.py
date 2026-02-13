import gc
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from delphi.latents.cache import LatentCache
from delphi.latents.collect_activations import collect_activations

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
        filters: dict[str, Float[Tensor, "indices"]] | None = None,  # noqa: F821
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

        with torch.no_grad():
            with tqdm(total=total_batches, desc="Caching latents") as pbar:
                for batch_number, batch in enumerate(token_batches):
                    total_tokens += batch.numel()

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
    ):
        if not self.streaming:
            return super().save_splits(
                n_splits=n_splits, save_dir=save_dir, save_tokens=save_tokens
            )

        assert self.width is not None, "Width must be set before saving splits"

        from safetensors.numpy import save_file

        split_indices = self._generate_split_indices(n_splits)

        for module_path in list(self.cache.latent_locations_batches.keys()):
            loc_batches = self.cache.latent_locations_batches[module_path]
            act_batches = self.cache.latent_activations_batches[module_path]
            tok_batches = self.cache.tokens_batches[module_path]

            module_dir = save_dir / module_path
            module_dir.mkdir(parents=True, exist_ok=True)

            # Concatenate tokens once (small relative to locations/activations)
            tokens_np = None
            if save_tokens and tok_batches:
                tokens_np = np.concatenate(
                    [b.cpu().numpy() for b in tok_batches], axis=0
                )

            # One pass per latent range — only the matching rows are
            # concatenated, keeping peak memory at ~data_size/n_splits.
            for start, end in split_indices:
                start_int, end_int = start.item(), end.item()
                parts_loc: list[np.ndarray] = []
                parts_act: list[np.ndarray] = []

                for loc_batch, act_batch in zip(loc_batches, act_batches):
                    mask = (loc_batch[:, 2] >= start_int) & (loc_batch[:, 2] <= end_int)
                    if mask.any():
                        parts_loc.append(loc_batch[mask].cpu().numpy())
                        parts_act.append(act_batch[mask].cpu().numpy())

                if parts_loc:
                    masked_locations = np.concatenate(parts_loc, axis=0)
                    masked_activations = np.concatenate(parts_act, axis=0).astype(
                        np.float16
                    )
                else:
                    masked_locations = np.empty((0, 3), dtype=np.uint16)
                    masked_activations = np.empty((0,), dtype=np.float16)

                del parts_loc, parts_act

                # Rebase latent index (column 2) relative to split start
                masked_locations[:, 2] = masked_locations[:, 2] - start_int

                # Dtype optimization to reduce file size (matches parent class)
                if masked_locations.shape[0] > 0:
                    if (
                        masked_locations[:, 2].max() < 2**16
                        and masked_locations[:, 0].max() < 2**16
                    ):
                        masked_locations = masked_locations.astype(np.uint16)
                    else:
                        masked_locations = masked_locations.astype(np.uint32)
                        logging.warning(
                            "Increasing the number of splits might reduce the "
                            "memory usage of the cache."
                        )
                else:
                    masked_locations = masked_locations.astype(np.uint16)

                split_data: dict[str, np.ndarray] = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }
                if save_tokens and tokens_np is not None:
                    split_data["tokens"] = tokens_np

                save_file(split_data, module_dir / f"{start_int}_{end_int}.safetensors")
                del masked_locations, masked_activations, split_data
                gc.collect()

            del tokens_np
            gc.collect()


def make_latent_cache(
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    batch_size: int,
    transcode: bool = False,
    filters: dict[str, Float[Tensor, "indices"]] | None = None,  # noqa: F821
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

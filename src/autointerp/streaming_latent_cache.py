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

    def save_splits(self, n_splits: int, save_dir: Path, save_tokens: bool = True):
        if not self.streaming:
            return super().save_splits(
                n_splits=n_splits, save_dir=save_dir, save_tokens=save_tokens
            )

        assert self.width is not None, "Width must be set before saving splits"
        split_indices = self._generate_split_indices(n_splits)

        for module_path in list(self.cache.latent_locations_batches.keys()):
            locations_batches = self.cache.latent_locations_batches[module_path]
            activations_batches = self.cache.latent_activations_batches[module_path]
            tokens_batches = self.cache.tokens_batches[module_path]

            tokens_np = None
            if save_tokens and tokens_batches:
                tokens_np = torch.cat(tokens_batches, dim=0).numpy()

            for start, end in split_indices:
                split_loc_parts = []
                split_act_parts = []

                for loc_batch, act_batch in zip(locations_batches, activations_batches):
                    latent_indices = loc_batch[:, 2]
                    mask = (latent_indices >= start) & (latent_indices <= end)
                    if mask.any():
                        split_loc = loc_batch[mask].clone()
                        split_loc[:, 2] = split_loc[:, 2] - start
                        split_act = act_batch[mask]
                        split_loc_parts.append(split_loc.cpu().numpy())
                        split_act_parts.append(split_act.cpu().numpy())

                if not split_loc_parts:
                    continue

                masked_locations = np.concatenate(split_loc_parts, axis=0)
                masked_activations = np.concatenate(split_act_parts, axis=0)

                module_dir = save_dir / module_path
                module_dir.mkdir(parents=True, exist_ok=True)
                output_file = module_dir / f"{start}_{end}.safetensors"

                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }
                if save_tokens and tokens_np is not None:
                    split_data["tokens"] = tokens_np

                from safetensors.numpy import save_file

                save_file(split_data, output_file)

            # Keep buffers so downstream stats can still read them


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

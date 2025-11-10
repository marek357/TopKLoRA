"""TopKLoRA autointerp utilities with Delphi integration support."""

from __future__ import annotations
from src.models import TopKLoRALinearSTE

import json
import logging
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from heapq import heappush, heappushpop
import asyncio

from delphi.latents import LatentCache, LatentDataset  # type: ignore
from delphi.latents.collect_activations import collect_activations  # type: ignore
from delphi.config import CacheConfig, ConstructorConfig, SamplerConfig  # type: ignore
from delphi.explainers import ContrastiveExplainer, DefaultExplainer  # type: ignore
from delphi.pipeline import Pipeline, process_wrapper  # type: ignore
from delphi.clients import Offline, OpenRouter  # type: ignore

DELPHI_AVAILABLE = True


logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_pad_token(tokenizer) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer.pad_token_id


@dataclass
class TopKLoRAFeatureActivation:
    """Container for a single feature activation instance."""

    feature_idx: int
    position: int
    activation_value: float
    tokens: List[int]
    token_strings: List[str]
    layer_name: str
    batch_idx: Optional[int] = None
    sequence_idx: Optional[int] = None
    pre_activation_tokens: Optional[List[int]] = None
    post_activation_tokens: Optional[List[int]] = None
    decoder_projection: Optional[np.ndarray] = None
    residual_stream: Optional[np.ndarray] = None


@dataclass
class TopKLoRAFeatureData:
    """Complete data for a single TopKLoRA feature."""

    feature_idx: int
    layer_name: str
    activations: List[TopKLoRAFeatureActivation]
    mean_activation: Optional[float] = None
    max_activation: Optional[float] = None
    activation_rate: Optional[float] = None
    decoder_vector: Optional[np.ndarray] = None
    typical_effects: Optional[List[str]] = None
    typical_triggers: Optional[List[str]] = None


def _pairwise_similarity(
    residual_vectors: torch.Tensor,
    decoder_vectors: torch.Tensor,
    metric: str,
    decoder_norms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute similarity between matched residual/decoder vectors."""

    if residual_vectors.numel() == 0:
        return torch.empty(0, device=residual_vectors.device)

    if metric == "dot":
        return (residual_vectors * decoder_vectors).sum(dim=1)

    if metric == "euclidean":
        return -torch.linalg.norm(residual_vectors - decoder_vectors, dim=1)

    res_norm = torch.linalg.norm(residual_vectors, dim=1)
    if decoder_norms is None:
        decoder_norms = torch.linalg.norm(decoder_vectors, dim=1)
    denom = torch.clamp(res_norm * decoder_norms, min=1e-8)
    return (residual_vectors * decoder_vectors).sum(dim=1) / denom


class TopKAccumulator:
    """Maintain top-k similarity entries per (module, feature)."""

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self._storage: Dict[Tuple[str, int],
                            List[Tuple[float, int, int, int]]] = defaultdict(list)

    def add(
        self,
        module_name: str,
        feature_idx: int,
        similarity: float,
        batch_idx: int,
        seq_idx: int,
        position: int,
    ) -> None:
        key = (module_name, feature_idx)
        entry = (similarity, batch_idx, seq_idx, position)
        heap = self._storage[key]
        if self.top_k <= 0:
            heap.append(entry)
            return
        if len(heap) < self.top_k:
            heappush(heap, entry)
            return
        if entry[0] > heap[0][0]:
            heappushpop(heap, entry)

    def export(self) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
        exported: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for key, entries in self._storage.items():
            sorted_entries = sorted(
                entries, key=lambda item: item[0], reverse=True)
            exported[key] = [
                {
                    "similarity": float(sim),
                    "batch_idx": int(batch_idx),
                    "seq_idx": int(seq_idx),
                    "position": int(position),
                }
                for sim, batch_idx, seq_idx, position in sorted_entries
            ]
        return exported

    def export_grouped(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for (module_name, feature_idx), entries in self.export().items():
            grouped[module_name].append(
                {
                    "feature_idx": int(feature_idx),
                    "measurements": entries,
                }
            )
        return grouped


class StreamingSimilarityTracker:
    """Accumulate streaming similarity results during caching."""

    def __init__(self, metric: str, threshold: float, top_k: int) -> None:
        self.metric = metric
        self.threshold = threshold
        self.accumulator = TopKAccumulator(top_k)

    @property
    def top_k(self) -> int:
        return self.accumulator.top_k

    def add_entry(
        self,
        module_name: str,
        feature_idx: int,
        similarity: float,
        batch_idx: int,
        seq_idx: int,
        position: int,
    ) -> None:
        self.accumulator.add(module_name, feature_idx,
                             similarity, batch_idx, seq_idx, position)

    def export(self) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
        return self.accumulator.export()

    def export_grouped(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.accumulator.export_grouped()

    def reset(self) -> None:
        current_top_k = self.top_k
        self.accumulator = TopKAccumulator(current_top_k)

    def save(self, path: Path) -> None:
        data = self.export_grouped()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)


class TopKLoRALatentCache(LatentCache):
    """Latent cache that records TopKLoRA activations and residuals on top of Delphi."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        wrapped_modules: Dict[str, TopKLoRALinearSTE],
        cfg: DictConfig,
        *,
        batch_size: int = 8,
        collect_decoder_projections: bool = True,
        collect_residuals: bool = True,
    ) -> None:
        if not DELPHI_AVAILABLE:
            raise RuntimeError(
                "Delphi library is required for TopKLoRALatentCache")

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.wrapped_modules = wrapped_modules
        self.collect_decoder_projections = collect_decoder_projections
        self.collect_residuals = collect_residuals
        self.cache_root = Path(cfg.evals.topk_lora_autointerp.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.pad_token_id = _ensure_pad_token(tokenizer)

        hookpoint_to_sparse_encode = {
            name: self._build_sparse_encoder(module)
            for name, module in wrapped_modules.items()
            if isinstance(module, TopKLoRALinearSTE)
        }

        super().__init__(
            model=model,
            hookpoint_to_sparse_encode=hookpoint_to_sparse_encode,
            batch_size=batch_size,
            transcode=False,
            filters=None,
            log_path=None,
        )

        flush_override = getattr(
            cfg.evals.topk_lora_autointerp, "flush_every_n_batches", 1
        )
        self.flush_every_n_batches = max(1, int(flush_override))
        self._batches_since_flush = 0

        stream_cfg = getattr(
            cfg.evals.topk_lora_autointerp, "decoder_similarity", None
        )
        self.streaming_tracker: Optional[StreamingSimilarityTracker] = None
        if stream_cfg is not None and bool(
            getattr(stream_cfg, "stream_during_collection", False)
        ):
            self.streaming_tracker = StreamingSimilarityTracker(
                metric=str(stream_cfg.similarity_metric),
                threshold=float(stream_cfg.similarity_threshold),
                top_k=int(stream_cfg.n_top_similar_positions),
            )
        self._current_batch_number: int = 0
        self._current_batch_seq_len: int = 0

        self.hook_handles: List[Any] = []
        self._reset_topk_state()
        self._setup_topk_hooks()

    def _reset_topk_state(self) -> None:
        self.encoder_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.mask_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.decoder_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.residual_buffers: Dict[str,
                                    List[torch.Tensor]] = defaultdict(list)
        self.saved_paths: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._shard_counters: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def _build_sparse_encoder(self, module: TopKLoRALinearSTE):
        def encoder(_activation: torch.Tensor) -> torch.Tensor:
            z = getattr(module, "_last_z", None)
            if z is None:
                raise RuntimeError(
                    f"TopKLoRALinearSTE module {module} did not record activations"
                )
            if module.training:
                g_soft = getattr(module, "_last_g_soft", None)
                if g_soft is None:
                    raise RuntimeError(
                        "TopKLoRALinearSTE did not cache soft gates during training"
                    )
                latents = z * g_soft
            else:
                k_val = module._current_k()
                mask = TopKLoRALatentCache._get_topk_mask(z, k_val)
                latents = z * mask
            return latents.detach().cpu()

        return encoder

    def _setup_topk_hooks(self) -> None:
        for module_name, module in self.wrapped_modules.items():
            if not isinstance(module, TopKLoRALinearSTE):
                continue

            def hook_fn(mod, _inputs, output, module_ref=module, name=module_name):
                z = getattr(module_ref, "_last_z", None)
                if z is None:
                    return None

                z_detached = z.detach()
                self.encoder_buffers[name].append(z_detached.cpu())

                k_val = module_ref._current_k()
                mask = self._get_topk_mask(z_detached, k_val)

                residual_tensor = output.detach() if isinstance(output, torch.Tensor) else None
                if residual_tensor is not None and self.streaming_tracker is not None:
                    self._update_streaming_similarity(
                        module_name=name,
                        module_ref=module_ref,
                        mask=mask,
                        residual_tensor=residual_tensor,
                    )

                self.mask_buffers[name].append(mask.cpu())

                if self.collect_decoder_projections:
                    B_weight = cast(torch.Tensor, module_ref.B_module.weight)
                    z_masked = z_detached * mask.to(z_detached.device)
                    decoder_proj = F.linear(z_masked, B_weight)
                    self.decoder_buffers[name].append(
                        decoder_proj.detach().cpu())

                if self.collect_residuals and residual_tensor is not None:
                    self.residual_buffers[name].append(residual_tensor.cpu())
                return None

            handle = module.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)

    @staticmethod
    def _get_topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.zeros_like(z)
        _, topk_indices = torch.topk(z, k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_indices, 1.0)
        return mask

    def _save_tensor_shards(
        self,
        *,
        module_name: str,
        cache_dir: Path,
        prefix: str,
        tensors: List[torch.Tensor],
    ) -> List[str]:
        if not tensors:
            return []

        shard_paths: List[str] = []
        for tensor in tensors:
            shard_idx = self._shard_counters[module_name][prefix]
            shard_path = cache_dir / f"{prefix}_{shard_idx:05d}.pt"
            torch.save(tensor, shard_path)
            shard_paths.append(str(shard_path))
            self._shard_counters[module_name][prefix] += 1

        tensors.clear()
        return shard_paths

    def _update_streaming_similarity(
        self,
        module_name: str,
        module_ref: TopKLoRALinearSTE,
        mask: torch.Tensor,
        residual_tensor: torch.Tensor,
    ) -> None:
        if self.streaming_tracker is None:
            return

        active_idx = (mask > 0).nonzero(as_tuple=False)
        if active_idx.numel() == 0:
            return

        residual_vectors = residual_tensor[
            active_idx[:, 0], active_idx[:, 1]
        ].to(dtype=torch.float32)

        decoder_weight = cast(torch.Tensor, module_ref.B_module.weight)
        decoder_vectors = (
            decoder_weight
            .to(residual_vectors.device, dtype=torch.float32)
            [:, active_idx[:, 2]]
            .T
            .contiguous()
        )

        metric = self.streaming_tracker.metric
        decoder_norms = None
        if metric == "cosine":
            decoder_norms = torch.linalg.norm(decoder_vectors, dim=1)

        similarities = _pairwise_similarity(
            residual_vectors,
            decoder_vectors,
            metric,
            decoder_norms,
        )

        keep_mask = similarities > self.streaming_tracker.threshold
        if not bool(torch.count_nonzero(keep_mask)):
            return

        similarities = similarities[keep_mask]
        selected_idx = active_idx[keep_mask]

        batch_number = self._current_batch_number
        seq_len = self._current_batch_seq_len or residual_tensor.shape[1]
        seq_len = int(seq_len)
        position_base = batch_number * seq_len

        for sim_value, idx_triplet in zip(similarities.tolist(), selected_idx.tolist()):
            batch_idx, seq_idx, feature_idx = idx_triplet
            position = position_base + seq_idx
            self.streaming_tracker.add_entry(
                module_name=module_name,
                feature_idx=int(feature_idx),
                similarity=float(sim_value),
                batch_idx=int(batch_idx),
                seq_idx=int(seq_idx),
                position=int(position),
            )

    def _flush_topk_buffers(self, *, force: bool) -> None:
        if not self.wrapped_modules:
            return

        for module_name in self.wrapped_modules.keys():
            has_data = (
                self.encoder_buffers[module_name]
                or self.mask_buffers[module_name]
                or self.decoder_buffers[module_name]
                or self.residual_buffers[module_name]
            )
            if not has_data and not force:
                continue

            cache_dir = self.cache_root / module_name
            cache_dir.mkdir(parents=True, exist_ok=True)

            encoder_paths = self._save_tensor_shards(
                module_name=module_name,
                cache_dir=cache_dir,
                prefix="encoder_activations",
                tensors=self.encoder_buffers[module_name],
            )
            if encoder_paths:
                self.saved_paths[module_name]["encoder_activations"].extend(
                    encoder_paths)

            mask_paths = self._save_tensor_shards(
                module_name=module_name,
                cache_dir=cache_dir,
                prefix="activation_masks",
                tensors=self.mask_buffers[module_name],
            )
            if mask_paths:
                self.saved_paths[module_name]["activation_masks"].extend(
                    mask_paths)

            if self.collect_decoder_projections:
                decoder_paths = self._save_tensor_shards(
                    module_name=module_name,
                    cache_dir=cache_dir,
                    prefix="decoder_projections",
                    tensors=self.decoder_buffers[module_name],
                )
                if decoder_paths:
                    self.saved_paths[module_name]["decoder_projections"].extend(
                        decoder_paths
                    )

            if self.collect_residuals:
                residual_paths = self._save_tensor_shards(
                    module_name=module_name,
                    cache_dir=cache_dir,
                    prefix="residual_streams",
                    tensors=self.residual_buffers[module_name],
                )
                if residual_paths:
                    self.saved_paths[module_name]["residual_streams"].extend(
                        residual_paths
                    )

    # type: ignore[override]
    def run(self, n_tokens: int, tokens: torch.Tensor) -> None:
        self._reset_topk_state()
        if self.streaming_tracker is not None:
            self.streaming_tracker.reset()
        token_batches = self.load_token_batches(n_tokens, tokens)
        if not token_batches:
            logger.warning(
                "No token batches generated for TopKLoRALatentCache run")
            return

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()
        hookpoints = list(self.hookpoint_to_sparse_encode.keys())
        self._batches_since_flush = 0

        with tqdm(total=total_batches, desc="Caching TopK latents") as pbar:
            for batch_number, batch in enumerate(token_batches):
                batch_cpu = batch
                batch_device = batch_cpu.to(self.model.device)
                total_tokens += tokens_per_batch
                self._current_batch_number = batch_number
                self._current_batch_seq_len = int(batch_device.shape[1])

                with torch.no_grad():
                    with collect_activations(self.model, hookpoints, self.transcode) as activations:
                        self.model(batch_device)

                    for hookpoint, activation in activations.items():
                        latent_tensor = self.hookpoint_to_sparse_encode[hookpoint](
                            activation)
                        self.cache.add(latent_tensor, batch_cpu,
                                       batch_number, hookpoint)
                        firing_counts = (latent_tensor > 0).sum((0, 1))
                        if self.width is None:
                            self.width = latent_tensor.shape[2]

                        if hookpoint not in self.hookpoint_firing_counts:
                            self.hookpoint_firing_counts[hookpoint] = firing_counts.cpu(
                            )
                        else:
                            self.hookpoint_firing_counts[hookpoint] += firing_counts.cpu()

                self._batches_since_flush += 1
                if self._batches_since_flush >= self.flush_every_n_batches:
                    self._flush_topk_buffers(force=False)
                    self._batches_since_flush = 0

                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

                if total_tokens >= n_tokens:
                    break

        logger.info(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()
        self.save_firing_counts()
        self._flush_topk_buffers(force=True)
        if self.streaming_tracker is not None:
            stream_path = self.cache_root / "streaming_decoder_similarity.json"
            self.streaming_tracker.save(stream_path)

    def save_splits(
        self,
        n_splits: int,
        save_dir: str | Path,
        *,
        save_tokens: bool = True,
        cache_config: Optional[CacheConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Emit Delphi-compatible safetensor shards for latent activations.
        super().save_splits(n_splits=n_splits,
                            save_dir=save_path, save_tokens=save_tokens)

        # Persist cache metadata if provided so LatentDataset can reconstruct tokens.
        if cache_config is not None and model_name is not None:
            super().save_config(save_path, cache_config, model_name)
            for module_path in self.cache.latent_locations.keys():
                config_file = save_path / module_path / "config.json"
                if not config_file.exists():
                    continue
                with open(config_file, "r", encoding="utf-8") as handle:
                    config_data = json.load(handle)
                if "ctx_len" not in config_data and "cache_ctx_len" in config_data:
                    config_data["ctx_len"] = config_data["cache_ctx_len"]
                config_data.setdefault(
                    "dataset_repo", cache_config.dataset_repo)
                config_data.setdefault(
                    "dataset_split", cache_config.dataset_split)
                if cache_config.dataset_name:
                    config_data.setdefault(
                        "dataset_name", cache_config.dataset_name)
                column_key = cache_config.dataset_column or config_data.get(
                    "dataset_column")
                if column_key:
                    config_data.setdefault("dataset_column", column_key)
                config_data.setdefault("batch_size", cache_config.batch_size)
                config_data.setdefault("n_tokens", cache_config.n_tokens)
                config_data.setdefault("n_splits", cache_config.n_splits)
                with open(config_file, "w", encoding="utf-8") as handle:
                    json.dump(config_data, handle, indent=2)

        # Retain auxiliary index for TopK-specific tensor dumps (activation masks, etc.).
        if self.saved_paths:
            index = {
                module: {prefix: paths for prefix, paths in paths_dict.items()}
                for module, paths_dict in self.saved_paths.items()
            }
            with open(save_path / "cache_index.json", "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=2)

    def cleanup(self) -> None:
        self._flush_topk_buffers(force=True)
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


class TopKLoRALatentDataset(LatentDataset):
    """LatentDataset variant that prefers cached tokens over dataset reloading."""

    def load_tokens(self):  # type: ignore[override]
        cached = getattr(self, "tokens", None)
        if cached is not None:
            return cached
        for buffer in getattr(self, "buffers", []):
            tokens = buffer.tokens
            if tokens is not None:
                self.tokens = tokens
                return tokens
        return super().load_tokens()


class DecoderSimilarityAnalyzer:
    """Compute decoder-residual similarities for TopKLoRA features."""

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        tokenizer,
        wrapped_modules: Dict[str, TopKLoRALinearSTE],
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.wrapped_modules = wrapped_modules
        self.cache_root = Path(cfg.evals.topk_lora_autointerp.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.decoder_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.module_decoder_matrices: Dict[str, torch.Tensor] = {}
        self.module_decoder_norms: Dict[str, torch.Tensor] = {}
        self.similarity_measurements: Dict[Tuple[str,
                                                 int], List[Dict[str, Any]]] = {}
        self.streaming_similarity_path = (
            self.cache_root / "streaming_decoder_similarity.json"
        )
        self.streaming_similarity_measurements: Dict[
            Tuple[str, int], List[Dict[str, Any]]
        ] = {}
        decoder_cfg = cfg.evals.topk_lora_autointerp.decoder_similarity
        self.prefer_streaming_results = bool(
            getattr(decoder_cfg, "prefer_streaming_results", True)
        )
        self._cache_decoder_vectors()
        self._load_streaming_similarity()

    def _discover_shards(self, module_name: str, prefix: str) -> List[Path]:
        cache_dir = self.cache_root / module_name
        shard_paths = sorted(cache_dir.glob(f"{prefix}_*.pt"))
        if not shard_paths:
            legacy_path = cache_dir / f"{prefix}.pt"
            if legacy_path.exists():
                shard_paths = [legacy_path]
        return shard_paths

    def _cache_decoder_vectors(self) -> None:
        logger.info("Caching decoder vectors from LoRA B matrices...")
        for layer_name, module in self.wrapped_modules.items():
            if not isinstance(module, TopKLoRALinearSTE):
                continue
            B_weight = cast(torch.Tensor, module.B_module.weight)
            weight = B_weight.detach().to(dtype=torch.float32)
            weight_cpu = weight.cpu()
            self.module_decoder_matrices[layer_name] = weight_cpu
            self.module_decoder_norms[layer_name] = torch.linalg.norm(
                weight_cpu, dim=0
            )
            for feature_idx in range(weight_cpu.shape[1]):
                self.decoder_cache[(layer_name, feature_idx)] = weight_cpu[
                    :, feature_idx
                ].clone()
        logger.info("Cached %d decoder vectors", len(self.decoder_cache))

    def _load_streaming_similarity(self) -> None:
        if not self.streaming_similarity_path.exists():
            return
        try:
            with open(self.streaming_similarity_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to read streaming similarity results: %s", exc)
            return

        loaded: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for module_name, feature_entries in data.items():
            if not isinstance(feature_entries, list):
                continue
            for entry in feature_entries:
                feature_idx = entry.get("feature_idx")
                measurements = entry.get("measurements", [])
                if feature_idx is None or not isinstance(measurements, list):
                    continue
                loaded[(module_name, int(feature_idx))] = [
                    {
                        "similarity": float(item.get("similarity", 0.0)),
                        "batch_idx": int(item.get("batch_idx", 0)),
                        "seq_idx": int(item.get("seq_idx", 0)),
                        "position": int(item.get("position", 0)),
                    }
                    for item in measurements
                    if isinstance(item, dict)
                ]
        self.streaming_similarity_measurements = loaded
        if loaded:
            logger.info(
                "Loaded streaming similarity results for %d features",
                len(loaded),
            )

    def compute_similarities_with_residuals(self) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
        logger.info("Computing decoder-residual similarities...")
        if self.prefer_streaming_results and self.streaming_similarity_measurements:
            logger.info(
                "Using streaming similarity results for %d features",
                len(self.streaming_similarity_measurements),
            )
            self.similarity_measurements = dict(
                self.streaming_similarity_measurements)
            return self.similarity_measurements

        decoder_cfg = self.cfg.evals.topk_lora_autointerp.decoder_similarity
        threshold = float(decoder_cfg.similarity_threshold)
        sample_step = max(1, int(decoder_cfg.sample_every_n_tokens))
        top_k = int(decoder_cfg.n_top_similar_positions)
        metric = str(decoder_cfg.similarity_metric)

        accumulator = TopKAccumulator(top_k)

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        if device.type == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")

        for module_name, decoder_matrix_cpu in tqdm(
            self.module_decoder_matrices.items(),
            desc="Collecting similarities",
        ):
            mask_paths = self._discover_shards(module_name, "activation_masks")
            residual_paths = self._discover_shards(
                module_name, "residual_streams")
            if not mask_paths or not residual_paths:
                continue
            if len(mask_paths) != len(residual_paths):
                raise ValueError(
                    f"Mask/residual shard count mismatch for {module_name}: "
                    f"{len(mask_paths)} vs {len(residual_paths)}"
                )

            decoder_matrix = (
                decoder_matrix_cpu.to(device)
                if decoder_matrix_cpu.device != device
                else decoder_matrix_cpu
            )
            decoder_norms = None
            if metric == "cosine":
                module_norms = self.module_decoder_norms.get(module_name)
                if module_norms is None:
                    raise RuntimeError(
                        f"Missing decoder norms for module {module_name}"
                    )
                decoder_norms = (
                    module_norms.to(device)
                    if module_norms.device != device
                    else module_norms
                )

            global_offset = 0
            seq_len: Optional[int] = None

            for mask_path, residual_path in zip(mask_paths, residual_paths):
                mask_tensor = torch.load(mask_path)
                residual_tensor = torch.load(residual_path)

                if mask_tensor.numel() == 0 or residual_tensor.numel() == 0:
                    global_offset += mask_tensor.shape[0] * \
                        mask_tensor.shape[1]
                    continue

                if seq_len is None:
                    seq_len = residual_tensor.shape[1]
                elif residual_tensor.shape[1] != seq_len:
                    raise ValueError(
                        "Residual shard sequence length mismatch; expected consistent seq_len",
                    )

                mask_bool = mask_tensor.to(device=device) > 0
                residual_chunk = residual_tensor.to(
                    device=device, dtype=torch.float32)

                num_rows = mask_bool.shape[0] * mask_bool.shape[1]
                row_indices = (
                    torch.arange(num_rows, device=device, dtype=torch.long)
                    + global_offset
                )

                mask_flat = mask_bool.view(num_rows, -1)
                residual_flat = residual_chunk.view(num_rows, -1)

                if sample_step > 1:
                    sampled_indices = torch.arange(
                        0, num_rows, sample_step, device=device, dtype=torch.long
                    )
                    mask_flat = mask_flat.index_select(0, sampled_indices)
                    residual_flat = residual_flat.index_select(
                        0, sampled_indices)
                    row_indices = row_indices.index_select(0, sampled_indices)

                active_idx = mask_flat.nonzero(as_tuple=False)
                if active_idx.numel() == 0:
                    global_offset += num_rows
                    continue

                row_positions = row_indices.index_select(0, active_idx[:, 0])
                feature_indices = active_idx[:, 1]

                residual_vectors = residual_flat.index_select(
                    0, active_idx[:, 0]).to(dtype=torch.float32)
                decoder_vectors = (
                    decoder_matrix.index_select(
                        1, feature_indices).T.contiguous()
                )
                decoder_subset_norms = None
                if metric == "cosine" and decoder_norms is not None:
                    decoder_subset_norms = decoder_norms.index_select(
                        0, feature_indices)

                similarities = _pairwise_similarity(
                    residual_vectors,
                    decoder_vectors,
                    metric,
                    decoder_subset_norms,
                )

                keep_mask = similarities > threshold
                if not keep_mask.any():
                    global_offset += num_rows
                    continue

                keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)
                kept_similarities = similarities.index_select(0, keep_indices)
                kept_rows = row_positions.index_select(0, keep_indices)
                kept_features = feature_indices.index_select(0, keep_indices)

                if seq_len is None:
                    raise RuntimeError("Sequence length not initialized")

                for sim_value, row_value, feature_value in zip(
                    kept_similarities.tolist(),
                    kept_rows.tolist(),
                    kept_features.tolist(),
                ):
                    batch_idx = row_value // seq_len
                    seq_idx = row_value % seq_len
                    accumulator.add(
                        module_name,
                        int(feature_value),
                        float(sim_value),
                        int(batch_idx),
                        int(seq_idx),
                        int(row_value),
                    )

                global_offset += num_rows

        self.similarity_measurements = accumulator.export()
        return self.similarity_measurements

    def prepare_delphi_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        latent_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        max_examples = int(
            self.cfg.evals.topk_lora_autointerp.delphi_integration.max_examples_per_feature)

        for (module_name, feature_idx), measurements in self.similarity_measurements.items():
            if not measurements:
                continue
            sorted_measurements = sorted(
                measurements, key=lambda item: item["similarity"], reverse=True)
            top_examples = sorted_measurements[:max_examples]
            latent_dict[module_name].append(
                {
                    "feature_idx": feature_idx,
                    "module": module_name,
                    "activations": top_examples,
                    "decoder_norm": float(self.decoder_cache[(module_name, feature_idx)].norm().item()),
                }
            )
        return latent_dict


class HybridCausalAnalyzer:
    """Collect causal traces for TopKLoRA latents."""

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        tokenizer,
        wrapped_modules: Dict[str, TopKLoRALinearSTE],
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.wrapped_modules = wrapped_modules
        self.device = next(model.parameters()).device
        self.cache_root = Path(cfg.evals.topk_lora_autointerp.cache_root)
        self.causal_traces: Dict[Tuple[str, int],
                                 List[Dict[str, Any]]] = defaultdict(list)

    @staticmethod
    def _get_topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.zeros_like(z)
        topk_vals, topk_indices = torch.topk(z, k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_indices, 1.0)
        return mask

    def _extract_trigger_window(self, input_ids: torch.Tensor, batch_idx: int, seq_idx: int) -> List[int]:
        window = int(
            self.cfg.evals.topk_lora_autointerp.hybrid_causal.causal_trigger_window)
        start = max(0, seq_idx - window)
        return input_ids[batch_idx, start:seq_idx].detach().cpu().tolist()

    def _extract_effect_window(self, input_ids: torch.Tensor, batch_idx: int, seq_idx: int) -> List[int]:
        window = int(
            self.cfg.evals.topk_lora_autointerp.hybrid_causal.causal_effect_window)
        end = min(input_ids.shape[1], seq_idx + window)
        return input_ids[batch_idx, seq_idx:end].detach().cpu().tolist()

    def collect_causal_traces(self, dataloader: DataLoader, max_tokens: int) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
        tokens_seen = 0
        for batch in tqdm(dataloader, desc="Collecting causal traces"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                tokens_seen += int(attention_mask.sum().item())
            else:
                tokens_seen += int(input_ids.numel())

            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)

            input_ids_cpu = input_ids.detach().cpu()
            for name, module in self.wrapped_modules.items():
                if not isinstance(module, TopKLoRALinearSTE):
                    continue
                z = getattr(module, "_last_z", None)
                if z is None:
                    continue
                z_cpu = z.detach().cpu()
                mask = self._get_topk_mask(z_cpu, module._current_k())
                activated_positions = torch.nonzero(mask, as_tuple=False)
                for batch_idx, seq_idx, feat_idx in activated_positions.tolist():
                    trigger_tokens = self._extract_trigger_window(
                        input_ids_cpu, batch_idx, seq_idx)
                    effect_tokens = self._extract_effect_window(
                        input_ids_cpu, batch_idx, seq_idx)
                    trace = {
                        "feature_idx": feat_idx,
                        "module": name,
                        "position": seq_idx,
                        "activation_value": float(z_cpu[batch_idx, seq_idx, feat_idx].item()),
                        "trigger_tokens": trigger_tokens,
                        "effect_tokens": effect_tokens,
                        "trigger_text": self.tokenizer.decode(trigger_tokens, skip_special_tokens=True),
                        "effect_text": self.tokenizer.decode(effect_tokens, skip_special_tokens=True),
                    }
                    self.causal_traces[(name, feat_idx)].append(trace)

            if tokens_seen >= max_tokens:
                break
        return self.causal_traces

    def prepare_causal_delphi_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        latent_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        max_examples = int(
            self.cfg.evals.topk_lora_autointerp.delphi_integration.max_examples_per_feature)
        for (module_name, feature_idx), traces in self.causal_traces.items():
            if not traces:
                continue
            sorted_traces = sorted(
                traces, key=lambda item: item["activation_value"], reverse=True)
            top_traces = sorted_traces[:max_examples]
            latent_dict[module_name].append(
                {
                    "feature_idx": feature_idx,
                    "module": module_name,
                    "causal_traces": top_traces,
                    "n_activations": len(traces),
                    "avg_activation": float(np.mean([trace["activation_value"] for trace in traces])),
                }
            )
        return latent_dict


def delphi_collect_activations_topk(
    cfg: DictConfig,
    model: torch.nn.Module,
    tokenizer,
    wrapped_modules: Dict[str, TopKLoRALinearSTE],
) -> TopKLoRALatentCache:
    logger.info("Collecting TopKLoRA activations using UltraChat pipeline...")
    from src.utils import ultrachat_to_flat_tokens, pack_1d_stream

    SEQ_LEN = cfg.evals.topk_lora_autointerp.seq_len
    N_TOKENS_TARGET = cfg.evals.topk_lora_autointerp.n_tokens

    # Get 1D token stream from UltraChat, normalized and rendered with chat template
    tokens_1d = ultrachat_to_flat_tokens(
        tokenizer,
        splits=("train_sft",),
        add_eos_between=True,
        token_budget=N_TOKENS_TARGET,
        num_proc=os.cpu_count(),
        render_batch_size=256,
        encode_batch_size=1024,
        eos_id=tokenizer.eos_token_id
    )

    if tokens_1d.numel() == 0:
        raise RuntimeError("UltraChat produced no tokens after cleaning.")

    tokens_array = pack_1d_stream(
        tokens_1d, seq_len=SEQ_LEN)  # shape [N, SEQ_LEN]
    if not isinstance(tokens_array, torch.Tensor):
        tokens_array = torch.tensor(tokens_array, dtype=torch.long)
    else:
        tokens_array = tokens_array.to(dtype=torch.long)

    cache = TopKLoRALatentCache(
        model=model,
        tokenizer=tokenizer,
        wrapped_modules=wrapped_modules,
        cfg=cfg,
        batch_size=cfg.evals.topk_lora_autointerp.batch_size,
        collect_decoder_projections=cfg.evals.topk_lora_autointerp.decoder_similarity.cache_decoder_projections,
        collect_residuals=cfg.evals.topk_lora_autointerp.decoder_similarity.cache_residuals,
    )

    cache.run(n_tokens=N_TOKENS_TARGET, tokens=tokens_array)

    n_splits = int(getattr(cfg.evals.topk_lora_autointerp,
                           "cache_splits", 5))
    raw_cache_dir = Path(
        cfg.evals.topk_lora_autointerp.cache_root) / "raw_latents"
    cache_config = CacheConfig(
        dataset_repo="HuggingFaceH4/ultrachat_200k",
        dataset_split="train_sft",
        dataset_name="",
        dataset_column="messages",
        batch_size=int(cfg.evals.topk_lora_autointerp.batch_size),
        cache_ctx_len=int(SEQ_LEN),
        n_tokens=int(N_TOKENS_TARGET),
        n_splits=n_splits,
    )
    model_identifier = cast(
        str,
        getattr(cfg.model, "base_model", None)
        or getattr(cfg.model, "model_it_name", None)
        or getattr(cfg.model, "name", "unknown-topk-model"),
    )
    cache.save_splits(
        n_splits=n_splits,
        save_dir=raw_cache_dir,
        save_tokens=True,
        cache_config=cache_config,
        model_name=model_identifier,
    )
    cache.cleanup()
    logger.info("TopKLoRA activations cached at %s",
                cfg.evals.topk_lora_autointerp.cache_root)
    return cache


def run_topk_decoder_similarity(
    cfg: DictConfig,
    model: torch.nn.Module,
    tokenizer,
    wrapped_modules: Dict[str, TopKLoRALinearSTE],
) -> Tuple[DecoderSimilarityAnalyzer, Dict[str, List[Dict[str, Any]]]]:
    analyzer = DecoderSimilarityAnalyzer(
        cfg, model, tokenizer, wrapped_modules)
    similarities = analyzer.compute_similarities_with_residuals()
    delphi_dataset = analyzer.prepare_delphi_dataset()

    results_path = Path(cfg.evals.topk_lora_autointerp.cache_root) / \
        "decoder_similarity_results.pkl"
    with open(results_path, "wb") as handle:
        pickle.dump({"similarities": similarities,
                    "delphi_dataset": delphi_dataset}, handle)
    logger.info("Decoder similarity results saved to %s", results_path)
    return analyzer, delphi_dataset


def run_topk_hybrid_causal(
    cfg: DictConfig,
    model: torch.nn.Module,
    tokenizer,
    wrapped_modules: Dict[str, TopKLoRALinearSTE],
) -> Tuple[HybridCausalAnalyzer, Dict[str, List[Dict[str, Any]]]]:
    dataset = load_dataset("EleutherAI/pile-10k", split="train[:1%]")

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.evals.topk_lora_autointerp.seq_len,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=list(getattr(dataset, "column_names", [])),
    )

    dataloader = DataLoader(
        tokenized_dataset,  # type: ignore[arg-type]
        batch_size=cfg.evals.topk_lora_autointerp.batch_size,
        shuffle=False,
    )

    analyzer = HybridCausalAnalyzer(cfg, model, tokenizer, wrapped_modules)
    traces = analyzer.collect_causal_traces(
        dataloader, cfg.evals.topk_lora_autointerp.n_tokens)
    delphi_dataset = analyzer.prepare_causal_delphi_dataset()

    results_path = Path(
        cfg.evals.topk_lora_autointerp.cache_root) / "hybrid_causal_results.pkl"
    with open(results_path, "wb") as handle:
        pickle.dump(
            {"traces": traces, "delphi_dataset": delphi_dataset}, handle)
    logger.info("Hybrid causal results saved to %s", results_path)
    return analyzer, delphi_dataset


def run_delphi_explanation_generation(
    cfg: DictConfig,
    tokenizer,
    delphi_dataset: Dict[str, List[Dict[str, Any]]],
    *,
    approach: str = "decoder_similarity",
) -> None:
    if not DELPHI_AVAILABLE:
        logger.warning(
            "Delphi library not available; skipping explanation generation.")
        return
    if not delphi_dataset:
        logger.info("No Delphi dataset entries to explain.")
        return

    logger.info("Generating Delphi explanations for approach=%s", approach)

    raw_dataset_dir = Path(
        cfg.evals.topk_lora_autointerp.cache_root) / "raw_latents"
    if not raw_dataset_dir.exists():
        logger.error(
            "Expected latent cache at %s but directory was not found.",
            raw_dataset_dir,
        )
        return

    latents_selection: Dict[str, torch.Tensor] = {}
    total_latents = 0
    for module_name, records in delphi_dataset.items():
        if not isinstance(records, list):
            continue
        feature_ids: List[int] = []
        for entry in records:
            if not isinstance(entry, dict):
                continue
            raw_idx = entry.get("feature_idx")
            if raw_idx is None:
                continue
            feature_ids.append(int(raw_idx))
        feature_ids.sort()
        if not feature_ids:
            continue
        latents_selection[module_name] = torch.tensor(
            feature_ids, dtype=torch.long)
        total_latents += len(feature_ids)

    if not latents_selection:
        logger.info(
            "No candidate latents available for Delphi explanations; skipping.")
        return

    integration_cfg = cfg.evals.topk_lora_autointerp.delphi_integration
    sampler_cfg = SamplerConfig(
        n_examples_train=int(getattr(integration_cfg, "n_examples_train", 30)),
        n_examples_test=int(getattr(integration_cfg, "n_examples_test", 40)),
        n_quantiles=int(getattr(integration_cfg, "n_quantiles", 10)),
        train_type=str(getattr(integration_cfg, "train_type", "mix")),
        test_type=str(getattr(integration_cfg, "test_type", "quantiles")),
        ratio_top=float(getattr(integration_cfg, "ratio_top", 0.3)),
    )

    constructor_kwargs: Dict[str, Any] = {}
    constructor_overrides = getattr(
        integration_cfg, "constructor_overrides", None)
    if constructor_overrides:
        container = OmegaConf.to_container(
            constructor_overrides, resolve=True)  # type: ignore[arg-type]
        if isinstance(container, dict):
            constructor_kwargs = {
                str(key): value for key, value in container.items()}

    constructor_cfg = ConstructorConfig(**constructor_kwargs)

    latent_dataset = TopKLoRALatentDataset(
        raw_dir=raw_dataset_dir,
        modules=list(latents_selection.keys()),
        latents=latents_selection,
        tokenizer=tokenizer,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
    )

    logger.info(
        "Prepared Delphi LatentDataset with %d modules and %d latents from %s",
        len(latents_selection),
        total_latents,
        raw_dataset_dir,
    )

    # offline_cls = cast(Any, Offline)
    # client = offline_cls(
    #     cfg.evals.topk_lora_autointerp.delphi_integration.explainer_model,
    #     max_memory=0.8,
    #     max_model_len=5120,
    #     num_gpus=1,
    # )
    client = Offline(
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        num_gpus=4,                 # TP=2
        max_model_len=18000,         # smaller KV → faster & safer
        max_memory=0.65,
        prefix_caching=False,
        batch_size=1,
        enforce_eager=False,        # allow CUDA graphs
        number_tokens_to_generate=14_500,
        # max_num_batched_tokens=3072,
    )
    logger.info('LOADED MODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    if approach == "hybrid_causal" and cfg.evals.topk_lora_autointerp.delphi_integration.adapt_prompts_for_causal:
        contrastive_cls = cast(Any, ContrastiveExplainer)
        explainer = contrastive_cls(
            client,
            threshold=0.3,
            max_examples=15,
            max_non_activating=5,
            verbose=True,
        )
    else:
        default_cls = cast(Any, DefaultExplainer)
        explainer = default_cls(client)

    explanation_dir = Path(
        cfg.evals.topk_lora_autointerp.cache_root) / "explanations"
    explanation_dir.mkdir(parents=True, exist_ok=True)

    def postprocess_callback(result):  # type: ignore[no-untyped-def]
        out_path = explanation_dir / f"{result.record.latent}.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result.explanation, handle, indent=2)
        return result

    explanation_pipe = process_wrapper(
        explainer, postprocess=postprocess_callback)
    if explanation_pipe is None:
        logger.warning(
            "Delphi process wrapper unavailable; skipping explanations.")
        print('EXITING EARLY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return
    print('len of latent_dataset:', total_latents)
    pipeline = Pipeline(
        latent_dataset,
        explanation_pipe          # runs simulation scoring in one stage
    )
    # for module_name, records in delphi_dataset.items():
    #     for record in records:
    #         try:
    #             pipeline(record)  # type: ignore[operator]
    #         except Exception as exc:  # pragma: no cover - defensive
    #             logger.warning("Failed to generate explanation for %s/%s: %s",
    #                            module_name, record["feature_idx"], exc)
    max_concurrent = 3  # Process one at a time to avoid memory pressure

    asyncio.run(pipeline.run(max_concurrent=max_concurrent))

    print(
        f"✅ Pipeline completed with max_concurrent={max_concurrent} (memory-safe)")
    print('EXITING LATE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


def visualize_topk_results(
    cfg: DictConfig,
    results: Dict[str, Any],
    *,
    approach: str = "decoder_similarity",
) -> None:
    if not cfg.evals.topk_lora_autointerp.generate_visualizations:
        return

    viz_dir = Path(cfg.evals.topk_lora_autointerp.cache_root) / \
        "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    if approach == "decoder_similarity":
        similarities = results.get("similarities", {})
        if not similarities:
            return
        all_sims: List[float] = []
        feature_counts: Dict[str, int] = {}
        for key, measurements in similarities.items():
            module_name, feature_idx = key
            feature_key = f"{module_name}:{feature_idx}"
            feature_counts[feature_key] = len(measurements)
            all_sims.extend([m["similarity"] for m in measurements])

        if all_sims:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(all_sims, bins=50, alpha=0.8, color="steelblue")
            axes[0].set_xlabel("Cosine similarity")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Decoder-residual similarity distribution")

            top_features = sorted(feature_counts.items(),
                                  key=lambda item: item[1], reverse=True)[:20]
            labels = [item[0] for item in top_features]
            values = [item[1] for item in top_features]
            axes[1].bar(range(len(values)), values, color="darkorange")
            axes[1].set_xticks(range(len(values)))
            axes[1].set_xticklabels(
                labels, rotation=45, ha="right", fontsize=8)
            axes[1].set_ylabel("High-similarity count")
            axes[1].set_title("Top features by similarity hits")
            plt.tight_layout()
            plt.savefig(viz_dir / f"{approach}_analysis.png", dpi=200)
            plt.close(fig)

    elif approach == "hybrid_causal":
        traces = results.get("traces", {})
        if not traces:
            return
        activation_values: List[float] = []
        trigger_lengths: List[int] = []
        effect_lengths: List[int] = []
        for trace_list in traces.values():
            activation_values.extend(
                [trace["activation_value"] for trace in trace_list])
            trigger_lengths.extend([len(trace["trigger_tokens"])
                                   for trace in trace_list])
            effect_lengths.extend([len(trace["effect_tokens"])
                                  for trace in trace_list])

        if activation_values:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(activation_values, bins=50,
                         alpha=0.8, color="forestgreen")
            axes[0].set_xlabel("Activation value")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Feature activation distribution")

            axes[1].hist(trigger_lengths, bins=30, alpha=0.6,
                         label="Trigger", color="royalblue")
            axes[1].hist(effect_lengths, bins=30, alpha=0.6,
                         label="Effect", color="salmon")
            axes[1].legend()
            axes[1].set_xlabel("Tokens")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Causal window lengths")
            plt.tight_layout()
            plt.savefig(viz_dir / f"{approach}_analysis.png", dpi=200)
            plt.close(fig)


def summarise_autointerp_run(cfg: DictConfig, wrapped_modules: Dict[str, TopKLoRALinearSTE]) -> Path:
    summary_path = Path(
        cfg.evals.topk_lora_autointerp.cache_root) / "analysis_summary.json"
    summary = {
        "approach": cfg.evals.topk_lora_autointerp.approach,
        "n_tokens_processed": cfg.evals.topk_lora_autointerp.n_tokens,
        "n_modules_analyzed": len(wrapped_modules),
        "timestamp": datetime.utcnow().isoformat(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path

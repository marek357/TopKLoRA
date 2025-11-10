from src.utils import load_ultrachat, normalize_ultrachat_messages
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from delphi.pipeline import process_wrapper, Pipeline
from delphi.scorers import DetectionScorer, SurprisalScorer, OpenAISimulator, FuzzingScorer
from delphi.explainers import DefaultExplainer, ContrastiveExplainer
from delphi.clients import Offline, OpenRouter
from delphi.config import SamplerConfig, ConstructorConfig
from delphi.latents import LatentDataset
import asyncio
from itertools import islice
from datasets import load_dataset
from delphi.latents import LatentCache
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams
import dataclasses
import torch
import json
import os
from typing import Iterable, List, Optional, Union, Dict, Any
import hashlib
import pickle
from functools import wraps
import fcntl
import time
import random
import sys
import re

# Add path for our improvements
sys.path.append('/scratch/network/ssd/marek/lora_interp/src')

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_interpretability_rankings():
    """Load our interpretability analysis results."""
    try:
        with open('/scratch/network/ssd/marek/lora_interp/latent_interpretability_analysis.json', 'r') as f:
            results = json.load(f)
        print(f"✓ Loaded interpretability rankings for {len(results)} latents")
        return results
    except Exception as e:
        print(f"Warning: Could not load interpretability rankings: {e}")
        return []


def get_priority_latents(interpretability_results, top_k=15):
    """Get the most interpretable latents based on our analysis."""
    if not interpretability_results or True:
        print(
            f"No interpretability results, sampling 250 latents per matrix (range 0-{top_k})")
        # sample randomly 250 latent ids from 0 to top_k for each matrix
        # this means that we will have an array of 7 * 250 = 1750 latents to explain
        # set random seed for reproducibility
        local_rng = random.Random(42)
        return [
            local_rng.sample(range(top_k), 150)
            for _ in range(7)
        ]

    # Sort by interpretability score
    sorted_latents = sorted(
        interpretability_results,
        key=lambda x: x['interpretability_score'],
        reverse=True
    )

    priority_latents = [result['latent_id']
                        for result in sorted_latents[:top_k]]

    print(f"\n{'='*60}")
    print("TARGETING MOST INTERPRETABLE LATENTS")
    print(f"{'='*60}")
    print(f"Selected top {len(priority_latents)} interpretable latents:")

    for i, result in enumerate(sorted_latents[:top_k]):
        latent_id = result['latent_id']
        score = result['interpretability_score']
        accuracy = result['metrics']['accuracy']
        activation_rate = result['metrics']['activation_rate']

        # Try to infer function from top tokens
        top_tokens = result['metrics'].get('most_common_tokens', [])
        if top_tokens and len(top_tokens[0]) > 1:
            main_token = top_tokens[0][0] if top_tokens[0][1] > 2 else "mixed"
        else:
            main_token = "unknown"

        print(f"  {i+1:2d}. Latent {latent_id:3d} (Score: {score:.2f}, Acc: {accuracy:.2f}, Act: {activation_rate:.2f}) - '{main_token}'")

    print(f"{'='*60}\n")

    return priority_latents


def _infer_function_from_tokens(metrics):
    """Infer latent function from top tokens."""
    top_tokens = metrics.get('most_common_tokens', [])
    if not top_tokens:
        return "Unknown"

    main_token = top_tokens[0][0]

    if main_token == ' or':
        return "Logical Disjunction (OR)"
    elif main_token in [' and', ',']:
        return "Logical Conjunction (AND/Lists)"
    elif main_token.isdigit():
        return "Numeric Content"
    elif main_token in [' in', ' of', ' for']:
        return f"Relational ({main_token.strip()})"
    elif main_token in ['\n', '.']:
        return "Formatting/Structure"
    elif main_token in [' his', ' her', ' their']:
        return "Possessive/Reference"
    else:
        return f"Token-focused ({main_token})"


def create_enhanced_save_functions(model_str):
    """Create enhanced save functions that include interpretability metadata."""

    # Load interpretability context once
    interp_results = load_interpretability_rankings()
    interp_lookup = {result['latent_id']: result for result in interp_results}

    def enhanced_save_explanation(result, explainer_type):
        """Enhanced explanation saving with interpretability context."""
        latent_str = str(result.record.latent)
        safe = latent_str.replace(".", "_").replace(":", "_").replace(" ", "_")

        # Extract latent ID
        match = re.search(r'(\d+)$', latent_str)
        latent_id = int(match.group(1)) if match else 0

        # Get interpretability context
        context = interp_lookup.get(latent_id, {})

        out_dir = f"autointerp/{model_str}/explanations/" + explainer_type
        os.makedirs(out_dir, exist_ok=True)

        # Enhanced output with interpretability metadata
        output_data = {
            "explanation": result.explanation,
            "latent_id": latent_id,
            "interpretability_metadata": {
                "interpretability_score": context.get('interpretability_score', 0),
                "activation_rate": context.get('metrics', {}).get('activation_rate', 0),
                "accuracy": context.get('metrics', {}).get('accuracy', 0),
                "avg_sparsity": context.get('metrics', {}).get('avg_sparsity', 0),
                "token_diversity": context.get('metrics', {}).get('token_diversity', 0),
                "most_common_tokens": context.get('metrics', {}).get('most_common_tokens', [])[:5],
                "predicted_function": _infer_function_from_tokens(context.get('metrics', {}))
            }
        }

        path = os.path.join(out_dir, f"{safe}.json")
        with open(path, "w") as f:
            json.dump(output_data, f, indent=2)

        # Print progress with context
        score = context.get('interpretability_score', 0)
        func = _infer_function_from_tokens(context.get('metrics', {}))
        print(
            f"✓ Explained Latent {latent_id} (Score: {score:.2f}, Function: {func})")

        return result

    def enhanced_save_score(result, scorer_type):
        """Enhanced score saving with interpretability context."""
        latent_str = str(result.record.latent)
        safe = latent_str.replace(".", "_").replace(":", "_").replace(" ", "_")

        # Extract latent ID
        match = re.search(r'(\d+)$', latent_str)
        latent_id = int(match.group(1)) if match else 0

        # Get interpretability context
        context = interp_lookup.get(latent_id, {})

        out_dir = f"autointerp/{model_str}/scores/{scorer_type}"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{safe}.json")

        # Convert score to dict and ensure it's a proper dict
        score_obj = result.score
        if hasattr(score_obj, "to_json_string"):
            score_data = json.loads(score_obj.to_json_string())
        elif isinstance(score_obj, list):
            score_data = {"scores": [
                dataclasses.asdict(elem) for elem in score_obj]}
        elif isinstance(score_obj, dict):
            score_data = dict(score_obj)  # Ensure it's a mutable dict
        else:
            score_data = {"score": str(score_obj)}

        # Ensure score_data is a dict before adding metadata
        if not isinstance(score_data, dict):
            score_data = {"original_score": score_data}

        # Add interpretability metadata
        score_data['interpretability_metadata'] = {
            "latent_id": latent_id,
            "interpretability_score": context.get('interpretability_score', 0),
            "predicted_function": _infer_function_from_tokens(context.get('metrics', {})),
            "baseline_accuracy": context.get('metrics', {}).get('accuracy', 0),
            "activation_sparsity": context.get('metrics', {}).get('avg_sparsity', 0),
            "token_diversity": context.get('metrics', {}).get('token_diversity', 0)
        }

        with open(path, "w") as f:
            json.dump(score_data, f, indent=2)

        # Print progress with context
        score = context.get('interpretability_score', 0)
        func = _infer_function_from_tokens(context.get('metrics', {}))
        print(
            f"✓ Scored Latent {latent_id} (Score: {score:.2f}, Function: {func}) - {scorer_type}")

        return result

    return enhanced_save_explanation, enhanced_save_score


class LLMResponseCache:
    """Cache for LLM responses to avoid redundant API calls."""

    def __init__(self, cache_dir: str = "llm_cache", base_dir: str = "autointerp"):
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.explanation_cache_dir = self.cache_dir / "explanations"
        self.detection_cache_dir = self.cache_dir / "detection"
        self.explanation_cache_dir.mkdir(exist_ok=True)
        self.detection_cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, data) -> str:
        """Generate a hash key for the input data."""
        if hasattr(data, '__dict__'):
            # For objects with attributes
            cache_data = str(sorted(data.__dict__.items()))
        elif isinstance(data, dict):
            cache_data = str(sorted(data.items()))
        else:
            cache_data = str(data)

        return hashlib.md5(cache_data.encode()).hexdigest()

    def _safe_file_operation(self, filepath, operation, mode='r', max_retries=5):
        """
        Perform file operations with proper locking and retry logic.
        Safe for concurrent access by multiple processes.
        """
        for attempt in range(max_retries):
            try:
                # Create lock file path
                lock_file = str(filepath) + '.lock'

                with open(lock_file, 'w') as lock_fd:
                    # Acquire exclusive lock
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)

                    # Perform the operation
                    return operation(filepath, mode)

            except (IOError, OSError) as e:
                if attempt < max_retries - 1:
                    # Wait with exponential backoff + jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
            finally:
                # Clean up lock file if it exists
                try:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                except:
                    pass  # Ignore cleanup errors

    def get_explanation(self, record) -> Optional[Dict[str, Any]]:
        """Get cached explanation or return None if not found."""
        cache_key = self._get_cache_key({
            'latent': str(record.latent),
            # First 5 for cache key
            'activating_examples': [str(ex) for ex in record.examples[:5]],
            # First 5 for cache key
            'non_activating_examples': [str(ex) for ex in record.not_active[:5]]
        })

        cache_file = self.explanation_cache_dir / f"{cache_key}.json"

        def read_explanation(filepath, mode):
            if not filepath.exists():
                return None
            with open(filepath, 'r') as f:
                data = json.load(f)
                print(f"✓ Found cached explanation for {record.latent}")
                return data['explanation']

        try:
            return self._safe_file_operation(cache_file, read_explanation, 'r')
        except Exception as e:
            print(
                f"Warning: Failed to load explanation cache {cache_file}: {e}")
            return None

    def save_explanation(
        self,
        record,
        explanation: str,
        *,
        activating_sequences: Optional[List[Any]] = None,
        non_activating_sequences: Optional[List[Any]] = None,
    ):
        """Save explanation to cache."""
        cache_key = self._get_cache_key({
            'latent': str(record.latent),
            'activating_examples': [str(ex) for ex in record.examples[:5]],
            'non_activating_examples': [str(ex) for ex in record.not_active[:5]]
        })
        # latent_str = str(record.latent)
        # cache_key = latent_str.replace(
        #     ".", "_").replace(":", "_").replace(" ", "_")

        cache_file = self.explanation_cache_dir / f"{cache_key}.json"

        activating_sequences = activating_sequences or []
        non_activating_sequences = non_activating_sequences or []

        def _make_json_safe(value):
            if isinstance(value, dict):
                return {k: _make_json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_make_json_safe(v) for v in value]
            if isinstance(value, (str, int, float)) or value is None:
                return value
            return str(value)

        def _sanitize_payload(seq_list):
            sanitized = []
            for item in seq_list:
                if item is None:
                    continue
                sanitized.append(_make_json_safe(item))
            return sanitized

        activating_payload = _sanitize_payload(activating_sequences)
        non_activating_payload = _sanitize_payload(non_activating_sequences)

        activating_preview = [
            CachedExplainer.format_sequence_with_metadata(item)
            for item in activating_payload
        ]
        non_activating_preview = [
            CachedExplainer.format_sequence_with_metadata(item)
            for item in non_activating_payload
        ]

        def write_explanation(filepath, mode):
            # Write to temporary file first, then atomic rename
            temp_file = str(filepath) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump({
                    'latent': str(record.latent),
                    'explanation': explanation,
                    'activating_sequences': activating_payload,
                    'non_activating_sequences': non_activating_payload,
                    'activating_sequences_preview': activating_preview,
                    'non_activating_sequences_preview': non_activating_preview,
                    # Simple timestamp
                    'timestamp': str(torch.tensor(0).item())
                }, f, indent=2)

            # Atomic rename
            os.rename(temp_file, filepath)
            print(f"✓ Cached explanation for {record.latent}")
            return True

        try:
            self._safe_file_operation(cache_file, write_explanation, 'w')
        except Exception as e:
            print(f"Warning: Failed to save explanation cache: {e}")

    def get_detection_score(self, record) -> Optional[Dict[str, Any]]:
        """Get cached detection score or return None if not found."""
        cache_key = self._get_cache_key({
            'latent': str(record.latent),
            'explanation': getattr(record, 'explanation', ''),
            'examples_hash': str(hash(str(record.examples[:3]) + str(record.not_active[:3])))
        })

        cache_file = self.detection_cache_dir / f"{cache_key}.pkl"

        def read_detection(filepath, mode):
            if not filepath.exists():
                return None
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                print(f"✓ Found cached detection score for {record.latent}")
                return data['score']

        try:
            return self._safe_file_operation(cache_file, read_detection, 'rb')
        except Exception as e:
            print(f"Warning: Failed to load detection cache {cache_file}: {e}")
            return None

    def save_detection_score(self, record, score):
        """Save detection score to cache."""
        cache_key = self._get_cache_key({
            'latent': str(record.latent),
            'explanation': getattr(record, 'explanation', ''),
            'examples_hash': str(hash(str(record.examples[:3]) + str(record.not_active[:3])))
        })

        cache_file = self.detection_cache_dir / f"{cache_key}.pkl"

        def write_detection(filepath, mode):
            # Write to temporary file first, then atomic rename
            temp_file = str(filepath) + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump({
                    'latent': str(record.latent),
                    'score': score,
                    'timestamp': str(torch.tensor(0).item())
                }, f)

            # Atomic rename
            os.rename(temp_file, filepath)
            print(f"✓ Cached detection score for {record.latent}")
            return True

        try:
            self._safe_file_operation(cache_file, write_detection, 'wb')
        except Exception as e:
            print(f"Warning: Failed to save detection cache: {e}")

    def clear_cache(self):
        """Clear all cached responses."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.explanation_cache_dir.mkdir(exist_ok=True)
            self.detection_cache_dir.mkdir(exist_ok=True)
            print("✓ Cache cleared")

    def get_cache_stats(self):
        """Get cache statistics."""
        explanation_count = len(
            list(self.explanation_cache_dir.glob("*.json")))
        detection_count = len(list(self.detection_cache_dir.glob("*.pkl")))
        return {
            'explanation_count': explanation_count,
            'detection_count': detection_count,
            'cache_dir': str(self.cache_dir)
        }


class CachedExplainer:
    """Wrapper around Delphi explainer with caching."""

    def __init__(self, base_explainer, cache: LLMResponseCache, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        self.base_explainer = base_explainer
        self.cache = cache
        self.tokenizer = tokenizer
        self.ran = False

    @staticmethod
    def _clean_token_text(text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("Ġ", " ").replace("▁", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _token_ids_from(example_tokens) -> List[int]:
        if example_tokens is None:
            return []
        if hasattr(example_tokens, "detach"):
            example_tokens = example_tokens.detach().cpu()
        if hasattr(example_tokens, "tolist"):
            example_tokens = example_tokens.tolist()
        if isinstance(example_tokens, (list, tuple)):
            return [int(t) for t in example_tokens]
        return []

    @staticmethod
    def _truncate_text(text: str, limit: int = 160) -> str:
        snippet = text.replace("\n", " ")
        if len(snippet) <= limit:
            return snippet
        return snippet[:limit - 3] + "…"

    @staticmethod
    def _activation_list_from(example_activations) -> List[float]:
        if example_activations is None:
            return []
        if hasattr(example_activations, "detach"):
            example_activations = example_activations.detach().cpu()
        if hasattr(example_activations, "tolist"):
            example_activations = example_activations.tolist()
        if isinstance(example_activations, (list, tuple)):
            return [float(a) for a in example_activations]
        return []

    def _extract_top_token_info(self, example) -> Optional[Dict[str, Any]]:
        activations = self._activation_list_from(
            getattr(example, "activations", None))
        if not activations:
            return None

        best_idx = int(max(range(len(activations)),
                       key=lambda idx: activations[idx]))
        best_activation = float(activations[best_idx])

        token_ids = self._token_ids_from(getattr(example, "tokens", None))
        token_id = token_ids[best_idx] if best_idx < len(token_ids) else None

        token_text = None
        str_tokens = getattr(example, "str_tokens", None)
        if isinstance(str_tokens, (list, tuple)) and best_idx < len(str_tokens):
            token_text = str(str_tokens[best_idx])
        elif self.tokenizer is not None and token_id is not None:
            try:
                token_text = self.tokenizer.convert_ids_to_tokens(token_id)
                if isinstance(token_text, (list, tuple)):
                    token_text = token_text[0] if token_text else None
            except Exception:
                token_text = None
            if not token_text:
                try:
                    token_text = self.tokenizer.decode(
                        [token_id], skip_special_tokens=False)
                except Exception:
                    token_text = None
        token_text = self._clean_token_text(token_text) if token_text else None

        normalized_acts = self._activation_list_from(
            getattr(example, "normalized_activations", None))
        normalized_value = float(normalized_acts[best_idx]) if best_idx < len(
            normalized_acts) else None

        return {
            "token_id": int(token_id) if token_id is not None else None,
            "token": token_text,
            "activation": best_activation,
            "normalized_activation": normalized_value,
            "position": best_idx,
        }

    def _normalize_cached_activating(self, activating: Iterable[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in activating or []:
            if isinstance(item, dict):
                sequence_text = item.get("sequence")
                top_token = item.get("top_token")
                if sequence_text is None and "text" in item:
                    sequence_text = item.get("text")
                cleaned_sequence = self._clean_token_text(
                    str(sequence_text)) if sequence_text else ""
                normalized.append({
                    "sequence": cleaned_sequence,
                    "top_token": top_token,
                })
            else:
                normalized.append(
                    {"sequence": self._clean_token_text(str(item))})
        return normalized

    def _render_sequence(self, example) -> str:
        # Prefer precomputed string tokens when available
        str_tokens = getattr(example, "str_tokens", None)
        if str_tokens:
            joined = " ".join(str(token) for token in str_tokens)
            text = self._clean_token_text(joined)
            if text:
                return text

        tokens = getattr(example, "tokens", None)
        token_ids = self._token_ids_from(tokens)
        if token_ids:
            if self.tokenizer is not None:
                try:
                    decoded = self.tokenizer.decode(
                        token_ids, skip_special_tokens=True)
                    decoded = self._clean_token_text(decoded)
                    if decoded:
                        return decoded
                except Exception:
                    pass
            return " ".join(str(t) for t in token_ids)

        if hasattr(example, "text") and example.text is not None:
            text = self._clean_token_text(str(example.text))
            if text:
                return text

        return self._clean_token_text(str(example))

    def _extract_sequences(self, record):
        activating_entries: List[Dict[str, Any]] = []
        for example in getattr(record, "examples", []):
            sequence_text = self._render_sequence(example)
            if not sequence_text:
                continue
            top_token = self._extract_top_token_info(example)
            activating_entries.append({
                "sequence": sequence_text,
                "top_token": top_token,
            })

        non_activating = [
            seq for seq in (
                self._render_sequence(example)
                for example in getattr(record, "not_active", [])
            )
            if seq
        ]
        return activating_entries, non_activating

    @classmethod
    def format_sequence_with_metadata(cls, entry: Any, *, limit: int = 400) -> str:
        if isinstance(entry, dict):
            raw_sequence = entry.get("sequence", "")
            top_token = entry.get("top_token")
        else:
            raw_sequence = entry
            top_token = None

        sequence_text = cls._clean_token_text(
            str(raw_sequence)) if raw_sequence is not None else ""
        snippet = cls._truncate_text(sequence_text, limit=limit)

        if top_token and isinstance(top_token, dict):
            token_bits = []
            token_text = top_token.get("token")
            token_id = top_token.get("token_id")
            activation = top_token.get("activation")
            normalized = top_token.get("normalized_activation")
            position = top_token.get("position")

            if token_text:
                token_bits.append(f"token='{token_text}'")
            if token_id is not None:
                token_bits.append(f"id={token_id}")
            if position is not None:
                token_bits.append(f"pos={position}")
            if activation is not None:
                token_bits.append(f"act={float(activation):.4f}")
            if normalized is not None:
                token_bits.append(f"norm={float(normalized):.3f}")

            if token_bits:
                snippet = f"{snippet} ({', '.join(token_bits)})"

        return snippet

    @staticmethod
    def _preview_sequences(latent, activating, non_activating, limit=3):
        def _preview_block(name, seqs):
            if not seqs:
                print(f"   • No {name} sequences cached")
                return
            print(f"   • Top {name} sequences:")
            for idx, entry in enumerate(seqs[:limit], start=1):
                snippet = CachedExplainer.format_sequence_with_metadata(entry)
                print(f"     {idx:02d}: {snippet}")

        print(f"📄 Latent {latent}: cached sequences")
        _preview_block("activating", activating)
        _preview_block("non-activating", non_activating)

    async def __call__(self, record):
        # Check cache first
        cached_payload = self.cache.get_explanation(record)
        if cached_payload is not None:
            if isinstance(cached_payload, dict):
                explanation_text = cached_payload.get("explanation")
                activating = self._normalize_cached_activating(
                    cached_payload.get("activating_sequences", [])
                )
                non_activating = [
                    self._clean_token_text(str(item))
                    for item in cached_payload.get("non_activating_sequences", [])
                ]
            else:  # Backwards compatibility for old cache entries
                explanation_text = str(cached_payload)
                activating = []
                non_activating = []

            if explanation_text is not None:
                self._preview_sequences(
                    record.latent, activating, non_activating)

            # Create result object with cached explanation
            result = type('ExplanationResult', (), {
                'record': record,
                'explanation': explanation_text,
                'activating_sequences': activating,
                'non_activating_sequences': non_activating,
            })()
            return result

        # If not cached, call the base explainer
        print(f"⚡ Generating new explanation for {record.latent}")
        result = await self.base_explainer(record)

        activating, non_activating = self._extract_sequences(record)
        self._preview_sequences(record.latent, activating, non_activating)

        # Cache the result with sequences
        self.cache.save_explanation(
            record,
            result.explanation,
            activating_sequences=activating,
            non_activating_sequences=non_activating,
        )

        return result


class CachedDetectionScorer:
    """Wrapper around Delphi detection scorer with caching."""

    def __init__(self, base_scorer, cache: LLMResponseCache):
        self.base_scorer = base_scorer
        self.cache = cache

    async def __call__(self, record):
        # Check cache first
        cached_score = self.cache.get_detection_score(record)
        if cached_score is not None:
            # Create result object with cached score
            result = type('ScoreResult', (), {
                'record': record,
                'score': cached_score
            })()
            return result

        # If not cached, call the base scorer
        print(f"⚡ Generating new detection score for {record.latent}")
        result = await self.base_scorer(record)

        # Cache the result
        self.cache.save_detection_score(record, result.score)

        return result


class ChatTemplateCollator:
    def __init__(self, tokenizer, device, max_length=1024):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # For generation tasks, left padding is typically better
        self.original_padding_side = tokenizer.padding_side
        self.tokenizer.padding_side = "left"

    def __call__(self, examples):
        texts = []
        for ex in examples:
            msgs = ex.get("input", ex.get("chosen", ex.get("rejected")))
            text = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Efficient batch tokenization with optimized settings
        batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            # Add these for extra efficiency:
            return_attention_mask=True,
            return_token_type_ids=False,  # Not needed for most models
        ).to(self.device)  # Move directly to device

        return batch

    def __del__(self):
        # Restore original padding side
        if hasattr(self, 'original_padding_side'):
            self.tokenizer.padding_side = self.original_padding_side


def save_explanation(result, model_str, explainer_type):
    latent_str = str(result.record.latent)
    safe = latent_str.replace(".", "_").replace(":", "_").replace(" ", "_")

    out_dir = f"autointerp/{model_str}/explanations/" + explainer_type
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{safe}.json")
    with open(path, "w") as f:
        json.dump({
            "explanation": result.explanation,
            # 'activating_sequences': result.activating_sequences,
            # 'non_activating_sequences': result.non_activating_sequences,
        }, f, indent=2)
    return result


def save_score(result, model_str, scorer):
    # 1) Build a safe filename from the latent
    latent_str = str(result.record.latent)  # e.g. "layers.5.self.topk:42"
    safe = latent_str.replace(".", "_").replace(":", "_").replace(" ", "_")

    # 2) Ensure output directory
    out_dir = f"autointerp/{model_str}/scores/{scorer}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{safe}.json")

    # 3) Serialize result.score
    score_obj = result.score

    if hasattr(score_obj, "to_json_string"):
        # HF ModelOutput
        text = score_obj.to_json_string()
    elif isinstance(score_obj, list):
        # List of dataclasses (e.g. SurprisalOutput)
        # Convert each element to dict
        dicts = [dataclasses.asdict(elem) for elem in score_obj]
        text = json.dumps(dicts, indent=2)
    elif isinstance(score_obj, dict):
        # Already a dict
        text = json.dumps(score_obj, indent=2)
    else:
        # Fallback to plain repr
        text = json.dumps({"score": score_obj}, indent=2)

    # 4) Write
    with open(path, "w") as f:
        f.write(text)

    return result


"""
    if dataset_name == 'ultrachat':
        flat_ds = load_ultrachat()
    else:
        flat_ds = load_dataset(
            dataset_name, "en", split="train", streaming=True)

"""


def delphi_collect_activations(cfg, model, tokenizer, wrapped_modules, dataset_name='allenai/c4'):
    print('starting activation collection')

    flat_ds = load_dataset(dataset_name, "en", split="train", streaming=True)

    def stream_and_format(dataset, max_examples):
        for example in islice(dataset, max_examples):
            yield {
                "input": [
                    {"role": "user", "content": example["text"]},
                    {"role": "assistant", "content": ""}
                ]
            }

    MAX_BATCHES = 500_000
    flat_ds = list(stream_and_format(flat_ds, MAX_BATCHES))
    chat_collate = ChatTemplateCollator(tokenizer, device, max_length=256)

    loader = DataLoader(  # type: ignore[arg-type]
        flat_ds,
        batch_size=cfg.evals.causal_auto_interp.batch_size,
        shuffle=False,
        collate_fn=chat_collate,
        drop_last=False
    )

    N_TOKENS = 50_000_000
    SEQ_LEN = 256
    n_seqs = (N_TOKENS + SEQ_LEN - 1) // SEQ_LEN

    rows = []
    for batch in loader:
        # batch["input_ids"]: Tensor[B, SEQ_LEN]
        arr = batch["input_ids"].detach().cpu().clone()  # shape (B, SEQ_LEN)
        for row in arr:
            rows.append(row)
            if len(rows) >= n_seqs:
                break
        if len(rows) >= n_seqs:
            break

    # shape (n_seqs, SEQ_LEN)
    tokens_array = torch.stack(rows[:n_seqs], dim=0)

    topk_modules = {
        f"{name}.topk": module.topk
        for name, module in wrapped_modules.items()
    }

    # Temporarily enable TopK experiment mode so hooks see gated latents
    original_modes = {}
    for module in wrapped_modules.values():
        if hasattr(module, "is_topk_experiment"):
            original_modes[module] = module.is_topk_experiment
            module.is_topk_experiment = True

    cache = LatentCache(
        model=model,
        hookpoint_to_sparse_encode=topk_modules,
        batch_size=cfg.evals.causal_auto_interp.batch_size,
        transcode=False,
    )

    try:
        cache.run(
            n_tokens=N_TOKENS,
            tokens=tokens_array,
        )
        print("Cache collection complete. Checking cache contents...")
        total_entries = 0
        for hookpoint, locations in cache.cache.latent_locations.items():
            num_entries = int(
                locations.shape[0]) if locations is not None else 0
            total_entries += num_entries
            print(f"  {hookpoint}: {num_entries} non-zero activations")
        if total_entries == 0:
            print("WARNING: No latent activations were recorded.")
        out_dir = Path(
            f"cache/delphi_cache_{cfg.evals.causal_auto_interp.r}_{cfg.evals.causal_auto_interp.k}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        cache.save_splits(n_splits=4, save_dir=out_dir)
        widths = {
            f"{name}.topk": wrapped_modules[name].r
            for name in wrapped_modules
        }

        for hookpoint in widths:
            # the directory is literally raw_dir / hookpoint
            hp_dir = out_dir / hookpoint
            hp_dir.mkdir(parents=True, exist_ok=True)

            config = {
                "hookpoint": hookpoint,
                "width": widths[hookpoint]
            }
            with open(hp_dir / "config.json", "w") as f:
                json.dump(config, f)
    finally:
        for module, original in original_modes.items():
            module.is_topk_experiment = original


def delphi_score(cfg, model, tokenizer, wrapped_modules):
    config = cfg.evals.causal_auto_interp if hasattr(
        cfg.evals, 'causal_auto_interp') else cfg.evals.topk_lora_autointerp

    # Create model-specific identifier string based on config
    # Format: {model_type}_{r}_{k}
    model_type = getattr(cfg.model, 'type', 'unknown')
    r_val = getattr(cfg.model, 'r', config.r)
    k_val = getattr(cfg.model, 'k', config.k)
    model_str = f"{model_type}_{r_val}_{k_val}"

    # Initialize cache with proper model-specific directory
    cache_subdir = f"llm_cache_{os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}"
    llm_cache = LLMResponseCache(
        cache_dir=cache_subdir, base_dir=f"autointerp_layer18/{model_str}")

    # Print cache status
    print("\n" + "="*50)
    print("LLM CACHE STATUS")
    print("="*50)
    print(f"Model identifier: {model_str}")
    explanation_files = len(
        list(llm_cache.explanation_cache_dir.glob("*.json")))
    detection_files = len(list(llm_cache.detection_cache_dir.glob("*.pkl")))
    print(f"Cached explanations: {explanation_files}")
    print(f"Cached detection scores: {detection_files}")
    print(f"Cache directory: {llm_cache.cache_dir}")
    print("="*50 + "\n")

    topk_modules = [
        # filter out query projections -- these have already been analyzed
        f"{name}.topk" for name, _ in wrapped_modules.items() if 'q_proj' not in name
    ]
    print(topk_modules)
    model.cpu()
    del model
    del wrapped_modules

    # Load interpretability rankings and get priority latents
    print("\n" + "="*60)
    print("ENHANCED INTERPRETABILITY-FOCUSED ANALYSIS")
    print("="*60)

    interp_results = load_interpretability_rankings()
    priority_latents = get_priority_latents(
        interp_results, top_k=config.r)
    model_type = getattr(cfg.model, 'type', 'sft_model')

    # 1) Load the raw cache you saved
    dataset = LatentDataset(
        raw_dir=Path(
            f"cache/{model_type}/layer18/{config.r}_{config.k}"
        ),
        modules=topk_modules,
        latents={
            # Focus on most interpretable latents only
            name: torch.tensor(priority_latents[idx + 1], dtype=torch.long)
            for idx, name in enumerate(topk_modules)
        },
        tokenizer=tokenizer,
        sampler_cfg=SamplerConfig(
            n_examples_train=30,     # Increased training examples for better analysis
            n_examples_test=40,      # More test examples for robust evaluation
            n_quantiles=10,          # Standard quantile analysis
            train_type='mix',        # Mixed sampling for diverse training examples
            test_type='quantiles',   # Quantile-based testing
            ratio_top=0.3           # Focus on top 30% activations
        ),
        constructor_cfg=ConstructorConfig(
            # Enhanced contrastive analysis for better interpretability
            # faiss_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            # faiss_embedding_cache_enabled=True,
            # faiss_embedding_cache_dir=".embedding_cache",
            # example_ctx_len=32,      # Context length for examples
            # min_examples=200,        # Minimum examples for robust analysis
            # n_non_activating=20,     # Non-activating examples for contrast
            # center_examples=True,    # Center examples for better analysis
            # non_activating_source="FAISS",  # Use FAISS for better negative examples
            # neighbours_type="co-occurrence"  # Co-occurrence based neighbors
        ),
    )

    # 2) Build your explainer client + explainer
    # class OpenRouter(Client):
    # def __init__(
    #     self,
    #     model: str,
    #     api_key: str | None = None,
    #     base_url="https://openrouter.ai/api/v1/chat/completions",
    #     max_tokens: int = 3000,
    #     temperature: float = 1.0,
    # ):
    # client = OpenRouter(
    #     "Qwen/Qwen2.5-32B-Instruct-AWQ",
    #     max_tokens=25_768, base_url="http://127.0.0.1:8081/v1/chat/completions"
    # )

    # GPU Memory Management Configuration

    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Based on testing, the optimal configuration is:
    # - Single GPU (GPU 0, 3, or 4 are all free)
    # - Conservative memory settings to avoid OOM errors
    # - Reduced context length to fit in memory

    # Use GPUs 0 and 3 (both completely free)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(
        f"🔧 Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} (multi-GPU with tensor parallelism)")

    # Set PyTorch CUDA memory management for fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

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

    # Add device attribute for SurprisalScorer compatibility
    client.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print("✅ Model loaded successfully with multi-GPU tensor parallelism!")
    print(
        f"   - GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} (tensor parallelism)")

    openai_run = False
    if not openai_run:

        base_explainer = DefaultExplainer(client, cot=True)
        explainer = CachedExplainer(
            base_explainer, cache=llm_cache, tokenizer=tokenizer)
        explainer_pipe = process_wrapper(
            explainer,
            postprocess=lambda x: save_explanation(
                x, model_str, 'enhanced_default')
        )

        base_detection_scorer = DetectionScorer(
            client, tokenizer=tokenizer, n_examples_shown=5)
        detection_scorer = CachedDetectionScorer(
            base_detection_scorer, cache=llm_cache)

        # Enhanced preprocessing and scoring
        def preprocess(explained):
            rec = explained.record
            rec.explanation = explained.explanation
            rec.extra_examples = rec.not_active
            return rec

        detection_pipe = process_wrapper(
            detection_scorer,
            preprocess=preprocess,
            postprocess=lambda x: save_score(
                x, model_str, 'enhanced_detection')
        )

        # Enhanced pipeline with multiple scoring methods
        print(
            f"Running enhanced interpretability analysis on {len(priority_latents)} latents")
        print(f"Analysis includes: explanations, detection scoring, and surprisal analysis")

        # Multi-stage pipeline
        # Capture model_str in closure for the async function
        _model_str = model_str

        async def comprehensive_scoring(explained):
            """Run both detection and surprisal scoring."""
            rec = explained.record
            rec.explanation = explained.explanation
            rec.extra_examples = rec.not_active

            # Run detection scoring
            try:
                det_result = await detection_scorer(rec)
                save_score(
                    det_result, _model_str, 'enhanced_detection')
            except Exception as e:
                print(f"Detection scoring failed for {rec.latent}: {e}")

            return explained

        comprehensive_pipe = process_wrapper(comprehensive_scoring)

        # 5) Run the enhanced pipeline
        pipeline = Pipeline(
            dataset,
            explainer_pipe,
            comprehensive_pipe,
            # comprehensive_pipe  # Use comprehensive scoring instead of just detection
        )
    else:
        simulator = OpenAISimulator(
            client,
            tokenizer=tokenizer,      # use the same tokenizer as your dataset

        )

        # 3. Wrap it in a process pipe (optional preprocess/postprocess callbacks)
        def sim_preprocess(result):
            # Convert record+interpretation into simulator input
            return result

        sim_pipe = process_wrapper(
            simulator,
            preprocess=sim_preprocess,
            postprocess=lambda x: save_score(
                x, model_str, 'OpenAISimulator')
        )

        # 4. Build and run the pipeline
        pipeline = Pipeline(
            dataset,      # loads feature records & contexts
            sim_pipe          # runs simulation scoring in one stage
        )

    # Reduce concurrency to prevent memory issues
    # With the 32B model, we need to be very conservative with parallel processing
    max_concurrent = 3  # Process one at a time to avoid memory pressure

    asyncio.run(pipeline.run(max_concurrent=max_concurrent))

    print(
        f"✅ Pipeline completed with max_concurrent={max_concurrent} (memory-safe)")

    # Generate summary after analysis
    print(f"\n{'='*60}")
    print("ENHANCED INTERPRETABILITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Analyzed {len(priority_latents)} most interpretable latents")
    print(f"Model identifier: {model_str}")
    print(f"Results saved to:")
    print(
        f"  - Explanations: autointerp/{model_str}/explanations/enhanced_default/")
    print(
        f"  - Detection scores: autointerp/{model_str}/scores/enhanced_detection/")
    print(
        f"  - LLM cache: {llm_cache.cache_dir}")
    print(f"{'='*60}\n")


def ultrachat_to_flat_tokens(
    tokenizer,
    splits=("train_sft",),
    add_eos_between=True,
    eos_id=None,
    num_proc=None,
    render_batch_size=256,
    encode_batch_size=1024,
    token_budget=None,  # int or None
) -> torch.Tensor:
    """
    Returns a 1-D torch.LongTensor of token ids.
    No NumPy, no per-example skipping.
    """
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    ds = load_ultrachat()

    # Step A: repair + render chat template to strings (fast, parallel)
    def render_batch(batch):
        texts = []
        for msgs in batch["messages"]:
            fixed, need_gen = normalize_ultrachat_messages(msgs)
            txt = tokenizer.apply_chat_template(
                fixed,
                tokenize=False,                 # render string only
                add_generation_prompt=need_gen
            )
            texts.append(txt)
        return {"text": texts}

    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])
    print("Rendering chat templates to strings...")
    print(ds[0])
    ds = ds.map(
        render_batch,
        batched=True,
        batch_size=render_batch_size,
        num_proc=(num_proc or os.cpu_count() or 4),
        desc="Render chat templates",
    )
    texts = ds["text"]

    # Step B: batch-encode strings → lists of ids, then to torch and cat once
    pieces = []
    total = 0
    for i in tqdm(range(0, len(texts), encode_batch_size)):
        chunk = texts[i:i + encode_batch_size]
        enc = tokenizer(
            chunk,
            add_special_tokens=False,   # template already added specials
            return_attention_mask=False,
            padding=False,
            truncation=False,
            return_tensors=None,        # get List[List[int]]
        )["input_ids"]

        for ids in enc:
            if add_eos_between and eos_id is not None:
                ids = ids + [eos_id]
            t = torch.tensor(ids, dtype=torch.long)
            if token_budget is None:
                pieces.append(t)
            else:
                if total >= token_budget:
                    break
                need = token_budget - total
                if t.numel() <= need:
                    pieces.append(t)
                    total += t.numel()
                else:
                    pieces.append(t[:need])
                    total += need
                    break
        if token_budget is not None and total >= token_budget:
            break

    if not pieces:
        return torch.empty(0, dtype=torch.long)

    return torch.cat(pieces, dim=0)  # 1-D long tensor


def pack_1d_stream(tokens_1d: torch.Tensor, seq_len: int) -> torch.Tensor:
    usable = (tokens_1d.numel() // seq_len) * seq_len
    if usable == 0:
        raise ValueError("Not enough tokens to form a single window.")
    return tokens_1d.narrow(0, 0, usable).view(-1, seq_len)


def dpo_dataset_to_flat_tokens(
    tokenizer,
    add_eos_between=True,
    eos_id=None,
    num_proc=None,
    encode_batch_size=1024,
    token_budget=None,
) -> torch.Tensor:
    """
    Load DPO dataset using the EXACT same prepare_hh_rlhf_datasets function from dpo.py.
    Converts prompt+chosen and prompt+rejected pairs to flat 1-D token tensor.
    """
    from src.dpo import prepare_hh_rlhf_datasets

    if eos_id is None:
        eos_id = tokenizer.eos_token_id

    print("Loading DPO dataset using prepare_hh_rlhf_datasets from dpo.py...")
    # Use the exact same function as DPO training
    try:
        train_dataset, _ = prepare_hh_rlhf_datasets(
            max_length=2048,  # Large enough to not filter out examples
            tokenizer=tokenizer,
            max_prompt_length=1024,
            max_completion_length=1024,
        )
    except Exception as e:
        print(f"ERROR: Failed to load DPO dataset: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(
        f"Loaded {len(train_dataset)} DPO examples (prompt + chosen/rejected pairs)")

    if len(train_dataset) == 0:
        raise RuntimeError(
            "DPO dataset is empty after prepare_hh_rlhf_datasets filtering!")

    # Each example has: prompt, chosen, rejected
    # We'll tokenize both prompt+chosen and prompt+rejected
    texts = []
    for example in train_dataset:
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Concatenate prompt with chosen response
        texts.append(prompt + chosen)
        # Concatenate prompt with rejected response
        texts.append(prompt + rejected)

    print(
        f"Created {len(texts)} text sequences from DPO dataset (2x examples for chosen+rejected)")

    if len(texts) == 0:
        raise RuntimeError("No text sequences created from DPO dataset!")

    # Tokenize and concatenate (same pattern as ultrachat)
    pieces = []
    total = 0
    for i in tqdm(range(0, len(texts), encode_batch_size), desc="Tokenizing DPO dataset"):
        chunk = texts[i:i + encode_batch_size]
        enc = tokenizer(
            chunk,
            add_special_tokens=False,
            return_attention_mask=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )["input_ids"]

        for ids in enc:
            if add_eos_between and eos_id is not None:
                ids = ids + [eos_id]
            t = torch.tensor(ids, dtype=torch.long)
            if token_budget is None:
                pieces.append(t)
            else:
                if total >= token_budget:
                    break
                need = token_budget - total
                if t.numel() <= need:
                    pieces.append(t)
                    total += t.numel()
                else:
                    pieces.append(t[:need])
                    total += need
                    break
        if token_budget is not None and total >= token_budget:
            break

    if not pieces:
        raise RuntimeError(
            "DPO dataset produced no token pieces after tokenization!")

    result = torch.cat(pieces, dim=0)
    print(f"DPO dataset: Created {result.numel():,} tokens")

    if result.numel() == 0:
        raise RuntimeError("DPO dataset produced zero tokens!")

    return result


def delphi_collect_activations_causal(cfg, model, tokenizer, wrapped_modules):
    print("Starting SEMI-CAUSAL activation collection")

    SEQ_LEN = cfg.evals.causal_auto_interp.seq_len
    N_TOKENS_TARGET = cfg.evals.causal_auto_interp.n_tokens

    # Determine which dataset to use based on model type
    model_type = getattr(cfg.model, 'type', 'sft_model')

    # sometimes I also add the layer number to model_type
    # for backward compatibility just check
    # if string contains 'dpo_model'
    if 'dpo_model' in model_type:
        print(
            f"Model type is '{model_type}' - using DPO dataset (Anthropic/hh-rlhf)")
        tokens_1d = dpo_dataset_to_flat_tokens(
            tokenizer,
            add_eos_between=True,
            token_budget=N_TOKENS_TARGET,
            num_proc=os.cpu_count(),
            encode_batch_size=1024,
            eos_id=tokenizer.eos_token_id
        )
    else:
        print(f"Model type is '{model_type}' - using SFT dataset (UltraChat)")
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
        raise RuntimeError(
            f"Dataset ({model_type}) produced no tokens after loading!")

    print(
        f"Successfully loaded {tokens_1d.numel():,} tokens from {model_type} dataset")

    tokens_array = pack_1d_stream(
        tokens_1d, seq_len=SEQ_LEN)  # shape [N, SEQ_LEN]

    print(
        f"Packed into {tokens_array.shape[0]:,} sequences of length {SEQ_LEN}")

    topk_modules = {f"{name}.topk": module.topk for name,
                    module in wrapped_modules.items()}

    print(f"Setting up LatentCache for {len(topk_modules)} TopK modules:")
    for name in topk_modules:
        print(f"  - {name}")

    original_modes = {}
    for module in wrapped_modules.values():
        if hasattr(module, "is_topk_experiment"):
            original_modes[module] = module.is_topk_experiment
            module.is_topk_experiment = True

    cache = LatentCache(
        model=model,
        hookpoint_to_sparse_encode=topk_modules,
        batch_size=cfg.evals.causal_auto_interp.batch_size,
        transcode=False,
    )

    print(f"Running cache collection on {tokens_array.shape[0]} sequences...")
    try:
        cache.run(
            n_tokens=N_TOKENS_TARGET,
            tokens=tokens_array,
        )

        print(f"Cache collection complete. Checking cache contents...")
        total_entries = 0
        for hookpoint, locations in cache.cache.latent_locations.items():
            num_entries = int(
                locations.shape[0]) if locations is not None else 0
            total_entries += num_entries
            print(f"  {hookpoint}: {num_entries} non-zero activations")
        if total_entries == 0:
            print("WARNING: No latent activations were recorded.")

        out_dir = Path(
            f"cache/{model_type}/layer18/{cfg.evals.causal_auto_interp.r}_{cfg.evals.causal_auto_interp.k}")
        out_dir.mkdir(parents=True, exist_ok=True)
        cache.save_splits(n_splits=8, save_dir=out_dir)

        widths = {
            f"{name}.topk": wrapped_modules[name].r for name in wrapped_modules}
        for hookpoint in widths:
            hp_dir = out_dir / hookpoint
            hp_dir.mkdir(parents=True, exist_ok=True)
            with open(hp_dir / "config.json", "w") as f:
                json.dump({"hookpoint": hookpoint,
                          "width": widths[hookpoint]}, f)
    finally:
        for module, original in original_modes.items():
            module.is_topk_experiment = original

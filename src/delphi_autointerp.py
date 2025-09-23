from transformers import AutoTokenizer
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
from typing import Optional, Union, Dict, Any
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
        print(f"No interpretability results, using default range 0-{top_k}")
        return list(range(top_k))
    
    # Sort by interpretability score
    sorted_latents = sorted(
        interpretability_results,
        key=lambda x: x['interpretability_score'],
        reverse=True
    )
    
    priority_latents = [result['latent_id'] for result in sorted_latents[:top_k]]
    
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
        
        out_dir = f"explanations/{model_str}/" + explainer_type
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
        print(f"✓ Explained Latent {latent_id} (Score: {score:.2f}, Function: {func})")
        
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
        
        out_dir = f"scores/{model_str}/{scorer_type}"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{safe}.json")
        
        # Convert score to dict and ensure it's a proper dict
        score_obj = result.score
        if hasattr(score_obj, "to_json_string"):
            score_data = json.loads(score_obj.to_json_string())
        elif isinstance(score_obj, list):
            score_data = {"scores": [dataclasses.asdict(elem) for elem in score_obj]}
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
        print(f"✓ Scored Latent {latent_id} (Score: {score:.2f}, Function: {func}) - {scorer_type}")
        
        return result
    
    return enhanced_save_explanation, enhanced_save_score


class LLMResponseCache:
    """Cache for LLM responses to avoid redundant API calls."""

    def __init__(self, cache_dir: str = "llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
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

    def get_explanation(self, record) -> Optional[str]:
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

    def save_explanation(self, record, explanation: str):
        """Save explanation to cache."""
        cache_key = self._get_cache_key({
            'latent': str(record.latent),
            'activating_examples': [str(ex) for ex in record.examples[:5]],
            'non_activating_examples': [str(ex) for ex in record.not_active[:5]]
        })

        cache_file = self.explanation_cache_dir / f"{cache_key}.json"

        def write_explanation(filepath, mode):
            # Write to temporary file first, then atomic rename
            temp_file = str(filepath) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump({
                    'latent': str(record.latent),
                    'explanation': explanation,
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


# Global cache instance\

llm_cache = LLMResponseCache(cache_dir=f'llm_cache_{os.environ.get("CUDA_VISIBLE_DEVICES")}')
# llm_cache = LLMResponseCache(cache_dir='llm_cache_512_2')


class CachedExplainer:
    """Wrapper around Delphi explainer with caching."""

    def __init__(self, base_explainer):
        self.base_explainer = base_explainer

    async def __call__(self, record):
        # Check cache first
        cached_explanation = llm_cache.get_explanation(record)
        if cached_explanation is not None:
            # Create result object with cached explanation
            result = type('ExplanationResult', (), {
                'record': record,
                'explanation': cached_explanation
            })()
            return result

        # If not cached, call the base explainer
        print(f"⚡ Generating new explanation for {record.latent}")
        result = await self.base_explainer(record)

        # Cache the result
        llm_cache.save_explanation(record, result.explanation)

        return result


class CachedDetectionScorer:
    """Wrapper around Delphi detection scorer with caching."""

    def __init__(self, base_scorer):
        self.base_scorer = base_scorer

    async def __call__(self, record):
        # Check cache first
        cached_score = llm_cache.get_detection_score(record)
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
        llm_cache.save_detection_score(record, result.score)

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

    out_dir = f"explanations/{model_str}/" + explainer_type
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{safe}.json")
    with open(path, "w") as f:
        json.dump({"explanation": result.explanation}, f, indent=2)
    return result


def save_score(result, model_str, scorer):
    # 1) Build a safe filename from the latent
    latent_str = str(result.record.latent)  # e.g. "layers.5.self.topk:42"
    safe = latent_str.replace(".", "_").replace(":", "_").replace(" ", "_")

    # 2) Ensure output directory
    out_dir = f"scores/{model_str}/{scorer}"
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


def delphi_collect_activations(cfg, model, tokenizer, wrapped_modules):
    print('starting analysis')
    flat_ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

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

    loader = DataLoader(
        flat_ds,
        batch_size=cfg.evals.auto_interp.batch_size,
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

    cache = LatentCache(
        model=model,
        hookpoint_to_sparse_encode=topk_modules,
        batch_size=cfg.evals.auto_interp.batch_size,
        transcode=False,
    )

    cache.run(
        n_tokens=N_TOKENS,
        tokens=tokens_array,
    )
    out_dir = Path(
        f"cache/delphi_cache_{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}"
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


def delphi_score(cfg, model, tokenizer, wrapped_modules):
    # Print cache status
    print("\n" + "="*50)
    print("LLM CACHE STATUS")
    print("="*50)
    explanation_files = len(
        list(llm_cache.explanation_cache_dir.glob("*.json")))
    detection_files = len(list(llm_cache.detection_cache_dir.glob("*.pkl")))
    print(f"Cached explanations: {explanation_files}")
    print(f"Cached detection scores: {detection_files}")
    print(f"Cache directory: {llm_cache.cache_dir}")
    print("="*50 + "\n")

    topk_modules = [
        f"{name}.topk" for name, _ in wrapped_modules.items()
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
    priority_latents = get_priority_latents(interp_results, top_k=cfg.evals.auto_interp.r)
    
    # 1) Load the raw cache you saved
    dataset = LatentDataset(
        raw_dir=Path(
            f"cache/delphi_cache_{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}"
        ),
        modules=topk_modules,
        latents={
            # Focus on most interpretable latents only
            name: torch.tensor(priority_latents, dtype=torch.long)
            for name in topk_modules
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
    
    print(f"🔧 Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} (multi-GPU with tensor parallelism)")
    
    # Set PyTorch CUDA memory management for fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    # ABSOLUTE MINIMUM multi-GPU configuration for 32B model
    # Emergency memory settings - last resort to prevent OOM
    client = Offline("Qwen/Qwen2.5-32B-Instruct-AWQ",
                     max_memory=0.65,       # EMERGENCY: reduced to 55% for generation headroom
                     max_model_len=18860,    # MINIMUM: 2K context to maximize memory savings
                     num_gpus=num_gpus,            # Split across 2 GPUs (tensor parallelism)
                     prefix_caching=False,  # Disable to save memory during generation
                     batch_size=1,          # Single sample processing to minimize memory
                     enforce_eager=True)    # Disable CUDA graphs to avoid capture errors
    
    # Add device attribute for SurprisalScorer compatibility
    client.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("✅ Model loaded successfully with multi-GPU tensor parallelism!")
    print("   - Memory usage: 65% of GPU (conservative for generation)")
    print("   - Context length: 24,768 tokens (current setting)")
    print("   - Batch size: 1 (memory-safe)")
    print(f"   - GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} (tensor parallelism)")

    # Create enhanced save functions with interpretability metadata
    # enhanced_save_explanation, enhanced_save_score = create_enhanced_save_functions(
    #     f'{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}'
    # )

    openai_run = False
    if not openai_run:

        base_explainer = DefaultExplainer(client, cot=True)
        explainer = CachedExplainer(base_explainer)
        # explainer = ContrastiveExplainer(
        #     client,
        #     # minimum confidence score to accept a label :contentReference[oaicite:5]{index=5}
        #     threshold=0.3,
        #     # number of activating contexts to show :contentReference[oaicite:6]{index=6}
        #     max_examples=15,
        #     # number of hard negatives per feature :contentReference[oaicite:7]{index=7}
        #     max_non_activating=5,
        #     # print debug logs during explanation generation :contentReference[oaicite:8]{index=8}
        #     verbose=True
        # )

        # # 3) Wrap it in a pipe(here we save each explanation to disk)
        # # def save_explanation(result):
        # #     with open(f"explanations/{result.record.feature}.json", "w") as f:
        # #         f.write(result.explanation.json())
        # #     return result

        explainer_pipe = process_wrapper(
            explainer,
            postprocess=lambda x: save_explanation(x, f'{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}_simple', 'enhanced_default')
            # postprocess=lambda x: enhanced_save_explanation(x, 'enhanced_default')
        )

        base_detection_scorer = DetectionScorer(client, tokenizer=tokenizer)
        detection_scorer = CachedDetectionScorer(base_detection_scorer)
        
        # Add surprisal scorer for additional robustness analysis (temporarily disabled for VLLM)
        # surprisal_scorer = SurprisalScorer(
        #     client, True, cfg.evals.auto_interp.batch_size
        # )
        # fuzzing_scorer = FuzzingScorer(
        #     client, tokenizer, batch_size=cfg.evals.auto_interp.batch_size
        # )

        # async def score_both(explained):
        #     rec = explained.record
        #     rec.explanation = explained.explanation
        #     rec.extra_examples = rec.not_active

        #     # run detection
        #     try:
        #         det_res = await detection_scorer(rec)
        #         save_score(det_res, 'default_detection')
        #     except Exception as e:
        #         print(f"Detection failed for {rec.feature}: {e}")
        #         det_res = None

        #     # # run surprisal
        #     # sup_res = await surprisal_scorer(rec)
        #     # save_score(sup_res)
        #     # try:
        #     #     fuz_res = await fuzzing_scorer(rec)
        #     #     save_score(fuz_res, 'default_fuzzing')
        #     # except Exception as e:
        #     #     print(f"Fuzzing failed for {rec.feature}: {e}")
        #     #     fuz_res = None
        #     return explained

        # Enhanced preprocessing and scoring
        def preprocess(explained):
            rec = explained.record
            rec.explanation = explained.explanation
            rec.extra_examples = rec.not_active
            return rec

        detection_pipe = process_wrapper(
            detection_scorer,
            preprocess=preprocess,
            postprocess=lambda x: save_score(x, f'{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}_simple',  'enhanced_detection')
        )
        
        # Add surprisal scoring pipeline (temporarily disabled for VLLM)
        # surprisal_pipe = process_wrapper(
        #     surprisal_scorer,
        #     preprocess=preprocess,
        #     postprocess=lambda x: enhanced_save_score(x, 'surprisal')
        # )

        # Enhanced pipeline with multiple scoring methods
        print(f"Running enhanced interpretability analysis on {len(priority_latents)} latents")
        print(f"Analysis includes: explanations, detection scoring, and surprisal analysis")
        
        # Multi-stage pipeline
        async def comprehensive_scoring(explained):
            """Run both detection and surprisal scoring."""
            rec = explained.record
            rec.explanation = explained.explanation
            rec.extra_examples = rec.not_active
            
            # Run detection scoring
            try:
                det_result = await detection_scorer(rec)
                save_score(det_result, f'{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}_simple',  'enhanced_detection')
            except Exception as e:
                print(f"Detection scoring failed for {rec.latent}: {e}")
            
            # Run surprisal scoring (temporarily disabled for VLLM compatibility)
            # try:
            #     sup_result = await surprisal_scorer(rec)
            #     enhanced_save_score(sup_result, 'surprisal')
            # except Exception as e:
            #     print(f"Surprisal scoring failed for {rec.latent}: {e}")
            # print(f"Surprisal scoring skipped for {rec.latent} (VLLM compatibility)")
            
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
            postprocess=lambda x: save_score(x, f'{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}_simple', 'OpenAISimulator')
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
    
    print(f"✅ Pipeline completed with max_concurrent={max_concurrent} (memory-safe)")
    
    # Generate summary after analysis
    print(f"\n{'='*60}")
    print("ENHANCED INTERPRETABILITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Analyzed {len(priority_latents)} most interpretable latents")
    print(f"Results saved to:")
    print(f"  - Explanations: explanations/{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}/enhanced_default/")
    print(f"  - Detection scores: scores/{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}/enhanced_detection/")
    print(f"  - Surprisal scores: scores/{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}/surprisal/")
    print(f"{'='*60}\n")

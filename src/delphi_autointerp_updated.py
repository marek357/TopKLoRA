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

device = "cuda" if torch.cuda.is_available() else "cpu"


class LLMResponseCache:
    """Cache for LLM prompt-response pairs to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.responses_cache_dir = self.cache_dir / "responses"
        self.responses_cache_dir.mkdir(exist_ok=True)
        
    def _get_prompt_hash(self, messages, model=None, temperature=None) -> str:
        """Generate a hash key for the prompt parameters."""
        # Create a deterministic representation of the prompt
        prompt_data = {
            'messages': messages if isinstance(messages, str) else str(messages),
            'model': model or 'default',
            'temperature': temperature or 0.0
        }
        
        # Sort for consistency
        cache_string = str(sorted(prompt_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
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

    def get_response(self, messages, model=None, temperature=None) -> Optional[str]:
        """Get cached response or return None if not found."""
        cache_key = self._get_prompt_hash(messages, model, temperature)
        cache_file = self.responses_cache_dir / f"{cache_key}.json"
        
        def read_response(filepath, mode):
            if not filepath.exists():
                return None
            with open(filepath, 'r') as f:
                data = json.load(f)
                print(f"✓ Found cached response for prompt hash {cache_key[:8]}...")
                return data['response']
        
        try:
            result = self._safe_file_operation(cache_file, read_response, 'r')
            return result
        except Exception as e:
            print(f"Warning: Failed to load response cache {cache_file}: {e}")
            return None

    def save_response(self, messages, response: str, model=None, temperature=None):
        """Save response to cache."""
        cache_key = self._get_prompt_hash(messages, model, temperature)
        cache_file = self.responses_cache_dir / f"{cache_key}.json"
        
        def write_response(filepath, mode):
            # Write to temporary file first, then atomic rename
            temp_file = str(filepath) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump({
                    'prompt_hash': cache_key,
                    'messages': messages if isinstance(messages, str) else str(messages)[:200] + "..." if len(str(messages)) > 200 else str(messages),
                    'model': model or 'default',
                    'temperature': temperature or 0.0,
                    'response': response,
                    'timestamp': str(torch.tensor(0).item())
                }, f, indent=2)
            
            # Atomic rename
            os.rename(temp_file, filepath)
            print(f"✓ Cached response for prompt hash {cache_key[:8]}...")
            return True
        
        try:
            result = self._safe_file_operation(cache_file, write_response, 'w')
        except Exception as e:
            print(f"Warning: Failed to save response cache: {e}")

    def clear_cache(self):
        """Clear all cached responses."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.responses_cache_dir.mkdir(exist_ok=True)
            print("✓ Cache cleared")

    def get_cache_stats(self):
        """Get cache statistics."""
        try:
            response_count = len(list(self.responses_cache_dir.glob("*.json")))
            return {
                'response_count': response_count,
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {
                'response_count': 0,
                'cache_dir': str(self.cache_dir)
            }


# Global cache instance
llm_cache = LLMResponseCache()


class CachedOpenRouterClient:
    """Wrapper around OpenRouter client with caching."""
    
    def __init__(self, base_client):
        self.base_client = base_client
    
    async def chat_completion(self, *args, **kwargs):
        """Cached version of chat completion."""
        
        # Extract key parameters for caching
        messages = kwargs.get('messages', args[0] if args else None)
        model = kwargs.get('model', getattr(self.base_client, 'model', None))
        temperature = kwargs.get('temperature', 0.0)
        
        # Check cache first
        cached_response = llm_cache.get_response(messages, model, temperature)
        if cached_response is not None:
            # Create a mock response object similar to OpenAI API
            class MockResponse:
                def __init__(self, content):
                    self.choices = [type('Choice', (), {
                        'message': type('Message', (), {'content': content})()
                    })()]
            
            return MockResponse(cached_response)
        
        # If not cached, call the base client
        print(f"⚡ Making new LLM API call...")
        result = await self.base_client.chat_completion(*args, **kwargs)
        
        # Extract and cache the response
        if hasattr(result, 'choices') and len(result.choices) > 0:
            response_content = result.choices[0].message.content
            llm_cache.save_response(messages, response_content, model, temperature)
        
        return result
    
    def __getattr__(self, name):
        """Delegate all other attributes to the base client."""
        return getattr(self.base_client, name)


# Global cache instance
llm_cache = LLMResponseCache()


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


def delphi_analysiss(cfg, model, tokenizer, wrapped_modules):
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
    # 1) Load the raw cache you saved
    dataset = LatentDataset(
        raw_dir=Path(
            f"cache/delphi_cache_{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}"
        ),
        modules=topk_modules,
        latents={
            # NOTE: replace 1024 with a parameter
            name: torch.arange(0, cfg.evals.auto_interp.r, dtype=torch.long)
            for name in topk_modules
        },
        tokenizer=tokenizer,
        sampler_cfg=SamplerConfig(),          # use defaults or tweak
        constructor_cfg=ConstructorConfig(
            # # trigger ContrastiveExplainer
            # non_activating_source="FAISS",
            # faiss_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            # faiss_embedding_cache_enabled=True,
            # faiss_embedding_cache_dir=".embedding_cache"
        ),     # e.g. FAISS negatives or not
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
        # Wrap the OpenRouter client with caching
    base_client = OpenRouter(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model="qwen/qwen-2.5-32b-instruct-awq"
    )
    
    # client = Offline("Qwen/Qwen2.5-32B-Instruct-AWQ",
    #                  device_map="auto",
    #                  sampler_config=sampler_config)

    if True:
        # Wrap client with caching
        client = CachedOpenRouterClient(base_client)
        base_explainer = DefaultExplainer(client, cot=True)
    else:
        base_explainer = ContrastiveExplainer(
            base_client,
            neg_threshold=0.3,
            include_activations=False,
            randomize_order=True,
        )

    # 3) Build your scorer(s)
    if True:
        # Wrap client with caching for detection scorer as well
        client_for_scorer = CachedOpenRouterClient(base_client)
        base_detection_scorer = DetectionScorer(client_for_scorer, tokenizer=tokenizer)
        detection_scorer = CachedDetectionScorer(base_detection_scorer)
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

        # score_pipe = process_wrapper(score_both)
        def preprocess(explained):
            rec = explained.record
            rec.explanation = explained.explanation
            rec.extra_examples = rec.not_active

            return rec

        detection_pipe = process_wrapper(
            detection_scorer,
            preprocess=preprocess,
            postprocess=lambda x: save_score(
                x, f'{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}',  'default_detection')
        )

        # def save_score(result):
        #     with open(f"scores/{result.record.feature}.json", "w") as f:
        #         f.write(result.score.json())
        #     return result

        # 5) Run the full pipeline
        pipeline = Pipeline(
            dataset,
            explainer_pipe,
            detection_pipe
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
            postprocess=lambda x: save_score(x, 'oai_autointerp_simulation')
        )                                     # :contentReference[oaicite:5]{index=5}

        # 4. Build and run the pipeline
        pipeline = Pipeline(
            dataset,      # loads feature records & contexts
            sim_pipe          # runs simulation scoring in one stage
        )

    asyncio.run(pipeline.run(max_concurrent=4))

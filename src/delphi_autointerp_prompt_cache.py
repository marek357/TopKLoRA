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

    def save_response(self, messages, response: str, model=None, temperature=None):
        """Save response to cache."""
        cache_key = self._get_prompt_hash(messages, model, temperature)
        cache_file = self.responses_cache_dir / f"{cache_key}.json"
        
        def write_response(filepath, mode):
            # Write to temporary file first, then atomic rename
            temp_file = str(filepath) + '.tmp'
            print(f"Debug: Writing to temp file {temp_file}")
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
            print(f"Debug: Renaming {temp_file} to {filepath}")
            os.rename(temp_file, filepath)
            print(f"✓ Cached response for prompt hash {cache_key[:8]}... to {filepath}")
            return True
        
        try:
            result = self._safe_file_operation(cache_file, write_response, 'w')
            print(f"Debug: Cache save result: {result}")
        except Exception as e:
            print(f"Warning: Failed to save response cache: {e}")
            import traceback
            traceback.print_exc()

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
            print(f"Debug: Found {response_count} files in {self.responses_cache_dir}")
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


def save_score(result, exp_id, type='default'):
    # Get the structured score information
    latent_str = str(result.record.latent)

    # Create directory based on experiment ID
    score_dir = Path(f"scores/{exp_id}")
    score_dir.mkdir(parents=True, exist_ok=True)

    # Save with appropriate filename
    filename = f"{latent_str}.json"
    save_path = score_dir / filename

    try:
        # Convert score to dictionary if it has a model_dump method
        if hasattr(result.score, 'model_dump'):
            score_data = result.score.model_dump()
        else:
            # Fallback to direct serialization
            score_data = result.score

        with open(save_path, "w") as f:
            json.dump(score_data, f, indent=2)

        print(f"Saved score to: {save_path}")

    except Exception as e:
        print(f"Failed to save score to {save_path}: {e}")

    return result


async def delphi_score(cfg, model, tokenizer, wrapped_modules):
    print(f"Loading latent data from disk...")

    # load from disk
    cache = LatentCache()
    cache_key = f"auto_interp_{cfg.model.r}_{cfg.model.k}_{cfg.model.train_steps}_{cfg.model.adapter_checkpoint_dir}"
    if cfg.evals.auto_interp.ablation_study or cfg.evals.auto_interp.activations_study:
        cache_key += "_ablations"

    # Cache lookup
    print(f"Looking for cache key: {cache_key}")
    if cache.exists(cache_key):
        print(f"Loading from cache: {cache_key}")
        flat_ds = cache.get(cache_key)
    else:
        print(f"Cache not found. This means the activation data is not ready yet.")
        print(f"Please run the eval with activations_study=True first to generate the data.")
        return

    print(f"Loaded {len(flat_ds)} latent records")

    # Transform into LatentDataset
    from delphi.latents import LatentRecord

    print("Converting to LatentDataset...")

    # Convert flat_ds to proper LatentRecord objects
    latent_records = []
    for item in flat_ds:
        # Create proper LatentRecord with the expected structure
        record = LatentRecord(
            latent=item['latent'],
            examples=item.get('examples', []),
            not_active=item.get('not_active', []),
            explanation=item.get('explanation', ''),
        )
        latent_records.append(record)

    print(f"Created {len(latent_records)} LatentRecord objects")

    # Create the dataset (as a simple list, since Pipeline expects an iterable)
    dataset = latent_records

    print("Dataset created, setting up pipeline...")

    # Display cache status
    stats = llm_cache.get_cache_stats()
    print("=" * 50)
    print("LLM CACHE STATUS")
    print("=" * 50)
    print(f"Cached responses: {stats['response_count']}")
    print(f"Cache directory: {stats['cache_dir']}")
    print("=" * 50)

    sampler_config = SamplerConfig()

    # 2) Build your explainer client + explainer
    # class OpenRouter(Client):
    #     def __init__(
    #         self,
    #         api_key: str,
    #         model: str = "qwen/qwen-2.5-72b-instruct",
    #         base_url="https://openrouter.ai/api/v1/chat/completions",
    #         timeout: int = 60,
    #     ):

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
            client,
            neg_threshold=0.3,
            include_activations=False,
            randomize_order=True,
        )

    # 3) Build your scorer(s)
    if True:
        # Wrap client with caching for detection scorer as well
        client_for_scorer = CachedOpenRouterClient(base_client)
        base_detection_scorer = DetectionScorer(client_for_scorer, tokenizer=tokenizer)
    else:
        surprisal_scorer = SurprisalScorer(
            client, True, cfg.evals.auto_interp.batch_size
        )
        fuzzing_scorer = FuzzingScorer(
            client, tokenizer, batch_size=cfg.evals.auto_interp.batch_size
        )
        oai_autointerp_simulator = OpenAISimulator(
            client,
            postprocess=lambda x: save_score(x, 'oai_autointerp_simulation')
        )

    # 4) Set up the pipeline processors

    # async def explain_all(records):
    #     for record in records:
    #         explanation = await explainer(record)
    #         record.explanation = explanation.explanation
    #     return records

    explainer_pipe = process_wrapper(base_explainer)

    # async def score_one(explained):
    #     rec = explained.record
    #     rec.explanation = explained.explanation

    #     # res = await surprisal_scorer(rec)
    #     det_res = await detection_scorer(rec)

    #     # scores = {"surprisal": res.score, "detection": det_res.score}
    #     return rec

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
        base_detection_scorer,
        preprocess=preprocess,
        postprocess=lambda x: save_score(
            x, f'{cfg.evals.auto_interp.r}_{cfg.evals.auto_interp.k}', 'default_detection')
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

    await pipeline.run(max_concurrent=4)


if __name__ == "__main__":
    from src.evals import init_model_tokenizer

    from omegaconf import DictConfig

    # Quick test config
    cfg = DictConfig({
        'model': {
            'r': 1024,
            'k': 8,
            'train_steps': 7500,
            'adapter_checkpoint_dir': '/scratch/network/ssd/marek/lora_interp/models/dpo/google/gemma-2-2b/google-gemma-2-2b_topk_dpo_20250813_110632_3030316f/final_adapter'
        },
        'evals': {
            'auto_interp': {
                'r': 1024,
                'k': 8,
                'ablation_study': False,
                'activations_study': True
            }
        }
    })

    print("Initializing model...")
    model, tokenizer, wrapped_modules = init_model_tokenizer(cfg, auto_interp=True)

    print("Running Delphi pipeline...")
    asyncio.run(delphi_score(cfg, model, tokenizer, wrapped_modules))

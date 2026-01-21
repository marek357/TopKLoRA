import asyncio
import dataclasses
import json
import os
import sys
from itertools import islice
from pathlib import Path

import torch
from datasets import load_dataset
from delphi.clients import Offline, OpenRoute
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.explainers import ContrastiveExplainer, DefaultExplainer
from delphi.latents import LatentCache, LatentDataset
from delphi.pipeline import Pipeline, process_wrapper
from delphi.scorers import (
    DetectionScorer,
    FuzzingScorer,
    OpenAISimulator,
    SurprisalScorer,
)
from torch.utils.data import DataLoader

# Add path for our improvements
sys.path.append('/scratch/network/ssd/marek/lora_interp/src')

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    # TODO: set the dirs through config
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
    # TODO: set the dirs through config
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


def delphi_collect_activations(cfg, model, tokenizer, wrapped_modules):
    print('starting activation collection')

    if not hasattr(cfg, "dataset_name"):
        raise ValueError("cfg must have 'dataset_name' attribute for activation collection")

    # TODO: Adapt to work with hh-rlhf dataset
    flat_ds = load_dataset(cfg.dataset_name, "en", split="train", streaming=True)

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

    topk_modules = [
        # filter out query projections -- these have already been analyzed
        f"{name}.topk" for name, _ in wrapped_modules.items() if 'q_proj' not in name
    ]
    print(topk_modules)
    topk_modules = [elem for elem in topk_modules if '18' in elem]
    print(f"Filtered to layer 18 modules: {topk_modules}")
    # assert False, "Debug stop"
    model.cpu()
    del model
    del wrapped_modules

    # Load interpretability rankings and get priority latents
    print("\n" + "="*60)
    print("ENHANCED INTERPRETABILITY-FOCUSED ANALYSIS")
    print("="*60)

    # TODO: Follow the methodology from causal-autointerp-hh
    interp_results = load_interpretability_rankings()


    # TODO: select most promising latents
    priority_latents = get_priority_latents(
        interp_results, top_k=config.r)
    

    model_type = getattr(cfg.model, 'type', 'sft_model')

    # 1) Load the raw cache you saved
    dataset = LatentDataset(
        raw_dir=Path(
            f"cache/{model_type}/layer_full/{config.r}_{config.k}"
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
        # TODO: Figure out how these may possibly improve explanations
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

        # TODO may have to modify to adapt for cot (not originally implemented)
        explainer = DefaultExplainer(client, cot=True)
        # explainer = CachedExplainer(
            # base_explainer, cache=llm_cache, tokenizer=tokenizer)
        explainer_pipe = process_wrapper(
            explainer,
            postprocess=lambda x: save_explanation(
                x, model_str, 'enhanced_default')
        )

        detection_scorer = DetectionScorer(
            client, tokenizer=tokenizer, n_examples_shown=5)
        # detection_scorer = CachedDetectionScorer(
        #     base_detection_scorer, cache=llm_cache)

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
    print(f"{'='*60}\n")

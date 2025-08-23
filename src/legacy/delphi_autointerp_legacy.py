from openai import OpenAI
from src.models import FixedTopKLoRALinear, TopKLoRALinear
# from delphi.__main__ import populate_cache, main as delphi_main
from delphi.latents import LatentDataset
from delphi.latents import LatentCache
from sparsify.data import chunk_and_tokenize
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
import numpy as np
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from safetensors.torch import save_file, load_file
from datasets import load_dataset
from transformers import AutoTokenizer
from neuron_explainer.activations.activations import ActivationRecord, NeuronRecord
from tqdm import tqdm
import asyncio
from functools import partial

# Delphi imports
from delphi.latents import LatentDataset, LatentCache
from delphi.config import SamplerConfig, ConstructorConfig
from delphi.explainers import DefaultExplainer, ContrastiveExplainer, explanation_loader
from delphi.scorers import FuzzingScorer, DetectionScorer
# from delphi.scorers import FuzzingScorer, RecallScorer, DetectionScorer
from delphi.clients import Offline, OpenRouter
from delphi.pipeline import Pipeline, process_wrapper, Pipe
import orjson


class NeuronRecordToDelphi:
    """Convert NeuronRecord format to Delphi's expected format."""

    def __init__(self, neuron_records_path: str, output_dir: str, module_name: str):
        self.neuron_records_path = neuron_records_path
        self.output_dir = Path(output_dir)
        self.module_name = module_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_delphi_format(self, max_examples_per_latent: int = 1000):
        """Convert NeuronRecord pickle files to Delphi safetensors format."""
        # Load the neuron records
        with open(self.neuron_records_path, 'rb') as f:
            neuron_records = pickle.load(f)

        # Prepare data structures for Delphi format
        all_activations = []
        all_locations = []
        all_tokens = []
        all_latent_ids = []

        print(
            f"Converting {len(neuron_records)} neuron records to Delphi format...")

        for latent_id, record in tqdm(neuron_records.items(), desc="Processing latents"):
            # Process most positive activation records
            examples = record.most_positive_activation_records[:max_examples_per_latent]

            for example_idx, activation_record in enumerate(examples):
                tokens = activation_record.tokens
                activations = activation_record.activations

                # For each token in the example
                for token_idx, (token, activation) in enumerate(zip(tokens, activations)):
                    if activation > 0:  # Only store non-zero activations
                        all_tokens.append(token)
                        all_activations.append(activation)
                        # [example_id, position]
                        all_locations.append([example_idx, token_idx])
                        all_latent_ids.append(latent_id)

        # Convert to tensors
        activations_tensor = torch.tensor(all_activations, dtype=torch.float32)
        locations_tensor = torch.tensor(all_locations, dtype=torch.int64)

        # Save in Delphi format - split by latent for efficiency
        unique_latents = sorted(set(all_latent_ids))
        n_splits = min(5, len(unique_latents))  # Use up to 5 splits
        latents_per_split = len(unique_latents) // n_splits

        for split_idx in range(n_splits):
            start_latent = split_idx * latents_per_split
            end_latent = (
                split_idx + 1) * latents_per_split if split_idx < n_splits - 1 else len(unique_latents)

            split_latents = unique_latents[start_latent:end_latent]

            # Filter data for this split
            split_mask = torch.tensor(
                [lid in split_latents for lid in all_latent_ids])

            split_data = {
                "activations": activations_tensor[split_mask],
                "locations": locations_tensor[split_mask],
                "tokens": [tok for tok, mask in zip(all_tokens, split_mask) if mask],
                "latent_ids": torch.tensor([lid for lid, mask in zip(all_latent_ids, split_mask) if mask])
            }

            # Save as safetensors
            output_path = self.output_dir / \
                f"{self.module_name}_split_{split_idx}.safetensors"

            # Convert non-tensor data to tensor format for saving
            tokens_encoded = torch.tensor(
                [hash(token) % 100000 for token in split_data["tokens"]],
                dtype=torch.int32
            )

            save_dict = {
                "activations": split_data["activations"],
                "locations": split_data["locations"],
                "tokens_hash": tokens_encoded,  # Save token hashes
                "latent_ids": split_data["latent_ids"]
            }

            save_file(save_dict, str(output_path))

            # Also save tokens separately as JSON for reference
            tokens_path = self.output_dir / \
                f"{self.module_name}_split_{split_idx}_tokens.json"
            with open(tokens_path, 'w') as f:
                json.dump(split_data["tokens"], f)

        print(f"Saved {n_splits} splits to {self.output_dir}")

        # Save metadata
        metadata = {
            "module_name": self.module_name,
            "n_latents": len(unique_latents),
            "n_splits": n_splits,
            "total_activations": len(all_activations)
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


class LoRALatentDataset(LatentDataset):
    """Custom LatentDataset for LoRA activations."""

    def __init__(
        self,
        raw_dir: str,
        module_name: str,
        sampler_cfg: SamplerConfig,
        constructor_cfg: ConstructorConfig,
        tokenizer: AutoTokenizer,
        latents: Optional[Dict[str, torch.Tensor]] = None
    ):
        # Initialize parent class with minimal requirements
        self.raw_dir = Path(raw_dir)
        self.module_name = module_name
        self.sampler_cfg = sampler_cfg
        self.constructor_cfg = constructor_cfg
        self.tokenizer = tokenizer
        self.latents = latents or {}

        # Load the converted data
        self._load_lora_data()

    def _load_lora_data(self):
        """Load the converted LoRA activation data."""
        metadata_path = self.raw_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.n_latents = metadata["n_latents"]
        self.n_splits = metadata["n_splits"]

        # Load all splits
        self.data = {
            "activations": [],
            "locations": [],
            "tokens": [],
            "latent_ids": []
        }

        for split_idx in range(self.n_splits):
            # Load safetensors
            split_path = self.raw_dir / \
                f"{self.module_name}_split_{split_idx}.safetensors"
            split_data = load_file(str(split_path))

            # Load tokens
            tokens_path = self.raw_dir / \
                f"{self.module_name}_split_{split_idx}_tokens.json"
            with open(tokens_path, 'r') as f:
                tokens = json.load(f)

            self.data["activations"].append(split_data["activations"])
            self.data["locations"].append(split_data["locations"])
            self.data["tokens"].extend(tokens)
            self.data["latent_ids"].append(split_data["latent_ids"])

        # Concatenate tensors
        self.data["activations"] = torch.cat(self.data["activations"])
        self.data["locations"] = torch.cat(self.data["locations"])
        self.data["latent_ids"] = torch.cat(self.data["latent_ids"])


def run_delphi_autointerp_pipeline(
    cfg,
    converted_data_dir: str,
    module_name: str,
    tokenizer: AutoTokenizer,
    explainer_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_openrouter: bool = False,
    openrouter_api_key: Optional[str] = None,
    max_latents: int = 100,
    n_processes: int = 10,
    output_dir: str = "delphi_outputs"
):
    """Run the full Delphi autointerp pipeline on converted LoRA activations."""

    # Create output directories
    output_path = Path(output_dir)
    explanation_dir = output_path / "explanations"
    recall_dir = output_path / "scores" / "recall"
    fuzz_dir = output_path / "scores" / "fuzzing"
    detection_dir = output_path / "scores" / "detection"

    for dir_path in [explanation_dir, recall_dir, fuzz_dir, detection_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Set up configuration
    sampler_cfg = SamplerConfig(
        train_set_size=5,  # Number of examples for generating explanations
        test_set_size=5,   # Number of examples for testing
        n_quantiles=10,
        n_random=3
    )

    constructor_cfg = ConstructorConfig(
        min_examples=3,
        max_examples=10,
        context_size=20,  # Tokens of context around activation
        non_activating_source="random",  # or "FAISS" for hard negatives
        min_token_activation=0.0
    )

    # Create latent dataset
    latent_dict = {
        module_name: torch.arange(0, max_latents)
    }

    dataset = LoRALatentDataset(
        raw_dir=converted_data_dir,
        module_name=module_name,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        tokenizer=tokenizer,
        latents=latent_dict
    )

    # Create data loader
    loader = dataset.load(n_latents=max_latents)

    # Set up client
    if use_openrouter:
        # client = OpenRouter(explainer_model, api_key=openrouter_api_key)
        client = OpenAI(
            api_key='API_KEY',
            base_url='http://localhost:8000/v1',
        )
    else:
        client = Offline(
            explainer_model,
            max_memory=0.8,
            max_model_len=5120,
            num_gpus=1
        )

    # Create explainer
    if constructor_cfg.non_activating_source == "FAISS":
        explainer = ContrastiveExplainer(
            client,
            threshold=0.3,
            max_examples=15,
            max_non_activating=5,
            verbose=True
        )
    else:
        explainer = DefaultExplainer(
            client,
            tokenizer=tokenizer,
            threshold=0.3,
            max_examples=10,
            verbose=True
        )

    # Create explainer pipeline
    def explainer_postprocess(result):
        with open(explanation_dir / f"{result.record.latent}.json", 'wb') as f:
            f.write(orjson.dumps({
                "latent_id": result.record.latent,
                "explanation": result.explanation,
                "examples": result.record.examples
            }))
        return result

    explainer_pipe = process_wrapper(
        explainer,
        postprocess=explainer_postprocess
    )

    # Create scorer pipeline
    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = getattr(record, 'not_active', [])
        return record

    def scorer_postprocess(result, score_dir):
        with open(score_dir / f"{result.record.latent}.json", 'wb') as f:
            f.write(orjson.dumps({
                "latent_id": result.record.latent,
                "score": result.score,
                "scorer_type": result.scorer_type if hasattr(result, 'scorer_type') else "unknown"
            }))
        return result

    # Create multiple scorers
    scorer_pipe = Pipe(
        # process_wrapper(
        #     RecallScorer(client, tokenizer=tokenizer,
        #                  batch_size=cfg.evals.auto_interp.batch_size),
        #     preprocess=scorer_preprocess,
        #     postprocess=partial(scorer_postprocess, score_dir=recall_dir)
        # ),
        process_wrapper(
            FuzzingScorer(client, tokenizer=tokenizer,
                          batch_size=cfg.evals.auto_interp.batch_size),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=fuzz_dir)
        ),
        process_wrapper(
            DetectionScorer(client, tokenizer=tokenizer,
                            batch_size=cfg.evals.auto_interp.batch_size),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=detection_dir)
        )
    )

    # Create and run pipeline
    pipeline = Pipeline(
        loader.load,
        explainer_pipe,
        scorer_pipe
    )

    print(f"Running Delphi autointerp pipeline for {max_latents} latents...")
    asyncio.run(pipeline.run(n_processes))

    print(f"Pipeline complete! Results saved to {output_dir}")

    # Generate summary statistics
    generate_summary_stats(output_path)


def generate_summary_stats(output_dir: Path):
    """Generate summary statistics from the autointerp results."""
    stats = {
        "explanations": 0,
        "recall_scores": [],
        "fuzzing_scores": [],
        "detection_scores": []
    }

    # Count explanations
    explanation_dir = output_dir / "explanations"
    stats["explanations"] = len(list(explanation_dir.glob("*.json")))

    # Collect scores
    for score_type in ["recall", "fuzzing", "detection"]:
        score_dir = output_dir / "scores" / score_type
        scores = []

        for score_file in score_dir.glob("*.json"):
            with open(score_file, 'rb') as f:
                data = orjson.loads(f.read())
                if isinstance(data.get("score"), (int, float)):
                    scores.append(data["score"])

        stats[f"{score_type}_scores"] = scores

    # Calculate summary statistics
    summary = {
        "n_explanations": stats["explanations"],
        "n_scored": len(stats["recall_scores"]),
        "recall": {
            "mean": np.mean(stats["recall_scores"]) if stats["recall_scores"] else 0,
            "std": np.std(stats["recall_scores"]) if stats["recall_scores"] else 0
        },
        "fuzzing": {
            "mean": np.mean(stats["fuzzing_scores"]) if stats["fuzzing_scores"] else 0,
            "std": np.std(stats["fuzzing_scores"]) if stats["fuzzing_scores"] else 0
        },
        "detection": {
            "mean": np.mean(stats["detection_scores"]) if stats["detection_scores"] else 0,
            "std": np.std(stats["detection_scores"]) if stats["detection_scores"] else 0
        }
    }

    # Save summary
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nSummary Statistics:")
    print(f"Explanations generated: {summary['n_explanations']}")
    print(f"Latents scored: {summary['n_scored']}")
    print(
        f"Recall Score: {summary['recall']['mean']:.3f} ± {summary['recall']['std']:.3f}")
    print(
        f"Fuzzing Score: {summary['fuzzing']['mean']:.3f} ± {summary['fuzzing']['std']:.3f}")
    print(
        f"Detection Score: {summary['detection']['mean']:.3f} ± {summary['detection']['std']:.3f}")


# Main execution function
def delphi_analysiss(cfg, model, tokenizer):
    """Main function to run the complete pipeline."""

    # Step 1: Run your existing activation collection (already done in your code)
    print("Step 1: Activation collection already completed")

    # Step 2: Convert NeuronRecords to Delphi format
    print("\nStep 2: Converting NeuronRecords to Delphi format...")

    # Add your module names
    module_list = [
        "model.layers.11.mlp.down_proj",
        "model.layers.11.mlp.gate_proj"
        "model.layers.11.mlp.up_proj",
        "model.layers.11.self_attn.k_proj",
        "model.layers.11.self_attn.q_proj",
        "model.layers.11.self_attn.o_proj",
        "model.layers.11.self_attn.v_proj",
    ]
    for module_name in module_list:
        converter = NeuronRecordToDelphi(
            neuron_records_path=f"cache/neuron_records/c4_fixed/{module_name}_neuron_record.pkl",
            output_dir=f"cache/delphi_format/{module_name}",
            module_name=module_name
        )
        converter.convert_to_delphi_format(max_examples_per_latent=100)

    # Step 3: Run Delphi autointerp pipeline
    print("\nStep 3: Running Delphi autointerp pipeline...")

    for module_name in module_list:  # Add your module names
        run_delphi_autointerp_pipeline(
            cfg=cfg,
            converted_data_dir=f"cache/delphi_format/{module_name}",
            module_name=module_name,
            tokenizer=tokenizer,
            explainer_model="Qwen/Qwen2.5-32B-Instruct-AWQ",  # Or use smaller model
            use_openrouter=True,  # Set to True to use API instead of local
            openrouter_api_key=None,  # Add your API key if using OpenRouter
            max_latents=100,  # Number of latents to interpret
            n_processes=10,  # Parallel processes
            output_dir=f"delphi_outputs/{module_name}"
        )

    print("\nAutointerp pipeline complete!")


# Usage with your existing code:
# if __name__ == "__main__":
#     # After running your analyse_model function:
#     # main(cfg, model, tokenizer)
#     pass


# Delphi imports

# Your custom modules


class LoRAToSAEWrapper(nn.Module):
    """
    Wrapper to make LoRA modules compatible with Delphi's SAE interface.
    This creates a pseudo-SAE that extracts the low-rank activations from LoRA.
    """

    def __init__(self, lora_module: TopKLoRALinear, module_name: str, k: int, r: int):
        super().__init__()
        self.lora_module = lora_module
        self.module_name = module_name
        self.k = k
        self.r = r
        self.n_latents = r  # Number of latents equals rank

        # Store the adapter name
        adapter = lora_module.lora_module.active_adapter
        if isinstance(adapter, (list, tuple)):
            adapter = adapter[0]
        self.adapter = adapter

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input activations to sparse latent representation.
        Returns: (latent_indices, latent_values)
        """
        # Get the low-rank activations from LoRA A matrix
        low_rank_acts = self.lora_module.lora_module.lora_A[self.adapter](x)
        # get the top-k batch-wise

        # TODO: change batch-topk to per-sequence top-k
        # Reshape to (batch * seq_len, r)
        batch_size, seq_len, hidden_dim = x.shape
        low_rank_acts = low_rank_acts.view(batch_size * seq_len, -1)

        # Get top-k activations
        topk_values, topk_indices = torch.topk(
            low_rank_acts,
            # low_rank_acts.abs(),
            k=min(self.k, self.r),
            dim=-1
        )

        # Create sparse representation
        # Shape: (batch * seq_len, k)
        return topk_indices, topk_values

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with Delphi's expectations.
        Returns dict with 'latent_indices' and 'latent_acts'.
        """
        latent_indices, latent_acts = self.encode(x)

        return {
            'latent_indices': latent_indices,
            'latent_acts': latent_acts,
            'raw_activations': self.lora_module.lora_module.lora_A[self.adapter](x)
        }


class DelphinLoRACache(LatentCache):
    """
    Custom LatentCache for LoRA modules that properly handles the activation collection.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        submodule_dict: Dict[str, LoRAToSAEWrapper],
        batch_size: int = 8,
        cfg=None
    ):
        """
        Initialize cache for LoRA modules.

        Args:
            model: The base language model
            submodule_dict: Dict mapping module names to LoRAToSAEWrapper instances
            batch_size: Batch size for processing
            cfg: Configuration object
        """
        self.model = model
        self.submodule_dict = submodule_dict
        self.batch_size = batch_size
        self.cfg = cfg
        self.device = next(model.parameters()).device

        # Storage for activations
        self.cached_data = {
            name: {
                'activations': [],
                'locations': [],
                'tokens': [],
                'latent_indices': []
            }
            for name in submodule_dict.keys()
        }

        # Hooks storage
        self.hooks = []
        self.handles = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on the actual LoRA modules."""

        def create_hook(module_name, wrapper):
            def hook(module, input, output):
                # Get the input tensor
                x = input[0] if isinstance(input, tuple) else input

                # Get sparse activations from wrapper
                result = wrapper(x)

                # Store the results
                batch_size, seq_len = x.shape[:2]

                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        # Get activations for this position
                        pos_idx = batch_idx * seq_len + seq_idx

                        latent_indices = result['latent_indices'][pos_idx]
                        latent_acts = result['latent_acts'][pos_idx]

                        # Only store non-zero activations
                        for latent_idx, act_value in zip(latent_indices, latent_acts):
                            if act_value > 0:  # Only positive activations
                                self.cached_data[module_name]['activations'].append(
                                    act_value.item())
                                self.cached_data[module_name]['locations'].append([
                                    self.total_examples + batch_idx,  # example index
                                    seq_idx  # position in sequence
                                ])
                                self.cached_data[module_name]['latent_indices'].append(
                                    latent_idx.item())

                # Store tokens (once per batch)
                if hasattr(self, 'current_tokens'):
                    for batch_idx in range(batch_size):
                        tokens = self.current_tokens[batch_idx].tolist()
                        self.cached_data[module_name]['tokens'].append(tokens)

            return hook

        # Register hooks on the actual LoRA modules
        for name, wrapper in self.submodule_dict.items():
            hook = create_hook(name, wrapper)
            handle = wrapper.lora_module.register_forward_hook(hook)
            self.handles.append(handle)

        self.total_examples = 0

    def run(self, n_tokens: int, tokens: torch.Tensor):
        """
        Run the model and collect activations.

        Args:
            n_tokens: Number of tokens to process
            tokens: Tokenized input data
        """
        n_batches = (n_tokens // (self.batch_size * tokens.shape[1])) + 1

        self.model.eval()
        with torch.no_grad():
            for batch_idx in tqdm(range(n_batches), desc="Caching activations"):
                # Get batch
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(tokens))

                if start_idx >= len(tokens):
                    break

                batch = tokens[start_idx:end_idx].to(self.device)
                self.current_tokens = batch

                # Forward pass
                _ = self.model(batch)

                self.total_examples = end_idx

                # Check if we've processed enough tokens
                tokens_processed = end_idx * tokens.shape[1]
                if tokens_processed >= n_tokens:
                    break

        # Clean up
        for handle in self.handles:
            handle.remove()

    def save_splits(self, n_splits: int = 5, save_dir: str = "raw_latents"):
        """Save cached activations in Delphi format."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for module_name, data in self.cached_data.items():
            module_path = save_path / module_name
            module_path.mkdir(parents=True, exist_ok=True)

            # Convert to tensors
            activations = torch.tensor(
                data['activations'], dtype=torch.float32)
            locations = torch.tensor(data['locations'], dtype=torch.int64)
            latent_indices = torch.tensor(
                data['latent_indices'], dtype=torch.int64)

            # Determine splits
            n_items = len(activations)
            items_per_split = n_items // n_splits + 1

            for split_idx in range(n_splits):
                start_idx = split_idx * items_per_split
                end_idx = min((split_idx + 1) * items_per_split, n_items)

                if start_idx >= n_items:
                    break

                # Get split data
                split_acts = activations[start_idx:end_idx]
                split_locs = locations[start_idx:end_idx]
                split_latents = latent_indices[start_idx:end_idx]

                # Save as safetensors
                save_dict = {
                    "activations": split_acts,
                    "locations": split_locs,
                    "latent_indices": split_latents
                }

                split_path = module_path / f"split_{split_idx}.safetensors"
                save_file(save_dict, str(split_path))

            # Save tokens separately
            tokens_path = module_path / "tokens.json"
            with open(tokens_path, 'w') as f:
                json.dump(data['tokens'], f)

            # Save metadata
            metadata = {
                "module_name": module_name,
                "n_latents": len(set(data['latent_indices'])),
                "n_splits": min(n_splits, (n_items // items_per_split) + 1),
                "total_activations": n_items,
                "wrapper_config": {
                    "k": self.submodule_dict[module_name].k,
                    "r": self.submodule_dict[module_name].r
                }
            }

            metadata_path = module_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"Saved activation cache to {save_path}")


def setup_lora_wrappers(model: AutoModelForCausalLM, cfg) -> Dict[str, LoRAToSAEWrapper]:
    """
    Create SAE wrappers for all LoRA modules in the model.
    """
    submodule_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, (TopKLoRALinear, FixedTopKLoRALinear)):
            # Create wrapper
            wrapper = LoRAToSAEWrapper(
                lora_module=module,
                module_name=name,
                k=cfg.evals.auto_interp.k,
                r=cfg.evals.auto_interp.r
            )
            submodule_dict[name] = wrapper
            print(
                f"Created wrapper for {name} with k={cfg.evals.auto_interp.k}, r={cfg.evals.auto_interp.r}")

    return submodule_dict


def collect_activations_with_delphi(
    cfg,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "allenai/c4",
    dataset_split: str = "train[:1%]",
    n_tokens: int = 10_000_000,
    max_seq_len: int = 256,
    save_dir: str = "delphi_lora_cache"
):
    """
    Collect activations using Delphi's native methods, adapted for LoRA modules.
    """
    print("Setting up LoRA wrappers...")
    submodule_dict = setup_lora_wrappers(model, cfg)

    if not submodule_dict:
        raise ValueError("No LoRA modules found in the model!")

    print(f"Found {len(submodule_dict)} LoRA modules")

    # Load and tokenize dataset
    print(f"Loading dataset {dataset_name}...")
    data = load_dataset(dataset_name, "en",
                        split=dataset_split, streaming=False)

    # For C4 dataset, the text field is called "text"
    text_key = "text" if "text" in data.column_names else "raw_content"

    print("Tokenizing dataset...")
    tokens = chunk_and_tokenize(
        data,
        tokenizer,
        max_seq_len=max_seq_len,
        text_key=text_key
    )["input_ids"]

    # Create cache
    print("Creating activation cache...")
    cache = DelphinLoRACache(
        model=model,
        submodule_dict=submodule_dict,
        batch_size=cfg.evals.auto_interp.batch_size,
        cfg=cfg
    )

    # Run activation collection
    print(f"Collecting activations for {n_tokens} tokens...")
    cache.run(n_tokens=n_tokens, tokens=tokens)

    # Save results
    print("Saving activation cache...")
    cache.save_splits(n_splits=5, save_dir=save_dir)

    return save_dir


def run_full_delphi_pipeline(
    cfg,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    explainer_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_api: bool = False,
    api_key: Optional[str] = None,
    n_tokens_to_cache: int = 10_000_000,
    max_latents_to_explain: int = 100
):
    """
    Run the complete Delphi pipeline: collection + interpretation.
    """
    # Step 1: Collect activations
    cache_dir = collect_activations_with_delphi(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        n_tokens=n_tokens_to_cache,
        save_dir="delphi_lora_cache"
    )

    # Step 2: Run autointerp for each module
    from delphi.pipeline import Pipeline, process_wrapper
    from delphi.explainers import DefaultExplainer
    from delphi.scorers import RecallScorer, FuzzingScorer, DetectionScorer
    from delphi.clients import Offline, OpenRouter
    import asyncio
    import orjson
    from functools import partial

    # Get list of modules
    module_dirs = [d for d in Path(cache_dir).iterdir() if d.is_dir()]

    for module_dir in module_dirs:
        module_name = module_dir.name
        print(f"\nRunning autointerp for module: {module_name}")

        # Set up output directories
        output_dir = Path(f"delphi_outputs/{module_name}")
        explanation_dir = output_dir / "explanations"
        score_dirs = {
            "recall": output_dir / "scores" / "recall",
            "fuzzing": output_dir / "scores" / "fuzzing",
            "detection": output_dir / "scores" / "detection"
        }

        for dir_path in [explanation_dir] + list(score_dirs.values()):
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create dataset
        latent_dict = {module_name: torch.arange(0, max_latents_to_explain)}

        dataset = LatentDataset(
            raw_dir=cache_dir,
            modules=[module_name],
            sampler_cfg=SamplerConfig(train_set_size=5, test_set_size=5),
            constructor_cfg=ConstructorConfig(
                min_examples=3,
                max_examples=10,
                context_size=20
            ),
            latents=latent_dict,
            tokenizer=tokenizer
        )

        # Set up client
        if use_api:
            client = OpenRouter(explainer_model, api_key=api_key)
        else:
            client = Offline(explainer_model, max_memory=0.8,
                             max_model_len=5120)

        # Create explainer
        explainer = DefaultExplainer(client, tokenizer=tokenizer)

        # Define postprocessing functions
        def save_explanation(result):
            with open(explanation_dir / f"{result.record.latent}.json", 'wb') as f:
                f.write(orjson.dumps(result.explanation))
            return result

        def save_score(result, score_type):
            with open(score_dirs[score_type] / f"{result.record.latent}.json", 'wb') as f:
                f.write(orjson.dumps(
                    {"score": result.score, "type": score_type}))
            return result

        # Create pipeline
        pipeline = Pipeline(
            dataset.load(n_latents=max_latents_to_explain),
            process_wrapper(explainer, postprocess=save_explanation),
            process_wrapper(
                RecallScorer(client, tokenizer=tokenizer),
                postprocess=partial(save_score, score_type="recall")
            )
        )

        # Run pipeline
        asyncio.run(pipeline.run(n_processes=10))

        print(f"Completed autointerp for {module_name}")


# Example usage with your existing model
def delphiii_collect(cfg, model, tokenizer):
    """
    Main entry point for native Delphi collection and interpretation.
    """
    # Option 1: Just collect activations
    cache_dir = collect_activations_with_delphi(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        n_tokens=10_000_000
    )

    # Option 2: Run full pipeline (collection + interpretation)
    run_full_delphi_pipeline(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        n_tokens_to_cache=10_000_000,
        max_latents_to_explain=100
    )

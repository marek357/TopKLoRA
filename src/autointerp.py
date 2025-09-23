from transformers import AutoTokenizer
from neuron_explainer.activations.activations import ActivationRecord, NeuronRecord
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch._dynamo as dynamo
from transformers import StaticCache
from src.delphi_autointerp import delphi_collect_activations
from src.utils import (
    autointerp_preprocess_to_messages,
    autointerp_violates_alternation,
    autointerp_is_valid_dpo_pair,
    AutointerpChatCollator
)
import heapq
import scipy
import dataclasses
import glob
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from src.models import FixedTopKLoRALinear, TopKLoRALinear
from datasets import load_dataset, concatenate_datasets
from peft import PeftModel
from itertools import islice
from tqdm import tqdm
from pprint import pprint
from typing import Union, Optional, List
from dataclasses import dataclass
import logging
from torch.utils.data import DataLoader
import pickle
import json
import torch
import re
import gc
import os
import faulthandler
faulthandler.enable()

# dynamo.config.fail_on_recompile_limit_hit = False
# dynamo.config.recompile_limit = 1e4
# dynamo.config.accumulated_recompile_limit = 1e5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NeuronRecord
# neuron_id: NeuronId
# """Identifier for the neuron."""

# random_sample: list[ActivationRecord] = field(default_factory=list)
# """
# Random activation records for this neuron. The random sample is independent from those used for
# other neurons.
# """
# random_sample_by_quantile: Optional[list[list[ActivationRecord]]] = None
# """
# Random samples of activation records in each of the specified quantiles. None if quantile
# tracking is disabled.
# """
# quantile_boundaries: Optional[list[float]] = None
# """Boundaries of the quantiles used to generate the random_sample_by_quantile field."""

# # Moments of activations
# mean: Optional[float] = math.nan
# variance: Optional[float] = math.nan
# skewness: Optional[float] = math.nan
# kurtosis: Optional[float] = math.nan

# most_positive_activation_records: list[ActivationRecord] = field(default_factory=list)


# ActivationRecord
# """Collated lists of tokens and their activations for a single neuron."""
# tokens: List[str]
# """Tokens in the text sequence, represented as strings."""
# activations: List[float]
# """Raw activation values for the neuron on each token in the text sequence."""

class OrderedActivationRecord(ActivationRecord):
    """
    An ordered version of ActivationRecord that keeps the order of tokens and activations.
    This is useful for cases where the order of tokens matters, such as in language models.
    """

    def __init__(self, tokens: List[str], activations: List[float]):
        super().__init__(tokens=tokens, activations=activations)
        self.tokens = tokens
        self.activations = activations

    def __lt__(self, other):
        return max(self.activations) < max(other.activations)


class BoundedMaxHeap:
    def __init__(self, key=lambda x: x, maxsize=None):
        """
        A max‑heap (based on negated keys) that never grows beyond `maxsize`.
        When full, pushing a new item with key > the current *weakest* will
        evict that weakest item; otherwise the new item is discarded.
        """
        self.key = key
        self.maxsize = maxsize
        self._data = []  # stores tuples (-key(item), item)

    def push(self, item):
        k = self.key(item)
        entry = (-k, item)
        # if we're at capacity, decide whether to keep this one
        if self.maxsize is not None and len(self._data) >= self.maxsize:
            # find the weakest entry (i.e. the tuple with the largest -k → smallest k)
            weakest = max(self._data, key=lambda x: x[0])
            # if new item is no stronger than weakest, drop it
            if entry[0] >= weakest[0]:
                return
            # otherwise remove that weakest entry
            self._data.remove(weakest)
            heapq.heapify(self._data)
        # now there's room, or we weren't full to begin with
        heapq.heappush(self._data, entry)

    def pop(self):
        """Remove & return the *strongest* item."""
        return heapq.heappop(self._data)[1]

    def __len__(self):
        return len(self._data)

    def peek(self):
        """Peek at the *strongest* item."""
        return self._data[0][1]

    def strongest_n(self):
        """Return items sorted descending by key."""
        return [item for _, item in sorted(self._data, key=lambda x: x[0])]


class MaxHeap:
    def __init__(self, key=lambda x: x):
        self.key = key
        self._data = []  # will store tuples (-key(item), item)

    def push(self, item):
        # invert the key so largest becomes most negative
        heapq.heappush(self._data, (-self.key(item), item))

    def pop(self):
        # return the original item
        return heapq.heappop(self._data)[1]

    def __len__(self):
        return len(self._data)

    def peek(self):
        return self._data[0][1]


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
            padding=True,  # Dynamic padding
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


class TopKHookNeuronRecord:
    def __init__(self, module_name: str, tokenizer: AutoTokenizer, cfg):
        self.module_name = module_name
        self.k = cfg.evals.auto_interp.k
        self.r = cfg.evals.auto_interp.r
        self.max_activating_examples = cfg.evals.auto_interp.max_examples_per_latent
        self.tokenizer = tokenizer
        # store activating examples for each latent
        # in the low-rank activation space
        # this will be later used to generate examples
        # and ablate the effect of the latent
        self.activating_examples = {
            latent: BoundedMaxHeap(
                key=lambda x: max(
                    abs(m) for m in x.activations
                ), maxsize=self.max_activating_examples
            )
            for latent in range(self.r)
        }
        self.random_examples = {
            latent: []
            for latent in range(self.r)
        }
        self.activation_counts = {
            latent: 0
            for latent in range(self.r)
        }
        # raise error if params don't make sense
        assert self.k <= self.r, f"Recording top-{self.k} latents when only {self.r} available"
        # these get updated before each forward pass:
        self.dataset_examples_seen = 0
        self.current_pad_mask = None
        self.input_ids = None

    def set_batch_state(self, pad_mask: torch.BoolTensor, ex_offset: int = 0, input_ids=None):
        """Call this once per microbatch, before running forward."""
        self.dataset_examples_seen += ex_offset
        self.current_pad_mask = pad_mask
        if input_ids is not None:
            self.input_ids = input_ids.detach().clone().cpu()

    def to_neuron_records(self):
        """Convert the collected data into a NeuronRecord."""
        latent_records = {}
        dead_latents = []
        for latent, heap in tqdm(self.activating_examples.items(), desc="Converting to NeuronRecords"):
            # Convert the heap to a list and sort by activation magnitude
            sorted_records = sorted(heap._data, key=lambda x: -x[0])
            # Extract the actual ActivationRecord objects
            most_positive_activation_records = [
                ActivationRecord(
                    tokens=record[1].tokens,
                    activations=record[1].activations
                ) for record in sorted_records
            ]

            if most_positive_activation_records:
                activations = torch.cat(
                    [
                        torch.tensor(rec.activations)
                        for rec in most_positive_activation_records
                    ]
                ).flatten()
                # quantile boundaries
                quantile_boundaries = torch.quantile(activations, torch.linspace(
                    0, 1, 11)
                ).tolist() if len(activations) > 0 else None
                random_sample_by_quantile = []
                if quantile_boundaries is not None:
                    for i in range(len(quantile_boundaries) - 1):
                        lower_bound = quantile_boundaries[i]
                        upper_bound = quantile_boundaries[i + 1]
                        # filter random examples by quantile
                        filtered_examples = [
                            rec for rec in self.random_examples[latent]
                            if lower_bound <= max(rec.activations) <= upper_bound
                        ]
                        random_sample_by_quantile.append(filtered_examples)

                latent_records[latent] = NeuronRecord(
                    neuron_id=latent,
                    random_sample=[
                        ActivationRecord(
                            tokens=example.tokens,
                            activations=example.activations
                        ) for example in self.random_examples[latent]
                    ],
                    random_sample_by_quantile=random_sample_by_quantile,
                    quantile_boundaries=quantile_boundaries,
                    mean=activations.mean().item() if most_positive_activation_records else None,
                    variance=activations.var().item() if most_positive_activation_records else None,
                    skewness=torch.tensor(
                        scipy.stats.skew(activations).item()
                    ) if most_positive_activation_records else None,
                    kurtosis=torch.tensor(
                        scipy.stats.kurtosis(activations).item()
                    ) if most_positive_activation_records else None,
                    most_positive_activation_records=most_positive_activation_records
                )
            else:
                # If no activating examples, create an empty NeuronRecord
                latent_records[latent] = NeuronRecord(
                    neuron_id=latent,
                    random_sample=[],
                    random_sample_by_quantile=None,
                    quantile_boundaries=None,
                    mean=None,
                    variance=None,
                    skewness=None,
                    kurtosis=None,
                    most_positive_activation_records=[]
                )
                dead_latents.append(latent)
        return latent_records, dead_latents

    def __call__(self, module, input_activations_tuple, out):
        input_activations = input_activations_tuple[0]
        num_batches, seq_length, _ = input_activations.shape
        mask = self.current_pad_mask
        if mask is None:
            mask = input_activations.new_ones(
                (num_batches, seq_length),
                dtype=torch.bool
            )

        # select LoRA adapter
        adapter = module.lora_module.active_adapter
        if isinstance(adapter, (list, tuple)):
            adapter = adapter[0]

        # compute the low-rank "hidden state"
        low_rank_batched_activations = module.lora_module.lora_A[
            adapter
        ](input_activations).view(num_batches, seq_length, -1)  # (B, L, R)
        r = low_rank_batched_activations.shape[-1]

        assert self.r == r, f"Misspecified LoRA adapter r expected: {self.r}, r observed: {r}"

        # iterate over each batch in input activations
        for batch_idx in range(num_batches):
            padding_offset = (~mask[batch_idx, :]).sum().item()
            tokens = self.tokenizer.convert_ids_to_tokens(
                self.input_ids[batch_idx, padding_offset:],
                skip_special_tokens=False
            ) if self.input_ids is not None else None

            activations = low_rank_batched_activations[batch_idx, padding_offset:, :].detach(
            ).cpu()

            latents_with_topk_activations_in_sequence = activations.topk(
                self.k, dim=1, largest=True, sorted=True
            ).indices.flatten().unique()

            non_activating_latents = set(
                range(self.r)) - set(latents_with_topk_activations_in_sequence.tolist())

            # create ActivationRecord for each latent
            for latent in latents_with_topk_activations_in_sequence:
                if latent >= self.r:
                    raise ValueError(
                        f"Latent index {latent} exceeds the number of latents {self.r}."
                    )

                assert len(activations[:, latent].tolist()) == len(tokens), \
                    f"Activations length {len(activations[:, latent].tolist())} does " \
                    f"not match tokens length {len(tokens)} for latent {latent}."

                record = OrderedActivationRecord(
                    tokens=tokens,
                    activations=activations[:, latent].tolist()
                )

                self.activating_examples[latent.item()].push(record)

            # for each non-activating latent, we still want to record
            # some random examples, so we sample from the current batch
            # if batch_idx % 3 == 0:
            #     # every 3rd batch, we record the top-k activations
            #     # for each non-activating latent in the low-rank space
            #     # this is to avoid excessive memory usage
            #     # and to ensure we have a diverse set of examples
            for latent in non_activating_latents:
                if latent >= self.r:
                    raise ValueError(
                        f"Latent index {latent} exceeds the number of latents {self.r}."
                    )

                # only keep a limited number of random examples
                if len(self.random_examples[latent]) < self.max_activating_examples:
                    record = OrderedActivationRecord(
                        tokens=tokens,
                        activations=activations[:, latent].tolist()
                    )
                    self.random_examples[latent].append(record)


def configure_eot_token(model, tokenizer):
    """Configure EOT token for proper generation stopping."""
    # Determine EOT token (Gemma uses second additional special token)
    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens", [tokenizer.eos_token])[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )

    # Convert to ID
    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)

    # Update generation config
    if hasattr(model.generation_config, 'eos_token_id'):
        if isinstance(model.generation_config.eos_token_id, list):
            if eot_token_id not in model.generation_config.eos_token_id:
                model.generation_config.eos_token_id.append(eot_token_id)
        else:
            prev_eos = model.generation_config.eos_token_id
            model.generation_config.eos_token_id = [prev_eos, eot_token_id]
    else:
        model.generation_config.eos_token_id = [
            tokenizer.eos_token_id, eot_token_id]

    print(f"Configured EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"Generation EOS token IDs: {model.generation_config.eos_token_id}")

    return eot_token_id


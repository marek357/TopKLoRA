from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from transformers import StaticCache
from src.utils import (
    autointerp_preprocess_to_messages,
    autointerp_violates_alternation,
    autointerp_is_valid_dpo_pair,
    AutointerpChatCollator
)
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from src.models import FixedTopKLoRALinear, TopKLoRALinear
from datasets import load_dataset, concatenate_datasets
from peft import PeftModel
from itertools import islice
from tqdm import tqdm
from pprint import pprint
from typing import Union, Optional
from dataclasses import dataclass
import logging
from torch.utils.data import DataLoader
import pickle
import json
import torch
import re
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class LatentActivation:
    activation_magnitude: float
    example_idx: int
    activating_token_id: int
    activating_idx: int
    activating_token: Optional[str]
    is_padding_token: Optional[bool]
    baseline_generation: Optional[str] = None
    ablated_generation: Optional[str] = None
    input_text: Optional[str] = None
    ablation_has_effect: Optional[bool] = None

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

class TopKHeapHook:
    def __init__(self, module_name: str, cfg):
        self.module_name = module_name
        self.k = cfg.evals.auto_interp.k
        self.r = cfg.evals.auto_interp.r
        # store activating examples for each latent
        # in the low-rank activation space
        # this will be later used to generate examples
        # and ablate the effect of the latent
        self.activating_examples = {
            latent: []
            for latent in range(self.r)
        }
        # raise error if params don't make sense
        assert self.k <= self.r, f"Recording top-{self.k} latents when only {self.r} available"
        # these get updated before each forward pass:
        self.dataset_examples_seen = 0
        self.current_pad_mask = None
        self.input_ids = None

    def set_batch_state(self, pad_mask: torch.BoolTensor, ex_offset: int = 0, input_ids = None):
        """Call this once per microbatch, before running forward."""
        self.dataset_examples_seen += ex_offset
        self.current_pad_mask  = pad_mask
        if input_ids is not None:
            self.input_ids = input_ids

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
            example_idx = self.dataset_examples_seen + batch_idx
            for analysed_token_position in range(seq_length):
                # skip tokens which are masked
                if not mask[batch_idx, analysed_token_position]:
                    continue
                low_rank_activations = low_rank_batched_activations[
                    batch_idx, analysed_token_position
                ]
                # pick which neurons to record
                active_latents = low_rank_activations \
                    .abs() \
                    .topk(self.k) \
                    .indices \
                    .tolist()

                for latent in active_latents:
                    self.activating_examples[latent].append(
                        LatentActivation(
                            activation_magnitude=low_rank_activations[latent].item(),
                            example_idx=example_idx,
                            activating_token_id=self.input_ids[batch_idx][analysed_token_position].item(),
                            activating_idx=analysed_token_position,
                            activating_token=None,
                            is_padding_token=None
                        )
                    )
                


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
        model.generation_config.eos_token_id = [tokenizer.eos_token_id, eot_token_id]
    
    print(f"Configured EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"Generation EOS token IDs: {model.generation_config.eos_token_id}")
    
    return eot_token_id


def plot_latent_activation_histogram(hook):
    activating_examples = hook.activating_examples
    latents = list(map(str, activating_examples.keys()))
    counts = [
        len(activating_examples[latent]) 
        for latent in activating_examples
    ]

    fig, ax = plt.subplots()
    y, x = list(zip(*sorted(zip(counts, latents), reverse=False)))
    ax.bar(x, y)
    fig.savefig('latents_activations_sorted.png')

def find_dead_neurons(hook):
    activating_examples = hook.activating_examples
    dead_neurons = []
    for latent in activating_examples:
        if len(activating_examples[latent]) == 0:
            dead_neurons.append(latent)
    return dead_neurons

def check_if_any_dead_neurons(hooks, verbose=True):
    no_dead_neurons = True
    for hook in hooks:
        dead_neurons = find_dead_neurons(hook)
        if dead_neurons:
            no_dead_neurons = False
            if verbose:
                print(
                    'In module:', hook.module_name, 
                    'there are following dead neurons:',
                    dead_neurons, '\n', '-'*60, end='\n\n'
                )
        else:
            if verbose:
                print(
                    'In module:', hook.module_name, 
                    'there are no dead neurons!', '\n', '-'*60, end='\n\n'
                )

    if no_dead_neurons and verbose:
        print('No dead neurons!')

    return no_dead_neurons


def make_disable_hook(layer: int, neuron_idx: int):
    # zeroes out that single component each forward
    def disable_latent(activations, hook):
        # activations: Tensor of shape (batch, seq_len, R)
        activations[..., neuron_idx] = 0.
        return activations
    return disable_latent

def make_enable_hook(layer: int, neuron_idx: int):
    # zeroes out that single component each forward
    def enable_latent(activations, hook):
        # activations: Tensor of shape (batch, seq_len, R)
        activations[..., neuron_idx] = 1.
        return activations
    return enable_latent


def get_hook_statistics(hooks):
    hook_statistics = {}
    for hook in hooks:
        counts = torch.tensor([
            len(hook.activating_examples[latent]) 
            for latent in hook.activating_examples
        ], dtype=torch.float)

        hook_statistics[hook.module_name] = {
            'min': counts.min(),
            'max': counts.max(),
            'mean': counts.mean(),
            'median': counts.median(),
            'top-25-q': torch.quantile(counts, 0.25),
            'top-75-q': torch.quantile(counts, 0.75),
            'top-95-q': torch.quantile(counts, 0.95),
        }
    
    return hook_statistics


def precompute_baseline_generations(model, tokenizer, flat_ds, example_indices, cfg, max_seq_len, collate_fn):
    """Precompute baseline generations for all examples that will be used."""
    print("Precomputing baseline generations...")
    
    baseline_cache = {}
    batch_size = cfg.evals.auto_interp.batch_size
    max_new_tokens = cfg.evals.auto_interp.max_new_tokens
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Process in batches
    for i in tqdm(range(0, len(example_indices), batch_size), desc="Computing baselines"):
        batch_indices = example_indices[i:i + batch_size]
        batch_examples = [flat_ds[idx] for idx in batch_indices]
        
        # Use the provided collate function
        input_batch = collate_fn(batch_examples)
        input_ids = input_batch["input_ids"].to(device)
        attention_mask = input_batch["attention_mask"].to(device)
        
        # Clear cache before generation to avoid warnings
        if hasattr(model, 'generation_config'):
            model.generation_config.use_cache = True
        
        # Generate baseline outputs
        static_cache = StaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_seq_len,
            device=device,
            dtype=model.dtype
        )

        with torch.no_grad():
            try:
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=autocast_dtype):
                        baseline_outputs = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            past_key_values=static_cache,  # Use pre-allocated cache
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=model.generation_config.eos_token_id,
                            do_sample=False,
                            use_cache=True,
                            num_beams=1,  # Greedy is fastest
                            early_stopping=True,
                            return_dict_in_generate=False  # Slightly faster
                        )

            except Exception as e:
                print(f"Warning: Cache issue, falling back to no cache: {e}")
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=autocast_dtype):
                        baseline_outputs = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=model.generation_config.eos_token_id,
                            do_sample=False,
                            use_cache=False,
                            num_beams=1,  # Greedy is fastest
                            early_stopping=True,
                            return_dict_in_generate=False  # Slightly faster
                        )
        # Store results
        baseline_texts_with_prompt = tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # for b, bas, inp in zip(baseline_texts, baseline_texts_with_prompt, input_texts):
        #     assert bas.removeprefix(inp) == b, f'{inp} || {bas}'
        
        for idx, input_text, baseline_text in zip(batch_indices, input_texts, baseline_texts_with_prompt):
            baseline_cache[str(idx)] = {
                'input': input_text,
                'baseline': baseline_text.removeprefix(input_text)
            }
    
    print(f"Precomputed baselines for {len(baseline_cache)} examples")
    return baseline_cache



def analyse_model(cfg, model, tokenizer):
    torch.set_float32_matmul_precision('high')
    # eot_token_id = configure_eot_token(model, tokenizer)
    chat_collate = ChatTemplateCollator(
        tokenizer, device,
        max_length=cfg.evals.auto_interp.max_length
    )
    hooks = []
    handles = []
    num_inserted_hooks = 0
    for name, module in model.base_model.model.named_modules():
        if isinstance(module, TopKLoRALinear):
            # we are using a stateful hook to keep track
            # of processing example's parameters 
            # (processing example's idx and mask)
            hook = TopKHeapHook(name, cfg)
            hooks.append(hook)
            handles.append(
                module.register_forward_hook(hook)
            )
            num_inserted_hooks += 1
    print(f"registered hooks for {num_inserted_hooks} LoRA blocks")

    raw_ds = load_dataset(
        cfg.evals.auto_interp.dataset_name,
        split="train"
    )

    _TAG_RE = re.compile(r"(Human|Assistant):")
    _ROLE_MAP = {"Human": "user", "Assistant": "assistant"}

    msg_ds = raw_ds.map(
        autointerp_preprocess_to_messages,
        remove_columns=raw_ds.column_names
    )

    msg_ds = msg_ds.filter(
        lambda ex:
        not autointerp_violates_alternation(ex["chosen"]) and
        not autointerp_violates_alternation(ex["rejected"]) and
        autointerp_is_valid_dpo_pair(ex["chosen"]) and
        autointerp_is_valid_dpo_pair(ex["rejected"])
    )

    chosen_ds = msg_ds.rename_column(
        "chosen", "input"
    ).remove_columns(["rejected"])

    rejected_ds = msg_ds.rename_column(
        "rejected", "input"
    ).remove_columns(["chosen"])

    flat_ds = concatenate_datasets(
        [chosen_ds, rejected_ds]
    ).shuffle(seed=cfg.seed)

    print(f"Dataset size after flattening: {len(flat_ds):,}")


    loader = DataLoader(
        flat_ds,
        batch_size=cfg.evals.auto_interp.batch_size,
        shuffle=False,
        collate_fn=chat_collate,
        drop_last=False
    )

    if cfg.evals.auto_interp.max_rows == -1:
        MAX_BATCHES = len(loader)
    else:
        MAX_BATCHES = int(
            cfg.evals.auto_interp.max_rows/cfg.evals.auto_interp.batch_size
        )
    MAX_BATCHES = 3500
    # MAX_BATCHES = 1
    # take only the first MAX_BATCHES batches
    limited_loader = islice(loader, MAX_BATCHES)

    try:
        with open(f'cache/hooks_r{cfg.evals.auto_interp.r}_k{cfg.evals.auto_interp.k}_steps{cfg.model.train_steps}_batches{MAX_BATCHES}_with_activations.pkl', 'rb+') as f:
            hooks = pickle.load(f)
        print('loaded hooks from cache')

    except Exception as e:
        print(e)
        for batch in tqdm(
            limited_loader, 
            total=MAX_BATCHES, 
            desc="Collecting hook firings"
        ):
            input_ids = batch["input_ids"].to(device)
            for hook in hooks:
                hook.set_batch_state(
                    pad_mask=batch["attention_mask"].to(torch.bool),
                    input_ids=input_ids
                )

            model(input_ids)
            
            for hook in hooks:
                hook.set_batch_state(
                    # increment by batch size
                    ex_offset=input_ids.size(0),
                    pad_mask=None
                )

        # add extra metadata
        for hook in hooks:
            for latent in hook.activating_examples:
                for latent_activation in hook.activating_examples[latent]:
                    tok_id = latent_activation.activating_token_id
                    latent_activation.activating_token = tokenizer.decode(
                        tok_id, skip_special_tokens=False
                    )
                    latent_activation.is_padding_token = bool(
                        tokenizer.pad_token == tok_id
                    )

        try:
            with open(f'cache/hooks_r{cfg.evals.auto_interp.r}_k{cfg.evals.auto_interp.k}_steps{cfg.model.train_steps}_batches{MAX_BATCHES}_with_activations.pkl', 'wb+') as f:
                pickle.dump(hooks, f)
        except Exception as err:
            print(err)

    for handle in handles:
        handle.remove()

    check_if_any_dead_neurons(hooks)

    all_example_indices = list(range(
        MAX_BATCHES * cfg.evals.auto_interp.batch_size
    ))

    print(f"Will precompute baselines for {len(all_example_indices)} unique examples")
    
    # Precompute all baseline generations
    max_new_tokens = cfg.evals.auto_interp.max_new_tokens
    max_seq_len = chat_collate.max_length + max_new_tokens
    max_len_path = max_new_tokens
    try:
        with open(f'cache/baselines_r{cfg.evals.auto_interp.r}_k{cfg.evals.auto_interp.k}_steps{cfg.model.train_steps}_maxlen{max_len_path}.json', 'r') as f:
            baseline_cache = json.load(f)
    except Exception as e:
        print(e)
        batch_size = cfg.evals.auto_interp.latent_interp_batch_size

        baseline_cache = precompute_baseline_generations(
            model, tokenizer, flat_ds, 
            all_example_indices, cfg, 
            max_seq_len, chat_collate
        )
        try:
            with open(f'cache/baselines_r{cfg.evals.auto_interp.r}_k{cfg.evals.auto_interp.k}_steps{cfg.model.train_steps}_maxlen{max_len_path}.json', 'w+') as f:
                json.dump(baseline_cache, f)
        except Exception as err:
            print(err)

    print('baselines computed!')

    ablation_results = {}

    try:
        with open(f'cache/experiment_r{cfg.evals.auto_interp.r}_k{cfg.evals.auto_interp.k}_steps{cfg.model.train_steps}_batches{MAX_BATCHES}_maxlen{max_len_path}_sorted.pkl', 'rb') as f:
            ablation_results = pickle.load(f)
        print('read ablation results from cache')
    except Exception as e:
        print(e)

    for hook in tqdm(hooks, desc="Analyzing TopKLora Adapters"):
        # find target adapter module 
        # (i.e. the probe we want to analyse)
        target_module = None
        for name, module in model.base_model.model.named_modules():
            if name == hook.module_name and isinstance(module, TopKLoRALinear):
                target_module = module
                break
        
        if target_module is None:
            logging.warning(f"Could not find module {hook.module_name}")
            continue
            
        adapter = target_module.lora_module.active_adapter
        if isinstance(adapter, (list, tuple)):
            adapter = adapter[0]

        # store generation results in results dict
        if hook.module_name not in ablation_results:
            ablation_results[hook.module_name] = {}

        batch_size = cfg.evals.auto_interp.latent_interp_batch_size
        max_new_tokens = cfg.evals.auto_interp.max_new_tokens
        max_seq_len = chat_collate.max_length + max_new_tokens

        
        # Create static cache once
        static_cache = StaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_seq_len,
            device=device,
            dtype=model.dtype
        )
        assert set(hook.activating_examples) == set(range(hook.r))

        for latent in tqdm(range(hook.r), desc=f"Latents in {hook.module_name}", leave=False):
            if str(latent) in ablation_results[hook.module_name]:
                logging.info(f'Loaded {hook.module_name}/{latent} from cache')
                continue
            if len(hook.activating_examples[latent]) == 0:
                logging.warning(f"[CRITICAL WARNING] Dead latent!!! {hook.module_name}/{latent}")
                continue

            ablation_results[hook.module_name][latent] = {
                'num_activations': len(hook.activating_examples[latent]),
                'examples': []
            }

            max_examples = cfg.evals.auto_interp.max_examples_per_latent
            examples_to_ablate = list(
                sorted(
                    hook.activating_examples[latent],
                    # descending sort according to 
                    # the activation magnitude
                    key=lambda x: abs(x.activation_magnitude),
                    reverse=True
                )
            # restrict to only the top 
            # activating examples within
            # this setting
            )[:max_examples]

            if cfg.evals.auto_interp.hook_type == 'disable':
                ablation_hook = make_disable_hook(hook.module_name, latent)
            elif cfg.evals.auto_interp.hook_type == 'enable':
                ablation_hook = make_enable_hook(hook.module_name, latent)
            else:
                raise NotImplementedError

            all_examples = []

            # Clear any conflicting generation config
            if hasattr(model.generation_config, 'cache_implementation'):
                model.generation_config.cache_implementation = None

            # Enable mixed precision for faster computation
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            for i in range(0, len(examples_to_ablate), batch_size):
                batch_indices = examples_to_ablate[i:i + batch_size]
                batch_examples = [flat_ds[idx.example_idx] for idx in batch_indices]
                
                # Collate batch
                input_batch = chat_collate(batch_examples)
                input_ids = input_batch["input_ids"].to(device)
                attention_mask = input_batch["attention_mask"].to(device)
                static_cache.reset()
                # Attach ablation hook directly (no lambda needed)
                ablation_handle = target_module.lora_module.lora_A[adapter].register_forward_hook(
                    ablation_hook
                )
                
                try:
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', dtype=autocast_dtype):
                            try:
                                ablated_outputs = model.generate(
                                    input_ids,
                                    attention_mask=attention_mask,
                                    # past_key_values=static_cache,  # Use pre-allocated cache
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=model.generation_config.eos_token_id,
                                    do_sample=False,
                                    use_cache=True,
                                    num_beams=1,  # Greedy is fastest
                                    # early_stopping=True,
                                    return_dict_in_generate=False  # Slightly faster
                                )
                            except RuntimeError as e:
                                if "index_copy_" in str(e):
                                    # Fallback to dynamic cache if static cache fails
                                    print(f"Static cache failed for batch, using dynamic cache: {e}")
                                    ablated_outputs = model.generate(
                                        input_ids,
                                        attention_mask=attention_mask,
                                        max_new_tokens=max_new_tokens,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=model.generation_config.eos_token_id,
                                        do_sample=False,
                                        use_cache=True,  # Will use dynamic cache
                                        num_beams=1,
                                        return_dict_in_generate=False
                                    )
                                else:
                                    raise e
                finally:
                    ablation_handle.remove()

                # Batch-decode in one go
                ablated_texts = tokenizer.batch_decode(
                    ablated_outputs,
                    skip_special_tokens=True,
                    # clean_up_tokenization_spaces=True
                )

                # Process batch results
                # ablated_texts = tokenizer.batch_decode(ablated_outputs, skip_special_tokens=True)
                
                batch_results = []
                for example, ablated_text in zip(examples_to_ablate, ablated_texts):
                    # Get precomputed baseline from cache
                    cached_data = baseline_cache[str(example.example_idx)]
                    baseline_text = cached_data['baseline']
                    input_text = cached_data['input']
                    example.input_text = input_text
                    example.baseline_text = baseline_text
                    example.ablated_text = ablated_text.removeprefix(input_text)
                    example.ablation_has_effect = baseline_text != example.ablated_text
                    batch_results.append(example)
                    # batch_results.append({
                    #     'example_idx': idx,
                    #     'input': input_text,
                    #     'baseline': baseline_text,
                    #     'ablated': ablated_text.removeprefix(input_text),
                    #     'has_effect': baseline_text != ablated_text.removeprefix(input_text)
                    # })
                all_examples.extend(batch_results)
            
            # Store all examples for this latent
            ablation_results[hook.module_name][latent]['examples'] = all_examples
            
            # Calculate effect rate for this latent
            effects = [ex.ablation_has_effect for ex in all_examples]
            ablation_results[hook.module_name][latent]['effect_rate'] = (
                sum(effects) / len(effects) if effects else 0
            )

            try:
                with open(f'cache/experiment_r{cfg.evals.auto_interp.r}_k{cfg.evals.auto_interp.k}_steps{cfg.model.train_steps}_batches{MAX_BATCHES}_maxlen{max_len_path}_sorted.pkl', 'wb+') as f:
                    pickle.dump(ablation_results, f)
            except KeyboardInterrupt as e:
                # finish saving to a file if keyboard interrupt
                with open(f'cache/experiment_r{cfg.evals.auto_interp.r}_k{cfg.evals.auto_interp.k}_steps{cfg.model.train_steps}_batches{MAX_BATCHES}_maxlen{max_len_path}_sorted.pkl', 'wb+') as f:
                    pickle.dump(ablation_results, f)
                raise e
            except Exception as e:
                print(e)
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect() 
            
    print("\n" + "="*60)
    print("ABLATION ANALYSIS RESULTS")
    print("="*60)
    
    for module_name, latents_data in ablation_results.items():
        print(f"\n{module_name}:")
        print("-" * len(module_name))
        
        # Sort latents by effect rate
        sorted_latents = sorted(
            [(l, d['effect_rate'], d['num_activations']) 
             for l, d in latents_data.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Show statistics
        total_latents = len(sorted_latents)
        active_latents = sum(1 for _, rate, _ in sorted_latents if rate > 0)
        print(f"Total latents analyzed: {total_latents}")
        print(f"Latents with effects: {active_latents} ({active_latents/total_latents*100:.1f}%)")
        
        # Show top impactful latents
        print("\nTop 5 most impactful latents:")
        for i, (latent, effect_rate, num_acts) in enumerate(sorted_latents[:5]):
            if effect_rate > 0:
                print(f"  {i+1}. Latent {latent}: {effect_rate:.1%} effect rate ({num_acts} activations)")
                
                # Show an example of the effect
                example_with_effect = next(
                    (ex for ex in ablation_results[module_name][latent]['examples'] if ex['has_effect']),
                    None
                )
                if example_with_effect:
                    print("     Example effect:")
                    print(f"     Input: {example_with_effect['input'][:100]}...")
                    print(f"     Baseline: {example_with_effect['baseline'][:100]}...")
                    print(f"     Ablated:  {example_with_effect['ablated'][:100]}...")

    return ablation_results

import torch._dynamo as dynamo

@dynamo.disable
def analyse_model_with_compile_disabled(cfg, model, tokenizer):
    """Wrapper that temporarily disables torch.compile during analysis."""
    
    # Disable dynamo/compile
    # with dynamo.disable():
    return analyse_model(cfg, model, tokenizer)

def analyse_model_safe(cfg, model, tokenizer):
    """Run analysis with model in eval mode to avoid compilation issues."""
    # Store original training mode
    was_training = model.training
    
    # Set to eval mode and use no_grad
    model.eval()
    
    with torch.no_grad():
        # Disable gradient checkpointing if it exists
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        
        # Run the original analyse_model
        result = analyse_model(cfg, model, tokenizer)
    
    # Restore original mode
    if was_training:
        model.train()
    
    return result



def analyse_model_transformerlens(cfg, model, tokenizer):
    # the input model is wrapped in PeftModelForCausalLM
    # this means that a lot of things expected by
    # the HookedTransformer API is one layer deeper
    model.base_model.model.model.to('cpu')
    model.base_model.model.to('cpu')
    model.base_model.to('cpu')
    model.to('cpu')

    # print(model)
    # print('-'*60)
    # print(model.base_model)
    # print('-'*60)
    # print(model.base_model.model.model.layers)

    # print(model.base_model.model.model.layers[0].self_attn.o_proj.weight)
    # assert False

    gemma2_for_causal_lm = model.base_model.model
    gemma2_for_causal_lm.embed_tokens = gemma2_for_causal_lm.get_input_embeddings()
    gemma2_for_causal_lm.layers      = gemma2_for_causal_lm.model.layers
    gemma2_for_causal_lm.norm        = gemma2_for_causal_lm.model.norm
    # print(device)
    # assert False
    hooked_model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=model,
        fold_ln=False,
        center_unembed=False,
        device=device
    )

    def record_lora_activation(activation, hook):
        # You could store these in a list, log their norms, etc.
        print(f"[{hook.name}] activation shape:", activation.shape)
        return activation

    # 5. Attach your hook to every submodule whose name contains "lora_B"
    hooked_model.add_hook(
        lambda name: "lora_A" in name,
        record_lora_activation,
        dir="fwd"
    )

    # 6. Run a forward pass and see your hook fire
    tokens = "The quick brown fox"
    logits, cache = hooked_model.run_with_cache(tokens)
    print(logits)
    print(cache)

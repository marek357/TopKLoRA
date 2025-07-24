from src.utils import (
    autointerp_preprocess_to_messages,
    autointerp_violates_alternation,
    autointerp_is_valid_dpo_pair,
    AutointerpChatCollator
)
from transformers import AutoTokenizer
from src.analysis import ChatTemplateCollator
from datasets import load_dataset, concatenate_datasets
import pickle
from tqdm import tqdm
import json

def get_properly_tokenized_text(example, tokenizer):
    """Tokenize text the same way as during batch processing"""
    # Apply chat template first
    text = tokenizer.apply_chat_template(
        example.get("input", example.get("chosen", example.get("rejected"))),
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Then tokenize
    tokens = tokenizer.tokenize(text)
    return tokens, text

def get_specific_hook(hook_list, example_idx):
    for h in hook_list:
        if h.example_idx == example_idx:
            return h
    return None

def get_module_hook_list(hooks, module):
    for h in hooks:
        if h.module_name == module:
            return h
    return None


def fix_activating_indices_postprocess(experiment, hooks, tokenizer, flat_ds, chat_collate, baselines):
    """Fix activating indices after data collection."""
    for module in tqdm(experiment, desc='Fixing modules'):
        adapter = experiment[module]
        module_hook_list = get_module_hook_list(hooks, module)
        for latent_idx in tqdm(adapter, desc='Fixing individual latents'):
            latent = adapter[latent_idx]
            hook_list = module_hook_list.activating_examples[latent_idx]
            for example_idx, example in enumerate(latent['examples']):
                hook = get_specific_hook(hook_list, example.example_idx)
                dataset_entry = flat_ds[example.example_idx]
                tokenized, text = get_properly_tokenized_text(dataset_entry, tokenizer)
                example.collated_input_text = text
                example.tokenised_input_text = tokenized
                example.activating_idx = [max(0, idx-1) for idx in hook.activating_idx]
                example.activating_idx_padded = [-1 for idx in hook.activating_idx]
                for idx, tok in zip(example.activating_idx, example.activating_token):
                    assert example.tokenised_input_text[idx] == tok, f'{module}, {latent_idx}, {example_idx}'

    with open('/home/cvenhoff/lora_interp/cache/coalesced_fixidx.pkl', 'wb+') as f:
        pickle.dump(experiment, f)


                            

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/home/cvenhoff/lora_interp/experiments/gemma-2-2b_topk_dpo_r1024_k8_steps5000/final_adapter")
    raw_ds = load_dataset(
        'Anthropic/hh-rlhf',
        split="train"
    )


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
    ).shuffle(seed=42)

    chat_collate = ChatTemplateCollator(
        tokenizer, 'cuda',
        max_length=512
    )

    with open('/home/cvenhoff/lora_interp/cache/baselines_r1024_k8_steps5000_maxlen125.json', 'r') as f:
        baselines = json.load(f)

    with open('/home/cvenhoff/lora_interp/cache/hooks_r1024_k8_steps5000_batches3500_with_activations_truncated_idxfix.pkl', 'rb') as f:
        hooks = pickle.load(f)

    with open('cache/coalesced.pkl', 'rb') as f:
        experiment = pickle.load(f)
    
    fix_activating_indices_postprocess(experiment, hooks, tokenizer, flat_ds, chat_collate, baselines)

    # with open('/home/cvenhoff/lora_interp/cache/hooks_r1024_k8_steps5000_batches3500_with_activations_truncated_fixed.pkl', 'wb+') as f:
    #     pickle.dump(hooks, f)
    

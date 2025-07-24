from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from src.analysis import ChatTemplateCollator
from src.utils import (
    autointerp_preprocess_to_messages,
    autointerp_violates_alternation,
    autointerp_is_valid_dpo_pair,
    AutointerpChatCollator
)
import pickle
from tqdm import tqdm
import glob
import json
import os

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

    path = '/home/cvenhoff/lora_interp/cache/hooks_r1024_k8_steps5000_batches3000_with_activations_truncated_idxfix_max250'
    os.makedirs(path + '_fixed', exist_ok=True)
    for file in glob.glob(f'{path}/*.pkl'):
        with open(file, 'rb') as f:
            hook_data = pickle.load(f)
        
        for latent_idx in tqdm(hook_data.activating_examples, desc='Fixing individual latents'):
            latent = hook_data.activating_examples[latent_idx]
            for example in latent:
                dataset_entry = flat_ds[example.example_idx]
                tokenized, text = get_properly_tokenized_text(dataset_entry, tokenizer)
                example.activating_idx = [ex - 1 if idx > 0 else ex for idx, ex in enumerate(example.activating_idx)]
                example.collated_input_text = text
                example.input_text = text
                example.tokenised_input_text = tokenized
                for idx, tok in zip(example.activating_idx, example.activating_token):
                    assert example.tokenised_input_text[idx] == tok, f'Tokens mismatch: {latent_idx}, {example.example_idx}'
                    
        with open(path + '_fixed/' + os.path.basename(file), 'wb+') as f:
            pickle.dump(hook_data, f)

    

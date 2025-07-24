import json
import re
import time
from ifeval import Evaluator, instruction_registry, get_default_dataset
from googleapiclient import discovery
from collections import defaultdict
import numpy as np
from openai import OpenAI
from src.models import TopKLoRALinear
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import (
    get_dataset_config_names,
    load_dataset,
    concatenate_datasets
)
from src.analysis import analyse_model
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from itertools import islice
from peft.tuners.lora import LoraLayer
from pathlib import Path
from tqdm import tqdm
import evaluate
import torch
import os
import logging
from src.utils import (
    AutointerpChatCollator,
    analyze_text_toxicity_eval,
    autointerp_build_activation_prompt,
    autointerp_build_dataset,
    autointerp_build_lora_json_with_responses,
    autointerp_build_lora_json_with_responses_local,
    autointerp_build_prompt,
    autointerp_collapse_heaps,
    autointerp_evaluate,
    autointerp_extract_digit,
    autointerp_is_valid_dpo_pair,
    autointerp_make_topk_hook,
    autointerp_preprocess_to_messages,
    autointerp_token_windows_dict,
    autointerp_violates_alternation,
    build_metrics_eval_messages,
    preprocess_to_perspective_message
)


device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'

import hashlib

# def hash_lora_weights(model):
#     import hashlib
#     sha = hashlib.sha256()
#     for name, module in model.named_modules():
#         if isinstance(module, TopKLoRALinear):
#             for pname, param in module.lora_module.named_parameters():
#                 sha.update(param.detach().cpu().numpy().tobytes())
#     return sha.hexdigest()

def hash_lora_weights(model):
    sha = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinear):
            for pname, param in module.lora_module.named_parameters():
                # pull to CPU once
                tensor = param.detach().cpu()
                # numpy() doesn’t support bfloat16 or float16 — cast if needed
                if tensor.dtype in (torch.bfloat16, torch.float16):
                    tensor = tensor.to(torch.float32)
                # now safe to get the raw bytes
                sha.update(tensor.numpy().tobytes())
    return sha.hexdigest()


def init_model_tokenizer(model_cfg, auto_interp=False):
    print('Initialising a model for eval...')
    print(model_cfg)
    print('Model init start...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.adapter_checkpoint_dir,
        use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model,
        torch_dtype="auto",
        device_map="cpu"
    )

    if not getattr(tokenizer, "chat_template", None) and not auto_interp:
        logging.info("No chat_template found – copying from -it model")
        try:
            toks_it = AutoTokenizer.from_pretrained(
                model_cfg.model_it_name,
                use_fast=False
            )
            if getattr(toks_it, "chat_template", None):
                tokenizer.chat_template = toks_it.chat_template
                logging.info("chat_template copied successfully")
            # Merge additional special tokens if needed
            extra = toks_it.special_tokens_map.get(
                "additional_special_tokens", []
            )
            if extra:
                new_tokens = [
                    t for t in extra if t not in tokenizer.get_vocab()
                ]
                if new_tokens:
                    tokenizer.add_special_tokens(
                        {"additional_special_tokens": new_tokens}
                    )
                    model.resize_token_embeddings(len(tokenizer))
                    logging.info(
                        "Added %d extra special tokens",
                        len(new_tokens)
                    )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to copy -it tokenizer: %s", exc)


    model = PeftModel.from_pretrained(
        model,
        model_cfg.adapter_checkpoint_dir,
        # there are issues with mps
        # so first loading to cpu
        # and then moving it to $device
        device_map="cpu",
        use_safetensors=True
    )
    replaced = 0
    wrapped_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinear):
            continue  # Already wrapped
        if hasattr(module, "lora_A"):
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            attr = name.split(".")[-1]
            wrapped = TopKLoRALinear(
                module,
                layer_name=name,
                r=module.r,
                alpha=module.lora_alpha,
                k=model_cfg.k,
                autointerp_evaluation_mode=auto_interp
            )
            setattr(parent, attr, wrapped)
            replaced += 1
    print(f"Wrapped {replaced} LoraLayer modules into TopKLoRALinear.")
    model.to(device)
    model.eval()

    print(f"Loaded adapter from: {model_cfg.adapter_checkpoint_dir}")
    print(f"Adapter SHA256 hash: {hash_lora_weights(model)}")

    def debug_lora_weights(model):
        total_params = 0
        nonzero_params = 0
        for name, module in model.named_modules():
            if isinstance(module, TopKLoRALinear):
                for pname, param in module.lora_module.named_parameters():
                    total_params += param.numel()
                    nonzero_params += param.nonzero().size(0)
        print(f"LoRA param stats: {nonzero_params}/{total_params} non-zero")

    debug_lora_weights(model)

    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinear):
            wrapped_modules[name] = module



    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens", [tokenizer.eos_token])[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )
    
    # Convert to ID
    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
    
    # Get the base EOS token ID
    base_eos_token_id = tokenizer.eos_token_id
    
    # Update generation config with both EOS and EOT tokens
    if hasattr(model.generation_config, 'eos_token_id'):
        # Create a list of both tokens
        eos_token_ids = []
        
        # Add base EOS token
        if isinstance(base_eos_token_id, list):
            eos_token_ids.extend(base_eos_token_id)
        else:
            eos_token_ids.append(base_eos_token_id)
        
        # Add EOT token if it's different
        if eot_token_id not in eos_token_ids:
            eos_token_ids.append(eot_token_id)
        
        model.generation_config.eos_token_id = eos_token_ids
    else:
        model.generation_config.eos_token_id = [base_eos_token_id, eot_token_id]
    
    # Log the configuration
    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS token ID(s): {model.generation_config.eos_token_id}")



    # if 'gemma' in model_cfg.name:
    #     tokenizer.padding_side = "left"
    #     tokenizer.truncation_side = "left"
    #     model.generation_config.eos_token_id = [1, 107]
    #     model.generation_config.max_length = 512

    # elif 'llama' in model_cfg.name:
    #     tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if auto_interp:
        return model, tokenizer, wrapped_modules
    else:
        return model, tokenizer

def check_bbt_norm():
    base_model='/home/cvenhoff/lora_interp/experiments/merged/google/gemma-2-2b_sft'
    adapter_dir='/home/cvenhoff/lora_interp/experiments/gemma-2-2b_topk_dpo_r1024_k8_steps5000/final_adapter'
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="cpu"
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        # there are issues with mps
        # so first loading to cpu
        # and then moving it to $device
        device_map="cpu",
        use_safetensors=True
    )
    for name, module in model.named_modules():
        if hasattr(module, "lora_B"):
            lora_B = module.lora_B
            B = lora_B['default'].weight
            B = B.T
            # Normalize
            B_norm = B / B.norm(dim=1, keepdim=True)
            # Compute cosine similarity matrix
            sim_matrix = B_norm @ B_norm.T  # Shape: [in_dim, in_dim]
            # Print the cosine similarity matrix
            # Check if any value is greater than 0.99
            if (sim_matrix > 0.8).any():
                print(f"High similarity detected in {name} with value: {torch.ones_like(sim_matrix)[sim_matrix > 0.8].sum()-1024} non-diagonal elements")
                # print(torch.nonzero(sim_matrix[sim_matrix > 0.99]))   
                    
            print("-" * 40)

def metrics():
    def eval_metrics(cfg):
        model, tokenizer = init_model_tokenizer(cfg.model)

        configs = get_dataset_config_names("HuggingFaceH4/hhh_alignment")
        parts = []
        for cfg_ in configs:
            ds = load_dataset(
                "HuggingFaceH4/hhh_alignment",
                cfg_, split="test"
            )
            ds = ds.add_column("subset", [cfg_] * len(ds))
            parts.append(ds)
            print(f"Loaded {cfg_:8} subset with {len(ds)} rows.")
        hhh_all = concatenate_datasets(parts)
        assert len(hhh_all) == 221

        print(f"Total rows: {len(hhh_all)}")
        print("-" * 60)

        metric_global = evaluate.load("accuracy")
        metrics_by_subset = defaultdict(lambda: evaluate.load("accuracy"))

        with torch.no_grad():
            for i, ex in tqdm(enumerate(hhh_all)):
                q = ex["input"]
                choices = ex["targets"]["choices"]
                gold_idx = ex["targets"]["labels"].index(1)

                base_prompt_raw = build_metrics_eval_messages(
                    q, choices[0], choices[1]
                )

                # Render prompt up to "Reply A:" and then append one of the replies
                base_prompt = tokenizer.apply_chat_template(
                    base_prompt_raw, tokenize=False, add_generation_prompt=False)

                # Create inputs for scoring reply A
                full_text_a = base_prompt + choices[0]
                full_text_b = base_prompt + choices[1]

                def logprob_of(text):
                    enc = tokenizer(text, return_tensors="pt",
                                    add_special_tokens=False).to(model.device)
                    input_ids = enc.input_ids

                    with torch.no_grad():
                        out = model(input_ids)
                        logits = out.logits

                    # Compute token-wise log probs
                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]

                    log_probs = F.softmax(shift_logits, dim=-1)
                    selected_logprobs = log_probs.gather(
                        2, shift_labels.unsqueeze(-1)).squeeze(-1)

                    # Sum log probs from the start of the reply only (not prompt)
                    reply_start = len(
                        tokenizer(base_prompt, add_special_tokens=False)["input_ids"])
                    reply_logprob = selected_logprobs[:, reply_start:].sum()

                    return reply_logprob.item()

                logp_A = logprob_of(full_text_a)
                logp_B = logprob_of(full_text_b)

                pred = 1 if logp_B > logp_A else 0
                if i % 10 == 0:
                    print(f"[EX {i}] GOLD={gold_idx} | logp_A={logp_A:.4f} | logp_B={logp_B:.4f} | pred={pred}")


                metric_global.add(prediction=pred, reference=gold_idx)
                metrics_by_subset[ex["subset"]].add(
                    prediction=pred, reference=gold_idx
                )

        results = {"overall": metric_global.compute()["accuracy"]}
        for subset, m in metrics_by_subset.items():
            results[subset] = m.compute()["accuracy"]

        print("\nFinal accuracy summary")
        for k, v in results.items():
            print(f"{k:8}: {v:.3%}")
        return results

    return eval_metrics


def auto_interp():
    def eval_auto_interp(cfg):
        model, tokenizer, _ = init_model_tokenizer(cfg.model, True)
        analyse_model(cfg, model, tokenizer)
        return
        print('Evaluating auto interp')
        if 'sft' in cfg.model.adapter_checkpoint_dir:
            print('SFT model, skipping auto interp...')
            return
        model, tokenizer, topk_modules = init_model_tokenizer(cfg.model, auto_interp=True)

        topk_store = defaultdict(
            lambda: defaultdict(lambda: {"pos": [], "neg": []})
        )
        example_counter = 0        # global running index over all examples

        # TODO: do NOT rely on global state
        current_vals = {
            'current_ex_offset': 0,
            'current_pad_mask': None  # will be set per batch
        }

        for name, mod in topk_modules.items():
            print(f"Hooking: {name} of type {type(mod)}")
            mod.register_forward_hook(
                autointerp_make_topk_hook(
                    name, cfg, topk_store, current_vals
                )
            )
        print(f"✔ registered hooks for {len(topk_modules)} LoRA blocks")

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

        autointerp_collate_chat = AutointerpChatCollator(tokenizer, device)
        loader = DataLoader(
            flat_ds,
            batch_size=cfg.evals.auto_interp.batch_size,
            shuffle=False,
            collate_fn=autointerp_collate_chat,
            drop_last=False
        )

        if cfg.evals.auto_interp.max_rows == -1:
            MAX_BATCHES = len(loader)
        else:
            MAX_BATCHES = int(
                cfg.evals.auto_interp.max_rows/cfg.evals.auto_interp.batch_size
            )

        # take only the first MAX_BATCHES batches
        limited_loader = islice(loader, MAX_BATCHES)

        for row_idx, batch in enumerate(
            tqdm(limited_loader, total=MAX_BATCHES), start=0
        ):
            input_ids = batch["input_ids"].to(device)
            current_vals['current_pad_mask'] = batch["attention_mask"].to(
                torch.bool
            )
            B = input_ids.size(0)

            _ = model(input_ids)             # hooks fire here

            current_vals['current_pad_mask'] = None         # clear reference
            current_vals['current_ex_offset'] += B

        adapters_pos_map = autointerp_collapse_heaps(topk_store)

        os.makedirs('temp', exist_ok=True)
        if cfg.evals.auto_interp.dump_generated:
            with open(
                "temp/adapters_pos_map_topk.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    adapters_pos_map, f,
                    indent=2, ensure_ascii=False
                )

        # try:
        #     with open('temp/lora_neuron_info.json', 'r') as f:
        #         json_blob = json.load(f)
        #     print('succesfully loaded neuron interp from cache')
        # except FileNotFoundError:
        #    print('neuron interp not found in cache, regenerating...')
        
        # json_blob = autointerp_build_lora_json_with_responses(
        #     adapters_pos_map,
        #     flat_ds, tokenizer,
        #     model="gpt-4o-mini",
        #     include_cot=False,
        #     include_few_shot=False
        # )

        json_blob = autointerp_build_lora_json_with_responses_local(
            adapters_pos_map, flat_ds, tokenizer,
            model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            include_cot=False,
            include_few_shot=False,
            temperature=0.0
        )

        activations = adapters_pos_map
        lora_info = json_blob
        examples = flat_ds
        dataset = autointerp_build_dataset(
            activations,
            lora_info,
            examples,
            cfg.model.adapter_checkpoint_dir,
            window=cfg.evals.auto_interp.context_size_each_side,
        )

        print(f"Generated {len(dataset)} test items.")

        print(autointerp_build_prompt(dataset[0]))

        client = OpenAI()
        predictions, correct = [], 0
        bar = tqdm(dataset, desc="Evaluating", unit="ex")
        for i, ex in enumerate(bar, 1):
            user_prompt = autointerp_build_prompt(ex)
            resp = client.chat.completions.create(
                model=cfg.evals.auto_interp.chat_model,
                temperature=cfg.evals.auto_interp.chat_model_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": cfg.evals.auto_interp.chat_system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
            )
            reply = resp.choices[0].message.content.strip()
            pred = autointerp_extract_digit(reply)
            predictions.append(pred)
            if pred == ex["answer"]:
                correct += 1
            bar.set_postfix(acc=f"{correct / i:.3f}")
            time.sleep(cfg.evals.auto_interp.sleep_sec)

        metrics = autointerp_evaluate(predictions, dataset)
        print("Evaluation metrics:", metrics)

    return eval_auto_interp


def toxicity():
    def eval_toxicity(cfg):
        model, tokenizer = init_model_tokenizer(cfg.model)

        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=os.getenv('GOOGLE_API_KEY'),
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        dataset = load_dataset(
            cfg.evals.toxicity.dataset_name,
            split=cfg.evals.toxicity.split
        )

        challenging_dataset = dataset.filter(
            lambda example: example["challenging"]
        )

        prompt_only = challenging_dataset.map(
            preprocess_to_perspective_message,
            batched=False,
            remove_columns=dataset.column_names
        )

        full_outputs = []
        completions_only = []

        pad_id = tokenizer.pad_token_id
        batch_size = cfg.evals.toxicity.batch_size

        for start in tqdm(
            range(0, len(prompt_only), batch_size),
            desc="Generating"
        ):
            # ── Slice a (shallow‑copied) mini‑dataset and pull the column as a python list
            batch_messages = prompt_only[start: start + batch_size]["messages"]

            # ── Convert each conversation into a single prompt string
            batch_prompts = [
                tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)
                for msgs in batch_messages
            ]

            # ── Tokenise & push to GPU
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            # ── Autoregressive generation
            gen_ids = model.generate(**enc)

            # ── Decode full strings
            full_outputs.extend(
                tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True
                )
            )

            # ── Extract only the newly generated continuation
            for prompt_ids, output_ids in zip(enc["input_ids"], gen_ids):
                # Locate the first END‑OF‑TURN token in the *generated* sequence
                eot_positions = (
                    output_ids == cfg.evals.toxicity.end_of_turn_id
                ).nonzero(as_tuple=True)[0]
                if len(eot_positions) > 0:
                    first_eot = eot_positions[0].item()
                    # after the EOT
                    completion_ids = output_ids[first_eot + 1:]
                else:
                    # Fallback: trim the prompt length, same as before
                    prompt_ids_no_pad = prompt_ids[prompt_ids != pad_id]
                    completion_ids = output_ids[len(prompt_ids_no_pad):]

                completions_only.append(
                    # TODO: decode or batch decode?
                    tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    ).strip()
                )

            # optional: free VRAM early
            del enc, gen_ids

        model.cpu()

        if device == 'cuda':
            torch.cuda.empty_cache()

        if cfg.evals.toxicity.dump_generated:
            with open(cfg.evals.toxicity.dump_path, 'w+') as f:
                json.dump(
                    [full_outputs, completions_only], f
                )

        # TODO: make configurable by hydra
        requested_attributes = {
            "TOXICITY": {},
            # any other attributes you want
        }
        # Single-threaded example (no concurrency):

        toxicity_scores = []
        if cfg.evals.toxicity.dump_analysis:
            analysis_dump = {
                'summary': {},
                'details': []
            }
        for text in completions_only:
            response = analyze_text_toxicity_eval(
                text, requested_attributes, client
            )
            toxicity_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            toxicity_scores.append(toxicity_score)
            if cfg.evals.toxicity.dump_analysis:
                analysis_dump['details'].append({
                    'text': text,
                    'score': toxicity_score
                })
            # printing first 30 chars because terminal gets flooded otherwise
            print(f"Text: {repr(text)[:30]}...\n Score: {toxicity_score}\n")
        print(f"Overall Toxicity: {(np.array(toxicity_scores) > 0.5).mean()} ")
        print(toxicity_scores)
        if cfg.evals.toxicity.dump_analysis:
            analysis_dump['summary'] = {
                'overall_toxicity': (np.array(toxicity_scores) > 0.5).mean(),
                'toxicity_scores': toxicity_scores
            }

        if cfg.evals.toxicity.dump_analysis:
            # ensure the directory exists before dumping
            path = Path(cfg.evals.toxicity.dump_path)
            path.mkdir(parents=True, exist_ok=True)
            with open(
                os.path.join(
                    cfg.evals.toxicity.dump_path,
                    'toxicity_analysis.json'
                ), 'w+'
            ) as f:
                json.dump(analysis_dump, f)
        return toxicity_scores

    return eval_toxicity


# Final accuracy summary
# overall: 67.873%
# harmless: 62.069%
# helpful: 72.881%
# honest: 72.131%
# other: 62.791%


def instruction_following():
    def eval_instruction_following(cfg):
        model, tokenizer = init_model_tokenizer(cfg.model)
        evaluator = Evaluator(instruction_registry)
        input_examples = get_default_dataset("en")

        responses = {
            ex.prompt: model.generate(
                **tokenizer(
                    [ex.prompt],
                    return_tensors="pt"
                ).to(device)
            )
            for ex in tqdm(input_examples)
        }

        for key in responses:
            responses[key] = tokenizer.batch_decode(
                responses[key], skip_special_tokens=True
            )[0]

        report, all_outputs = evaluator.evaluate(input_examples, responses)
        print(report)

    return eval_instruction_following

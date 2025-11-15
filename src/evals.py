import hashlib
import json
import re
import time
from ifeval import Evaluator, instruction_registry, get_default_dataset
from googleapiclient import discovery
from collections import defaultdict
import numpy as np
from openai import OpenAI
from src.delphi_autointerp import delphi_collect_activations, delphi_collect_activations_causal, delphi_score
from src.models import TopKLoRALinearSTE
from src.models import TopKLoRALinear
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import (
    get_dataset_config_names,
    load_dataset,
    concatenate_datasets,
    Dataset as HFDataset,
)
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
from typing import cast
from src.utils import (
    analyze_text_toxicity_eval,
    build_metrics_eval_messages,
    preprocess_to_perspective_message
)


device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


def init_model_tokenizer_fixed(model_cfg):
    """Load model with PEFT-compatible TopK wrappers"""

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.adapter_checkpoint_dir,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model,
        torch_dtype="auto",
        device_map="cpu"
    )

    # Load the PEFT adapter (this should work now!)
    model = PeftModel.from_pretrained(
        model,
        model_cfg.adapter_checkpoint_dir,
        device_map="cpu",
        use_safetensors=True
    )

    # NOW wrap with TopK for inference
    wrapped_modules = {}
    replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and not isinstance(module, TopKLoRALinearSTE):
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            attr = name.split(".")[-1]

            # Use the PEFT-compatible wrapper
            wrapped = TopKLoRALinearSTE(
                base=module,
                layer_name=name,
                k=model_cfg.k,
                temperature=0.0,  # Not used in eval
                hard_eval=True,
                relu_latents=True,
                alpha_over_r=True,
                k_final=model_cfg.k,
                temperature_final=0.0,
                is_topk_experiment=True
            )
            setattr(parent, attr, wrapped)
            wrapped_modules[name] = wrapped
            replaced += 1

    print(f"Wrapped {replaced} LoRA modules with TopK for inference")
    model.to(device)
    model.eval()

    print("Sanity checking LoRA B weights...")
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinearSTE):
            b_max = module.B_module.weight.detach().abs().max()
            a_max = module.A_module.weight.detach().abs().max()
            assert b_max != 0, f"lora_B weights in {name} are all zero!"
            assert a_max != 0, f"lora_A weights in {name} are all zero!"
            print(f"{name}: B max = {b_max:.6f};  A max = {a_max:.6f}")

    return model, tokenizer, wrapped_modules



def metrics():
    def eval_metrics(cfg):
        model, tokenizer, _ = init_model_tokenizer_fixed(cfg.model)

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
                    print(
                        f"[EX {i}] GOLD={gold_idx} | logp_A={logp_A:.4f} | logp_B={logp_B:.4f} | pred={pred}")

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
        model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(
            cfg.model
        )
        if cfg.evals.auto_interp.collect_activations:
            print('Collecting activations...')
            torch.set_float32_matmul_precision('high')
            delphi_collect_activations(cfg, model, tokenizer, wrapped_modules)
        delphi_score(cfg, model, tokenizer, wrapped_modules)
        return

    return eval_auto_interp


def causal_auto_interp():
    def eval_causal_auto_interp(cfg):
        model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(
            cfg.model
        )
        # sanity check below -- PEFT has a weird bug
        # where it doesn't load the weights sometimes
        # and initialises them to random (matrix_A)
        # and zeros (matrix_B) instead
        print("Sanity checking LoRA B weights...")
        for name, module in model.named_modules():
            if isinstance(module, TopKLoRALinearSTE):
                b_max = module.B_module.weight.detach().abs().max()
                a_max = module.A_module.weight.detach().abs().max()
                assert b_max != 0, f"lora_B weights in {name} are all zero!"
                assert a_max != 0, f"lora_A weights in {name} are all zero!"
                print(f"{name}: B max = {b_max:.6f};  A max = {a_max:.6f}")

        if cfg.evals.causal_auto_interp.collect_activations:
            print('Collecting activations...')
            torch.set_float32_matmul_precision('high')
            delphi_collect_activations_causal(
                cfg, model, tokenizer, wrapped_modules
            )

        if cfg.evals.causal_auto_interp.score_activations:
            print('Scoring activations...')
            delphi_score(cfg, model, tokenizer, wrapped_modules)
        return

    return eval_causal_auto_interp


def toxicity():
    def eval_toxicity(cfg):
        """
        Runs toxicity evaluation and returns a list of toxicity scores.

        Returns:
            toxicity_scores (List[float]): A list of toxicity scores (floats between 0 and 1) for each generated completion,
            where higher values indicate greater toxicity as measured by the Perspective API.
        """
        if cfg.evals.toxicity.eval_base_model:
            print('Evaluating toxicity on the base model...')
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.base_model,
                use_fast=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.base_model,
                torch_dtype="auto",
                device_map="cpu"
            )

            if not getattr(tokenizer, "chat_template", None):
                logging.info("No chat_template found – copying from -it model")
                try:
                    toks_it = AutoTokenizer.from_pretrained(
                        cfg.model.model_it_name,
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

            model.to(device)
            model.eval()


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
                model.generation_config.eos_token_id = [
                    base_eos_token_id, eot_token_id]

            # Log the configuration
            print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
            print(f"EOS token ID(s): {model.generation_config.eos_token_id}")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            print('Evaluating toxicity on the adapter model...')
            model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(
                cfg.model
            )
            print(model)
            # assert False

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
                max_length=256,
                truncation=True,
            ).to(device)

            # ── Autoregressive generation
            with torch.no_grad():
                gen_ids = model.generate(**enc)

            # ── Extract only the newly generated continuation
            # Precompute prompt lengths for the batch (excluding pad tokens)
            prompt_lengths = (enc["input_ids"] != pad_id).sum(dim=1).tolist()
            for i, (prompt_ids, output_ids) in enumerate(zip(enc["input_ids"], gen_ids)):
                # Locate the first END‑OF‑TURN token in the *generated* sequence
                eot_positions = (
                    output_ids == cfg.evals.toxicity.end_of_turn_id
                ).nonzero(as_tuple=True)[0]
                if len(eot_positions) > 0:
                    first_eot = eot_positions[0].item()
                    # after the EOT
                    completion_ids = output_ids[first_eot + 1:]
                else:
                    # Fallback: trim the prompt length, using precomputed length
                    prompt_len = prompt_lengths[i]
                    completion_ids = output_ids[prompt_len:]

                completions_only.append(
                    # TODO: decode or batch decode?
                    tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    ).strip()
                )

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
            os.makedirs(
                cfg.evals.toxicity.dump_path,
                exist_ok=True
            )
            with open(cfg.evals.toxicity.dump_path + '/generated.json', 'w+') as f:
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
        for idx, text in enumerate(
            tqdm(completions_only, desc='Collecting toxicity evals')
        ):
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
            # print(f"Text: {repr(text)[:30]}...\n Score: {toxicity_score}\n")
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
            if not cfg.evals.toxicity.eval_base_model:
                with open(cfg.model.adapter_checkpoint_dir + '/../hparams.json', 'r') as f:
                    model_hparams = json.load(f)
                r = model_hparams['lora_topk']['r']
                k = model_hparams['lora_topk']['k_final']

            # ensure the directory exists before dumping
            os.makedirs(
                f'{cfg.evals.toxicity.dump_path}_{"base" if cfg.evals.toxicity.eval_base_model else f"adapter_{r}_{k}"}',
                exist_ok=True
            )
            # dump the analysis
            print(
                f"Dumping toxicity analysis to: {cfg.evals.toxicity.dump_path}_{'base' if cfg.evals.toxicity.eval_base_model else f'adapter_{r}_{k}'}"
            )
            with open(
                os.path.join(
                    f'{cfg.evals.toxicity.dump_path}_{"base" if cfg.evals.toxicity.eval_base_model else f"adapter_{r}_{k}"}',
                    'toxicity_analysis.json'
                ), 'w+'
            ) as f:
                json.dump(analysis_dump, f)
        return toxicity_scores

    return eval_toxicity


def instruction_following():
    def eval_instruction_following(cfg):
        # model, tokenizer = init_model_tokenizer(cfg.model)
        if cfg.evals.instruction_following.eval_base_model:
            print('Evaluating instruction following on the base model...')
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.base_model,
                use_fast=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.base_model,
                torch_dtype="auto",
                device_map="cpu"
            )

            if not getattr(tokenizer, "chat_template", None):
                logging.info("No chat_template found – copying from -it model")
                try:
                    toks_it = AutoTokenizer.from_pretrained(
                        cfg.model.model_it_name,
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

            model.to(device)
            model.eval()

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
                model.generation_config.eos_token_id = [
                    base_eos_token_id, eot_token_id]

            # Log the configuration
            print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
            print(f"EOS token ID(s): {model.generation_config.eos_token_id}")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            print('Evaluating instruction following on the adapter model...')
            model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(
                cfg.model
            )

        evaluator = Evaluator(instruction_registry)
        input_examples = get_default_dataset("en")

        prompts_with_text = []
        for ex in input_examples:
            if getattr(tokenizer, "apply_chat_template", None):
                try:
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": ex.prompt}],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                except TypeError:
                    rendered = ex.prompt
            else:
                rendered = ex.prompt
            prompts_with_text.append((ex.prompt, rendered))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        pad_id = tokenizer.pad_token_id
        batch_size = getattr(
            cfg.evals.instruction_following, "batch_size", 100
        )
        max_length = getattr(
            cfg.evals.instruction_following, "max_length", 512
        )
        max_new_tokens = getattr(
            cfg.evals.instruction_following, "max_new_tokens", 256
        )
        do_sample = getattr(
            cfg.evals.instruction_following, "do_sample", False
        )
        temperature = getattr(
            cfg.evals.instruction_following, "temperature", 0.0
        )
        top_p = getattr(
            cfg.evals.instruction_following, "top_p", 1.0
        )
        repetition_penalty = getattr(
            cfg.evals.instruction_following, "repetition_penalty", 1.0
        )

        responses = {}
        for start in tqdm(range(0, len(prompts_with_text), batch_size), desc="Generating"):
            batch = prompts_with_text[start:start + batch_size]
            batch_texts = [rendered for _, rendered in batch]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            prompt_lengths = (enc["input_ids"] != pad_id).sum(dim=1)

            gen_kwargs = dict(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # temperature=temperature if do_sample else 0.0,
                top_p=top_p if do_sample else 1.0,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_id,
                eos_token_id=getattr(
                    model.generation_config, "eos_token_id", tokenizer.eos_token_id
                ),
            )

            with torch.no_grad():
                generated = model.generate(**gen_kwargs)

            for idx, (raw_prompt, _) in enumerate(batch):
                output_ids = generated[idx]
                prompt_len = prompt_lengths[idx].item()
                continuation_ids = output_ids[prompt_len:]
                completion = tokenizer.decode(
                    continuation_ids, skip_special_tokens=True
                ).strip()
                responses[raw_prompt] = completion

            del enc, generated
            if device == 'cuda':
                torch.cuda.empty_cache()

        report, all_outputs = evaluator.evaluate(input_examples, responses)
        print(report)

        # Save report to file
        output_dir = Path(cfg.evals.instruction_following.get(
            "output_dir", "if_outputs/instruction_following"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename based on model type
        if cfg.evals.instruction_following.eval_base_model:
            filename = "report_base_model.json"
        else:
            # Load hyperparameters if available
            try:
                with open(cfg.model.adapter_checkpoint_dir + '/../hparams.json', 'r') as f:
                    model_hparams = json.load(f)
                r = model_hparams['lora_topk']['r']
                k = model_hparams['lora_topk']['k_final']
                filename = f"report_adapter_{r}_{k}.json"
            except FileNotFoundError:
                filename = "report_adapter.json"

        report_path = output_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {report_path}")

    return eval_instruction_following

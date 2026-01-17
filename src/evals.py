import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import (
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from googleapiclient import discovery
from ifeval import Evaluator, get_default_dataset, instruction_registry
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.delphi_autointerp import (
    delphi_collect_activations,
    delphi_collect_activations_causal,
    delphi_score,
)
from src.models import TopKLoRALinearSTE
from src.utils import (
    analyze_text_toxicity_eval,
    build_metrics_eval_messages,
    configure_eos_eot,
    ensure_chat_template_and_special_tokens,
    format_adapter_suffix,
    generate_completions_from_prompts,
    preprocess_to_perspective_message,
    wrap_topk_lora_modules,
    write_json,
    wikitext_detokenizer,
)
from src.autointerp_framework_hh import run_autointerp_framework

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)


def init_model_tokenizer_fixed(model_cfg):
    """Load model with PEFT-compatible TopK wrappers"""

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.adapter_checkpoint_dir, use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model, torch_dtype="auto", device_map="cpu"
    )

    # Load the PEFT adapter (this should work now!)
    model = PeftModel.from_pretrained(
        model, model_cfg.adapter_checkpoint_dir, device_map="cpu", use_safetensors=True
    )

    # NOW wrap with TopK for inference
    replaced, wrapped_modules = wrap_topk_lora_modules(
        model,
        k=model_cfg.k,
        temperature=0.0,
        temperature_schedule="constant",
        k_schedule="constant",
        k_final=model_cfg.k,
        temperature_final=0.0,
        is_topk_experiment=True,
        set_train=False,
    )

    print(f"Wrapped {replaced} LoRA modules with TopK for inference")
    model.to(device)
    model.eval()
    print(f"Loaded model dtype: {model.dtype}")

    print("Sanity checking LoRA B weights...")
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinearSTE):
            b_max = module.B_module.weight.detach().abs().max()
            a_max = module.A_module.weight.detach().abs().max()
            assert b_max != 0, f"lora_B weights in {name} are all zero!"
            assert a_max != 0, f"lora_A weights in {name} are all zero!"
            print(f"{name}: B max = {b_max:.6f};  A max = {a_max:.6f}")

    return model, tokenizer, wrapped_modules


def load_base_model_for_eval(cfg):
    """Load base model/tokenizer and apply shared eval-time setup."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        torch_dtype="auto",
        device_map="cpu",
    )
    print(f"Loaded model dtype: {model.dtype}")

    model.to(device)
    model.eval()

    ensure_chat_template_and_special_tokens(
        tokenizer,
        model,
        cfg.model.model_it_name,
    )
    eot_token, eot_token_id = configure_eos_eot(tokenizer, model)

    # Log the configuration
    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS token ID(s): {model.generation_config.eos_token_id}")

    return model, tokenizer


def metrics():
    def eval_metrics(cfg):
        model, tokenizer, _ = init_model_tokenizer_fixed(cfg.model)

        configs = get_dataset_config_names("HuggingFaceH4/hhh_alignment")
        parts = []
        for cfg_ in configs:
            ds = load_dataset("HuggingFaceH4/hhh_alignment", cfg_, split="test")
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
                    q,
                    choices[0],
                    choices[1],
                )

                # Render prompt up to "Reply A:" and then append one of the replies
                base_prompt = tokenizer.apply_chat_template(
                    base_prompt_raw, tokenize=False, add_generation_prompt=False
                )

                # Create inputs for scoring reply A
                full_text_a = base_prompt + choices[0]
                full_text_b = base_prompt + choices[1]

                def logprob_of(text):
                    enc = tokenizer(
                        text, return_tensors="pt", add_special_tokens=False
                    ).to(model.device)
                    input_ids = enc.input_ids

                    with torch.no_grad():
                        out = model(input_ids)
                        logits = out.logits

                    # Compute token-wise log probs
                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]

                    log_probs = F.softmax(shift_logits, dim=-1)
                    selected_logprobs = log_probs.gather(
                        2, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)

                    # Sum log probs from the start of the reply only (not prompt)
                    reply_start = len(
                        tokenizer(base_prompt, add_special_tokens=False)["input_ids"]
                    )
                    reply_logprob = selected_logprobs[:, reply_start:].sum()

                    return reply_logprob.item()

                logp_A = logprob_of(full_text_a)
                logp_B = logprob_of(full_text_b)

                pred = 1 if logp_B > logp_A else 0
                if i % 10 == 0:
                    print(
                        f"[EX {i}] GOLD={gold_idx} | logp_A={logp_A:.4f} | logp_B={logp_B:.4f} | pred={pred}"
                    )

                metric_global.add(prediction=pred, reference=gold_idx)
                metrics_by_subset[ex["subset"]].add(prediction=pred, reference=gold_idx)

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
        model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(cfg.model)
        if cfg.evals.auto_interp.collect_activations:
            print("Collecting activations...")
            torch.set_float32_matmul_precision("high")
            delphi_collect_activations(cfg, model, tokenizer, wrapped_modules)
        delphi_score(cfg, model, tokenizer, wrapped_modules)
        return

    return eval_auto_interp


def causal_auto_interp():
    def eval_causal_auto_interp(cfg):
        model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(cfg.model)
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
            print("Collecting activations...")
            torch.set_float32_matmul_precision("high")
            delphi_collect_activations_causal(cfg, model, tokenizer, wrapped_modules)

        if cfg.evals.causal_auto_interp.score_activations:
            print("Scoring activations...")
            delphi_score(cfg, model, tokenizer, wrapped_modules)
        return

    return eval_causal_auto_interp


def causal_autointerp_framework():
    def eval_causal_autointerp_framework(cfg):
        model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(cfg.model)
        run_autointerp_framework(cfg, model, tokenizer, wrapped_modules)
        return

    return eval_causal_autointerp_framework


def toxicity():
    def eval_toxicity(cfg):
        """
        Runs toxicity evaluation and returns a list of toxicity scores.

        Returns:
            toxicity_scores (List[float]): A list of toxicity scores (floats between 0 and 1) for each generated completion,
            where higher values indicate greater toxicity as measured by the Perspective API.
        """
        if cfg.evals.toxicity.eval_base_model:
            print("Evaluating toxicity on the base model...")
            model, tokenizer = load_base_model_for_eval(cfg)
        else:
            print("Evaluating toxicity on the adapter model...")
            model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(cfg.model)
            print(model)
            # assert False

        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=os.getenv("GOOGLE_API_KEY"),
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        dataset = load_dataset(
            cfg.evals.toxicity.dataset_name, split=cfg.evals.toxicity.split
        )

        challenging_dataset = dataset.filter(lambda example: example["challenging"])

        prompt_only = challenging_dataset.map(
            preprocess_to_perspective_message,
            batched=False,
            remove_columns=dataset.column_names,
        )

        full_outputs = []
        completions_only = []

        batch_size = cfg.evals.toxicity.batch_size

        for start in tqdm(range(0, len(prompt_only), batch_size), desc="Generating"):
            # ── Slice a (shallow‑copied) mini‑dataset and pull the column as a python list
            batch_messages = prompt_only[start : start + batch_size]["messages"]

            # ── Convert each conversation into a single prompt string
            batch_prompts = [
                tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False
                )
                for msgs in batch_messages
            ]

            batch_completions = generate_completions_from_prompts(
                model,
                tokenizer,
                batch_prompts,
                device=device,
                max_length=256,
                truncation=True,
                end_of_turn_id=cfg.evals.toxicity.end_of_turn_id,
            )
            completions_only.extend(batch_completions)
            for prompt, completion in zip(batch_prompts, batch_completions):
                full_outputs.append(prompt + completion)

        model.cpu()

        if device == "cuda":
            torch.cuda.empty_cache()

        if cfg.evals.toxicity.dump_generated:
            write_json(
                os.path.join(cfg.evals.toxicity.dump_path, "generated.json"),
                [full_outputs, completions_only],
            )

        # TODO: make configurable by hydra
        requested_attributes = {
            "TOXICITY": {},
            # any other attributes you want
        }
        # Single-threaded example (no concurrency):

        toxicity_scores = []
        if cfg.evals.toxicity.dump_analysis:
            analysis_dump = {"summary": {}, "details": []}
        for idx, text in enumerate(
            tqdm(completions_only, desc="Collecting toxicity evals")
        ):
            response = analyze_text_toxicity_eval(text, requested_attributes, client)
            toxicity_score = response["attributeScores"]["TOXICITY"]["summaryScore"][
                "value"
            ]
            toxicity_scores.append(toxicity_score)
            if cfg.evals.toxicity.dump_analysis:
                analysis_dump["details"].append({"text": text, "score": toxicity_score})
            # printing first 30 chars because terminal gets flooded otherwise
            # print(f"Text: {repr(text)[:30]}...\n Score: {toxicity_score}\n")
        print(f"Overall Toxicity: {(np.array(toxicity_scores) > 0.5).mean()} ")
        print(toxicity_scores)
        if cfg.evals.toxicity.dump_analysis:
            analysis_dump["summary"] = {
                "overall_toxicity": (np.array(toxicity_scores) > 0.5).mean(),
                "toxicity_scores": toxicity_scores,
            }

        if cfg.evals.toxicity.dump_analysis:
            model_tag = (
                "base"
                if cfg.evals.toxicity.eval_base_model
                else format_adapter_suffix(cfg.model.adapter_checkpoint_dir)
            )
            analysis_dir = f"{cfg.evals.toxicity.dump_path}_{model_tag}"
            print(f"Dumping toxicity analysis to: {analysis_dir}")
            write_json(
                os.path.join(analysis_dir, "toxicity_analysis.json"),
                analysis_dump,
            )
        return toxicity_scores

    return eval_toxicity


def instruction_following():
    def eval_instruction_following(cfg):
        # model, tokenizer = init_model_tokenizer(cfg.model)
        if cfg.evals.instruction_following.eval_base_model:
            print("Evaluating instruction following on the base model...")
            model, tokenizer = load_base_model_for_eval(cfg)
        else:
            print("Evaluating instruction following on the adapter model...")
            model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(cfg.model)

        evaluator = Evaluator(instruction_registry)
        input_examples = get_default_dataset("en")

        prompts_with_text = []
        for ex in input_examples:
            if getattr(tokenizer, "apply_chat_template", None):
                try:
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": ex.prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except TypeError:
                    rendered = ex.prompt
            else:
                rendered = ex.prompt
            prompts_with_text.append((ex.prompt, rendered))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        pad_id = tokenizer.pad_token_id
        batch_size = getattr(cfg.evals.instruction_following, "batch_size", 100)
        max_length = getattr(cfg.evals.instruction_following, "max_length", 512)
        max_new_tokens = getattr(cfg.evals.instruction_following, "max_new_tokens", 256)
        repetition_penalty = getattr(
            cfg.evals.instruction_following, "repetition_penalty", 1.0
        )
        max_samples = getattr(
            cfg.evals.instruction_following, "max_samples", -1
        )  # Added possibility of choosing less samples. -1 means all

        if max_samples == -1:
            max_samples = len(prompts_with_text)

        responses = {}
        for start in tqdm(range(0, max_samples, batch_size), desc="Generating"):
            batch = prompts_with_text[start : start + batch_size]
            batch_texts = [rendered for _, rendered in batch]

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                # this should in principle be set to False because we want to have a
                # deterministic behaviour -- greedy decoding decreases noise in eval results
                do_sample=False,
                # temperature=temperature if do_sample else 0.0,
                top_p=1.0,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_id,
                eos_token_id=getattr(
                    model.generation_config, "eos_token_id", tokenizer.eos_token_id
                ),
            )

            completions = generate_completions_from_prompts(
                model,
                tokenizer,
                batch_texts,
                device=device,
                max_length=max_length,
                truncation=True,
                gen_kwargs=gen_kwargs,
            )

            for idx, (raw_prompt, _) in enumerate(batch):
                responses[raw_prompt] = completions[idx]
            if device == "cuda":
                torch.cuda.empty_cache()

        report, all_outputs = evaluator.evaluate(input_examples, responses)
        print(report)

        # Save report to file
        output_dir = Path(
            cfg.evals.instruction_following.get(
                "output_dir", "if_outputs/instruction_following"
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        model_tag = (
            "base_model"
            if cfg.evals.instruction_following.eval_base_model
            else format_adapter_suffix(cfg.model.adapter_checkpoint_dir)
        )
        report_path = output_dir / f"report_{model_tag}.json"
        write_json(str(report_path), report)

        print(f"Report saved to: {report_path}")

    return eval_instruction_following


def perplexity():
    def eval_perplexity(cfg):
        if cfg.evals.perplexity.eval_base_model:
            logging.info("Evaluating perplexity on the base model...")
            model, tokenizer = load_base_model_for_eval(cfg)
            logging.info(
                "Note that the chat template is NOT applied for perplexity eval."
            )
        else:
            logging.info("Evaluating perplexity on the adapter model...")
            model, tokenizer, _ = init_model_tokenizer_fixed(cfg.model)

        if cfg.evals.perplexity.dataset_config is None:
            raise ValueError(
                "Please specify dataset_config for perplexity eval (e.g., 'wikitext-2-raw-v1')."
            )

        dataset = load_dataset(
            cfg.evals.perplexity.dataset_name,
            cfg.evals.perplexity.dataset_config,
            split=cfg.evals.perplexity.split,
        )
        text_column = cfg.evals.perplexity.text_column
        if cfg.evals.perplexity.max_samples > 0:
            dataset = dataset.select(range(cfg.evals.perplexity.max_samples))

        processing_func = (
            wikitext_detokenizer
            if cfg.evals.perplexity.dataset_name == "wikitext"
            else (lambda x: x)
        )
        texts = [
            processing_func(t) for t in dataset[text_column] if t and not t.isspace()
        ]
        enc = tokenizer("\n\n".join(texts), return_tensors="pt")
        input_ids = enc.input_ids

        # Show the decoded first 100 tokens
        logging.info("Decoded first 100 tokens:")
        preview_tokens = input_ids[0, : min(100, input_ids.size(1))]
        logging.info(tokenizer.decode(preview_tokens))

        logging.info(f"Total tokens in input: {input_ids.size(1)}")
        max_tokens = cfg.evals.perplexity.max_tokens
        logging.info(f"Using {max_tokens} max tokens for scoring.")
        if max_tokens > 0 and input_ids.size(1) > max_tokens:
            input_ids = input_ids[:, :max_tokens]

        model_max_length = getattr(model.config, "max_position_embeddings", None)
        block_size = cfg.evals.perplexity.block_size
        if model_max_length is not None:
            block_size = min(block_size, model_max_length)
        block_size = min(block_size, tokenizer.model_max_length)
        stride = (
            block_size
            if cfg.evals.perplexity.stride is None
            else cfg.evals.perplexity.stride
        )
        if stride == 0:
            raise ValueError("Stride cannot be zero.")

        total_nll = 0.0
        total_tokens = 0
        for i in tqdm(range(0, input_ids.size(1), stride), desc="Scoring"):
            begin_loc = max(i + stride - block_size, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i
            input_ids_slice = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids_slice.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids_slice, labels=target_ids)
                total_nll += outputs.loss.item() * trg_len
                total_tokens += trg_len

        ppl = math.exp(total_nll / max(total_tokens, 1))
        logging.info(f"Perplexity: {ppl:.4f} over {total_tokens} tokens.")
        report = {
            "perplexity": ppl,
            "tokens_scored": total_tokens,
            "dataset_name": cfg.evals.perplexity.dataset_name,
            "dataset_config": cfg.evals.perplexity.dataset_config,
            "split": cfg.evals.perplexity.split,
            "block_size": block_size,
            "stride": stride,
            "max_tokens": max_tokens,
        }
        print(report)

        output_dir = Path(
            cfg.evals.perplexity.get("output_dir", "eval_outputs/perplexity")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model_tag = (
            "base_model"
            if cfg.evals.perplexity.eval_base_model
            else format_adapter_suffix(cfg.model.adapter_checkpoint_dir)
        )
        report_path = output_dir / f"report_{model_tag}.json"
        write_json(str(report_path), report)
        print(f"Report saved to: {report_path}")

        return report

    return eval_perplexity

from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import (
    get_dataset_config_names,
    load_dataset,
    concatenate_datasets
)
from tqdm import tqdm
import evaluate
import torch

from src.utils import build_metrics_eval_messages


device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


def metrics():
    def eval_metrics(cfg):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.evals.metrics.model_path
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.evals.metrics.model_path
        ).to(device)

        if 'gemma' in model.config._name_or_path:
            model.generation_config.eos_token_id = [1, 107]

        model.eval()
        configs = get_dataset_config_names("HuggingFaceH4/hhh_alignment")
        parts = []
        for cfg in configs:
            ds = load_dataset("HuggingFaceH4/hhh_alignment", cfg, split="test")
            ds = ds.add_column("subset", [cfg] * len(ds))
            parts.append(ds)
            print(f"Loaded {cfg:8} subset with {len(ds)} rows.")
        hhh_all = concatenate_datasets(parts)
        assert len(hhh_all) == 221

        print(f"Total rows: {len(hhh_all)}")
        print("-" * 60)

        metric_global = evaluate.load("accuracy")
        metrics_by_subset = defaultdict(lambda: evaluate.load("accuracy"))

        tok_id_A = tokenizer.convert_tokens_to_ids("A")
        tok_id_B = tokenizer.convert_tokens_to_ids("B")

        with torch.no_grad():
            for i, ex in tqdm(enumerate(hhh_all)):
                q = ex["input"]
                choices = ex["targets"]["choices"]
                gold_idx = ex["targets"]["labels"].index(1)

                msgs = build_metrics_eval_messages(q, choices[0], choices[1])
                input_ids = tokenizer.apply_chat_template(
                    msgs,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(device)

                logits = model(input_ids).logits
                last_logits = logits[0, -1]

                logp_A = last_logits[tok_id_A].item()
                logp_B = last_logits[tok_id_B].item()
                pred = 1 if logp_B > logp_A else 0

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
        # tokenizer = AutoTokenizer.from_pretrained(
        #     cfg.checkpoint_dir,
        #     use_fast=True
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     cfg.checkpoint_dir,
        #     torch_dtype="auto"
        # ).to(device)
        # model.eval()

        # topk_store = defaultdict(
        #     lambda: defaultdict(lambda: {"pos": [], "neg": []})
        # )
        # example_counter = 0        # global running index over all examples

        print('evaluating auto interp')
    return eval_auto_interp


def perspective():
    def eval_perspective(cfg):
        print('evaluating perspective')
    return eval_perspective

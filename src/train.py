import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from src.models import TopKLoRALinear, MemoryClearCallback
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer, DPOConfig
import logging
import wandb

class LoggingDPOTrainer(DPOTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,               # catch extra Trainer args
    ):
        # forward ALL kwargs (except return_outputs which we override) to super
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            **kwargs
        )

        logp_chosen   = outputs["logps/chosen"]
        logp_rejected = outputs["logps/rejected"]

        # 3) Compute their difference (batch-mean if needed)
        #    Convert to float so WandB can log it
        if isinstance(logp_chosen, torch.Tensor):
            logp_margin = (logp_chosen - logp_rejected).mean().item()
        else:
            logp_margin = float(logp_chosen - logp_rejected)

        # 4) Log it under a custom name
        self.log({"train/logp_margin": logp_margin})

        # 5) Return in the same format the parent expects
        return (loss, outputs) if return_outputs else loss


def run_dpo(cfg, model, quant_cfg):


    def get_dataset(test=False):
        CACHE_DIR = os.getcwd()+'/cache'
        if test:
            dataset = load_dataset("stanfordnlp/shp", cache_dir=CACHE_DIR, split="test")
        else:
            dataset = load_dataset("stanfordnlp/shp", cache_dir=CACHE_DIR, split="train")

        original_columns = dataset.column_names
        def return_prompt_and_responses(samples):
            # build the same prompt from history every time
            prompts = [
                f"###Question:\n{h}\n\n###Answer:\n"
                for h in samples["history"]
            ]

            # chosen vs. rejected based purely on the label
            chosen = [
                A if lab == 1 else B
                for lab, A, B in zip(samples["labels"],
                                    samples["human_ref_A"],
                                    samples["human_ref_B"])
            ]
            rejected = [
                B if lab == 1 else A
                for lab, A, B in zip(samples["labels"],
                                    samples["human_ref_A"],
                                    samples["human_ref_B"])
            ]

            return {"prompt": prompts, "chosen": chosen, "rejected": rejected}
        return dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=original_columns,
        )    


    MAX_LENGTH = 1024

    train_dataset = get_dataset()

    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= MAX_LENGTH
        and len(x["prompt"]) + len(x["rejected"]) <= MAX_LENGTH
    )
    eval_dataset = get_dataset(test=True).take(150)

    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= MAX_LENGTH
        and len(x["prompt"]) + len(x["rejected"]) <= MAX_LENGTH
    )


    OUTPUT_DIR = "./sanity_check/gemma-2-2b-dpo-lora"
    SFT_DIR    = f'experiments/merged/{cfg.training.model.model_name}_sft'
    # SFT_DIR    = 'meta-llama/Llama-3.2-1B'

    # 1) Quant config (you had a placeholder)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2) Tokenizer (from your SFT merge)
    tokenizer = AutoTokenizer.from_pretrained(SFT_DIR)

    policy_model = AutoModelForCausalLM.from_pretrained(
        SFT_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    policy_model = prepare_model_for_kbit_training(policy_model)
    lora_config = LoraConfig(
        r=1024,
        lora_alpha=32,
        target_modules=[
            'layers.11.self_attn.q_proj',
            'layers.11.self_attn.k_proj',
            'layers.11.self_attn.v_proj',
            'layers.11.self_attn.o_proj',
            'layers.11.mlp.gate_proj',
            'layers.11.mlp.up_proj',
            'layers.11.mlp.down_proj'
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(policy_model, lora_config)
    model.config.use_cache = False
    # freeze & clone for ref_model
    ref_model = AutoModelForCausalLM.from_pretrained(
        SFT_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens", [tokenizer.eos_token])[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )

    # Update generation config for both policy and ref models
    prev_eos_token_id = policy_model.generation_config.eos_token_id
    policy_model.generation_config.eos_token_id = [prev_eos_token_id, tokenizer.convert_tokens_to_ids(eot_token)]
    ref_model.generation_config.eos_token_id = policy_model.generation_config.eos_token_id



    # 4) Attach fresh LoRA everywhere
    # Inject Top-k wrappers
    replaced = 0
    topk_k = 8
    for name, module in model.named_modules():
        if getattr(module, "lora_A", None) is None:
            continue
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        attr = name.split(".")[-1]
        setattr(parent, attr, TopKLoRALinear(
            module,
            layer_name=name,
            r=module.r,
            alpha=module.lora_alpha,
            k=topk_k,
        ))
        replaced += 1
    logging.info("TopKLoRALinear injected in %d layers", replaced)

    # print(model)
    model.print_trainable_parameters()


    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 6) DPOConfig (lower lr, low β, linear decay)
    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # still eff_bs=16
        learning_rate=5e-6,             # x2–3
        beta=0.05,                      # stronger KL
        lr_scheduler_type="linear",
        warmup_steps=10,                # actual warmup
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,                  # more frequent feedback
        bf16=True,
        report_to="wandb",
        run_name="gemma-2-2b-dpo-lora",
        remove_unused_columns=False,
    )

    # 7) Trainer with explicit ref_model
    trainer = LoggingDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
        processing_class=tokenizer,
        callbacks=[MemoryClearCallback()],
    )

    if trainer.optimizer is not None:
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

    # Train
    trainer.train()

    # ── Unwrap Top-k wrappers before saving ─────────────────────────────
    unwrapped = 0
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinear):
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            attr = name.split(".")[-1]
            setattr(parent, attr, module.lora_module)
            unwrapped += 1
    logging.info("Reverted %d TopK wrappers back to LoraLayer", unwrapped)

    # Save adapter
    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    out_path = os.path.join(f'experiments/{model_str}_dpo', "final_adapter")
    trainer.save_model(out_path)
    logging.info("Adapter saved to %s", out_path)
    wandb.finish()

    return trainer.model
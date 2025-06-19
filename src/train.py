import wandb
import torch
from trl import (
    SFTTrainer,
    SFTConfig,
    DPOConfig,
    DPOTrainer,
    setup_chat_format,
    extract_prompt
)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import peft
import time
import os
from src.models import TopKLoRALinear
from src.utils import build_quant_config, get_conversational_dataset, hh_rlhf_preprocess_to_messages, is_valid_dpo_pair, merge_lora_adapter, preprocess_to_messages, violates_alternation
from peft import PeftModelForCausalLM
import numpy as np
import logging

device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}


def run_sft(cfg, peft_config, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.training.model.huggingface_model_id
    ).to(device)

    try:
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=cfg.training.model.huggingface_model_id,
            # name of the adapter is the dataset name
            adapter_name=cfg.training.sft_dataset.name,
            is_trainable=True
        ).to(device)
    except ValueError:
        pass

    try:
        model, tokenizer = setup_chat_format(
            model=model, tokenizer=tokenizer
        )
    except ValueError:
        pass

    train_dataset, eval_dataset = get_conversational_dataset(
        cfg.training.sft_dataset.huggingface_dataset_id, tokenizer
    )
    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    training_args = SFTConfig(
        output_dir=f'experiments/{model_str}_sft',
        logging_dir=f'experiments/{model_str}_sft/logs',
        learning_rate=cfg.training.sft.lr,
        eval_steps=cfg.training.sft.eval_steps,
        max_steps=cfg.training.sft.max_steps,
        logging_steps=cfg.logger.logging_steps,
        report_to=cfg.logger.report_to,
        gradient_checkpointing=cfg.training.sft.gradient_checkpointing,
        per_device_train_batch_size=cfg.training.sft.batch_size_train,
        per_device_eval_batch_size=cfg.training.sft.batch_size_eval,
        num_train_epochs=cfg.training.sft.num_epochs,
        weight_decay=cfg.training.sft.weight_decay,
        push_to_hub=cfg.training.sft.push_to_hub,
        save_steps=cfg.training.sft.save_steps,
        lr_scheduler_type=cfg.lr_scheduler.type,
        do_eval=cfg.training.sft.do_eval,
        eval_strategy='steps',
        save_strategy='steps'
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        peft_config=peft_config
    )

    trainer.train()
    trainer.model.save_pretrained(
        f'adapters/sft/{cfg.training.sft_experiment.lora.r}-{cfg.training.sft_experiment.lora.alpha}-'
        f'{cfg.training.sft_experiment.lora.dropout}/{cfg.training.sft_dataset.name}/'
        f'{"-".join(cfg.training.sft_experiment.lora.target_modules)}'
    )

    return trainer.model


def run_dpo(cfg, peft_config, tokenizer, model):
    train_dataset = load_dataset(
        cfg.dataset_dpo.huggingface_dataset_id,
        split="train"
    )

    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    training_args = DPOConfig(
        output_dir=f'experiments/{model_str}_dpo',
        learning_rate=cfg.training.dpo.lr,
        max_steps=cfg.training.dpo.max_steps,
        logging_steps=cfg.logger.logging_steps,
        report_to=cfg.logger.report_to,
        gradient_checkpointing=cfg.training.dpo.gradient_checkpointing,
        per_device_train_batch_size=cfg.training.dpo.batch_size_train,
        per_device_eval_batch_size=cfg.training.dpo.batch_size_eval,
        num_train_epochs=cfg.training.dpo.num_epochs,
        weight_decay=cfg.training.dpo.weight_decay,
        push_to_hub=cfg.training.dpo.push_to_hub,
        save_steps=cfg.training.dpo.save_steps,
        lr_scheduler_type=cfg.lr_scheduler.type,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config
    )

    trainer.train()
    trainer.model.save_pretrained(
        f'adapters/dpo/{cfg.training.sft_experiment.lora.r}-{cfg.training.sft_experiment.lora.alpha}-'
        f'{cfg.training.sft_experiment.lora.dropout}/{cfg.training.sft_dataset.name}/'
        f'{"-".join(cfg.training.sft_experiment.lora.target_modules)}'
    )


def lukas_sft(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.training.model.model_name, fast=False
    )

    quant_cfg = build_quant_config(
        cfg.training.quantization
    )
    logging.info("Using quantisation: %s", quant_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.training.model.model_name,
        # quantization doesn't work on Apple Metal
        # quantization_config=quant_cfg if device != 'mps' else None,
    ).to(device)

    if 'gemma' in cfg.training.model.name:
        tokenizer.padding_side = 'right'

    # Ensure chat template exists; attempt to copy from -it model.
    if not getattr(tokenizer, "chat_template", None):
        logging.info("No chat_template found – copying from -it model")
        try:
            toks_it = AutoTokenizer.from_pretrained(
                cfg.training.model.model_it_name,
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

    dataset = load_dataset(
        cfg.training.sft_dataset.huggingface_dataset_id,
        split=cfg.training.sft_dataset.split
    )

    message_dataset = dataset.map(
        preprocess_to_messages,
        remove_columns=dataset.column_names,
    )

    # TODO: why train test split if we set split in load_dataset?
    message_dataset = message_dataset.train_test_split(test_size=0.1)
    train_dataset, val_dataset = message_dataset["train"], message_dataset["test"]

    # Determine EOT token (Gemma uses second additional special token)
    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens", [tokenizer.eos_token])[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )
    prev_eos_token_id = model.generation_config.eos_token_id
    model.generation_config.eos_token_id = [prev_eos_token_id, eot_token]

    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    training_args = SFTConfig(
        packing=cfg.training.sft.packing,
        # changes the tokenizers eos token to eot and the google gemma-2b-it doesn't have that will default to the list [...] in the tokenizer bos and end of turn
        eos_token=eot_token,
        completion_only_loss=cfg.training.sft.completion_only_loss,
        max_seq_length=cfg.training.sft.max_seq_length,
        num_train_epochs=cfg.training.sft.num_epochs,
        per_device_train_batch_size=cfg.training.sft.batch_size_train,
        gradient_accumulation_steps=cfg.training.sft.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.sft.gradient_checkpointing,
        # optim=cfg.training.sft.optim,
        learning_rate=cfg.training.sft.lr,
        warmup_ratio=cfg.training.sft.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler.type,
        bf16=cfg.training.sft.bf16,
        fp16=cfg.training.sft.fp16,
        max_grad_norm=cfg.training.sft.max_grad_norm,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
        logging_steps=cfg.logger.logging_steps,
        save_strategy=cfg.training.sft.save_strategy,
        save_steps=cfg.training.sft.save_steps,
        save_total_limit=cfg.training.sft.save_total_limit,
        output_dir=f'experiments/{model_str}_sft',
        eval_strategy=cfg.training.sft.eval_strategy,
        eval_steps=cfg.training.sft.eval_steps,
        logging_dir=f'experiments/{model_str}_sft/logs',
        max_steps=cfg.training.sft.max_steps,
        report_to=cfg.logger.report_to,
        per_device_eval_batch_size=cfg.training.sft.batch_size_eval,
        weight_decay=cfg.training.sft.weight_decay,
        push_to_hub=cfg.training.sft.push_to_hub,
        do_eval=cfg.training.sft.do_eval,
    )

    peft_config = LoraConfig(
        r=cfg.training.sft_experiment.lora.r,
        lora_alpha=cfg.training.sft_experiment.lora.alpha,
        lora_dropout=cfg.training.sft_experiment.lora.dropout,
        # bias=cfg.training.sft_experiment.lora.bias, # getting NotImplementedError when set (?)
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=list(cfg.training.sft_experiment.lora.target_modules),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
    )

    ft_model = trainer.model
    # 1) Grab all names of parameters that belong to LoRA
    lora_param_names = [
        name for name, _ in ft_model.named_parameters()
        if "lora_" in name
    ]

    logging.info(f"Found {len(lora_param_names)} LoRA parameters, e.g.:")
    for n in lora_param_names[:10]:
        print("  ", n)
    print("...")

    # 2) Verify coverage of your target_modules
    #    Make sure each target module has at least one LoRA_A or LoRA_B
    for tm in peft_config.target_modules:
        hits = [n for n in lora_param_names if tm in n]
        if hits:
            logging.info(f"[OK]    {tm:15} → {len(hits)} adapter weights")
        else:
            logging.info(f"[MISSING] {tm:15} → NO LoRA weights found!")

    logging.info(f"EOS: {str(trainer.processing_class.eos_token_id)}")
    # 1) Raw sample
    sample = train_dataset[0]
    logging.info("Sample messages: %s", sample["messages"])

    # 2) One batch from the Trainer’s dataloader
    train_loader = trainer.get_train_dataloader()
    batch = next(iter(train_loader))
    logging.info("Batch keys: %s", list(batch.keys()))
    logging.info("input_ids[0]: %s", batch["input_ids"][0])
    logging.info("attention_mask[0]: %s", batch["attention_mask"][0])
    logging.info("labels[0]:    %s", batch["labels"][0])

    # ------------------------------- Training ------------------------------
    start_ts = time.time()
    trainer.train()
    runtime_min = (time.time() - start_ts) / 60
    logging.info("Training finished in %.1f min", runtime_min)

    # ------------------------------- Saving -------------------------------
    out_path = os.path.join(f'experiments/{model_str}_sft', "final_adapter")
    trainer.save_model(out_path)
    trainer.model.save_pretrained(
        f'adapters/sft/{cfg.training.sft_experiment.lora.r}-{cfg.training.sft_experiment.lora.alpha}-'
        f'{cfg.training.sft_experiment.lora.dropout}/{cfg.training.sft_dataset.name}/'
        f'{"-".join(cfg.training.sft_experiment.lora.target_modules)}'
    )
    logging.info("Adapter saved to %s", out_path)
    wandb.finish()

    return trainer.model


def lukas_dpo_old(cfg, model):
    quant_cfg = build_quant_config(
        cfg.training.quantization
    )
    logging.info("Using quantisation: %s", quant_cfg)

    # if SFT ran before, model is not None
    if model is None:
        # otherwise, if just running DPO
        # initialise model from scratch
        model = AutoModelForCausalLM.from_pretrained(
            cfg.training.model.model_name,
            # quantization doesn't work on Apple Metal
            quantization_config=quant_cfg if device != 'mps' else None,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.training.model.model_name, fast=False
    )

    if 'gemma' in cfg.training.model.name:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
    elif 'llama' in cfg.training.model.name:
        tokenizer.pad_token = tokenizer.eos_token

    # copy chat template & special tokens if missing
    if not getattr(tokenizer, "chat_template", None):
        try:
            toks_it = AutoTokenizer.from_pretrained(
                cfg.training.model.model_it_name,
                use_fast=False
            )
            if getattr(toks_it, "chat_template", None):
                tokenizer.chat_template = toks_it.chat_template
                logging.info("chat_template copied from -it model")
            extra = toks_it.special_tokens_map.get(
                "additional_special_tokens", []
            )
            new_tokens = [
                t for t in extra
                if t not in tokenizer.get_vocab()
            ]
            if new_tokens:
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": new_tokens}
                )
                model.resize_token_embeddings(len(tokenizer))
                logging.info("Added %d extra special tokens", len(new_tokens))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to copy -it tokenizer: %s", exc)
    else:
        print("Tokenizer already has a chat-template.")

    # ------------------ Dataset ------------------
    print('Loading DPO dataset')
    raw_dataset = load_dataset(
        cfg.training.dpo_dataset.huggingface_dataset_id,
        split=f'{cfg.training.dpo_dataset.split}[:10%]'
    )
    print('Dataset loaded')
    # 1) HH string  →  chosen/rejected lists
    msg_dataset = raw_dataset.map(
        preprocess_to_messages,
        remove_columns=raw_dataset.column_names
    )

    # 2) drop role‑alternation violations (code from previous answer)
    msg_dataset = msg_dataset.filter(
        lambda ex: not violates_alternation(ex["chosen"])
        and not violates_alternation(ex["rejected"])
    )

    # 3) ensure at least two turns and assistant‑ending
    msg_dataset = msg_dataset.filter(
        lambda ex: is_valid_dpo_pair(ex["chosen"])
        and is_valid_dpo_pair(ex["rejected"])
    )

    logging.info("Dataset after all filters: %d rows", len(msg_dataset))

    # adds 'prompt' field expected by DPO
    msg_dataset = msg_dataset.map(extract_prompt)
    # TODO: again, why are we manually splitting if we can use the default split from huggingface?
    msg_dataset = msg_dataset.train_test_split(test_size=0.1, seed=cfg.seed)
    train_ds, eval_ds = msg_dataset["train"], msg_dataset["test"]
    logging.info(train_ds)

    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens",
            [tokenizer.eos_token]
        )[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )
    model.generation_config.eos_token_id = [
        model.generation_config.eos_token_id,
        tokenizer.convert_tokens_to_ids(eot_token),
    ]
    logging.info("EOT token set to %s", eot_token)

    # ------------------ LoRA ------------------
    topk_k = cfg.training.dpo_experiment.lora.k
    peft_config = LoraConfig(
        r=cfg.training.dpo_experiment.lora.r,
        lora_alpha=cfg.training.dpo_experiment.lora.alpha,
        lora_dropout=cfg.training.dpo_experiment.lora.dropout,
        # getting NotImplementedError when bias set else than 'none' (?)
        bias=cfg.training.dpo_experiment.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=list(
            cfg.training.dpo_experiment.lora.target_modules
        ),
    )

    # Apply standard LoRA to model
    model = get_peft_model(model, peft_config)

    if cfg.training.dpo_experiment.lora.top_k_experiment:
        # ── Inject Top-k masking ------------------------------------------------
        replaced = 0
        for name, module in model.named_modules():
            # print(isinstance(module, lora.Linear), module)
            print(type(module), module)
            if isinstance(module, peft.tuners.lora.layer.Linear) and hasattr(module, "lora_dropout"):
                parent = model.get_submodule(".".join(name.split(".")[:-1]))
                setattr(
                    parent, name.split(".")[-1],
                    TopKLoRALinear(
                        module, r=peft_config.r,
                        alpha=peft_config.lora_alpha,
                        k=topk_k
                    )
                )
                replaced += 1
        logging.info("TopKLoRALinear injected in %d layers", replaced)
        assert False

    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    dpo_cfg = DPOConfig(
        max_prompt_length=cfg.training.dpo.max_prompt_length,
        max_completion_length=cfg.training.dpo.max_completion_length,
        beta=cfg.training.dpo.beta,
        loss_type=cfg.training.dpo.loss_type,
        num_train_epochs=cfg.training.dpo.num_train_epochs,
        max_steps=cfg.training.dpo.max_steps,
        per_device_train_batch_size=cfg.training.dpo.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.dpo.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.dpo.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.dpo.gradient_checkpointing,
        optim=cfg.training.dpo.optim,
        learning_rate=cfg.training.dpo.learning_rate,
        warmup_ratio=cfg.training.dpo.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler.type,
        bf16=cfg.training.dpo.bf16,
        fp16=cfg.training.dpo.fp16,
        max_grad_norm=cfg.training.dpo.max_grad_norm,
        logging_steps=cfg.logger.logging_steps,
        save_strategy=cfg.training.dpo.save_strategy,
        save_steps=cfg.training.dpo.save_steps,
        save_total_limit=cfg.training.dpo.save_total_limit,
        # eval_strategy=cfg.training.dpo.eval_strategy,
        # eval_steps=cfg.training.dpo.eval_steps,
        report_to=cfg.logger.report_to,
        output_dir=f'experiments/{model_str}_dpo',
        logging_dir=f'experiments/{model_str}_dpo/logs',
        do_eval=False
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,       # frozen copy auto‑created
        args=dpo_cfg,
        peft_config=None,           # already applied
        train_dataset=train_ds,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    # TODO: is this necessary?
    # for name, module in model.named_modules():
    #     if "11" in name and isinstance(module, torch.nn.Linear):
    #         logging.info(name)

    # ------------------ Sanity check ------------------
    # SAFE: log the first chosen conversation
    # for i in range(10):
    #     sample = trainer.train_dataset[i]

    #     # Decode each field
    #     prompt_text = tokenizer.decode(
    #         sample['prompt_input_ids'],
    #         skip_special_tokens=True
    #     )
    #     chosen_text = tokenizer.decode(
    #         sample['chosen_input_ids'],
    #         skip_special_tokens=True
    #     )
    #     rejected_text = tokenizer.decode(
    #         sample['rejected_input_ids'],
    #         skip_special_tokens=True
    #     )

    #     logging.info("Prompt:")
    #     logging.info(prompt_text)
    #     logging.info("\nChosen:")
    #     logging.info(chosen_text)
    #     logging.info("\nRejected:")
    #     logging.info(rejected_text)

    # ------------------ Training ------------------
    start = time.time()
    trainer.train()
    logging.info("Training finished in %.1f min", (time.time() - start) / 60)

    # ------------------ Saving ------------------
    out_path = os.path.join(f'experiments/{model_str}_dpo', "final_adapter")
    trainer.model.to('cpu')
    trainer.save_model(out_path)
    trainer.model.save_pretrained(
        f'adapters/dpo/{cfg.training.dpo_experiment.lora.r}-{cfg.training.dpo_experiment.lora.alpha}-'
        f'{cfg.training.dpo_experiment.lora.dropout}/{cfg.training.dpo_dataset.name}/'
        f'{"-".join(cfg.training.dpo_experiment.lora.target_modules)}'
    )
    logging.info("Adapter saved to %s", out_path)
    wandb.finish()

    return trainer.model


def lukas_dpo(cfg, model):
    quant_cfg = build_quant_config(
        cfg.training.quantization
    )
    logging.info("Using quantisation: %s", quant_cfg)

    # if SFT ran before, model is not None
    if model is None:
        # otherwise, if just running DPO
        # initialise model from scratch
        model = AutoModelForCausalLM.from_pretrained(
            cfg.training.model.model_name,
            # quantization doesn't work on Apple Metal
            # quantization_config=quant_cfg if device != 'mps' else None,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.training.model.model_name, fast=False
    )

    if 'gemma' in cfg.training.model.name:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
    elif 'llama' in cfg.training.model.name:
        tokenizer.pad_token = tokenizer.eos_token

    # copy chat template & special tokens if missing
    if not getattr(tokenizer, "chat_template", None):
        try:
            toks_it = AutoTokenizer.from_pretrained(
                cfg.training.model.model_it_name,
                use_fast=False
            )
            if getattr(toks_it, "chat_template", None):
                tokenizer.chat_template = toks_it.chat_template
                logging.info("chat_template copied from -it model")
            extra = toks_it.special_tokens_map.get(
                "additional_special_tokens", []
            )
            new_tokens = [
                t for t in extra
                if t not in tokenizer.get_vocab()
            ]
            if new_tokens:
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": new_tokens}
                )
                model.resize_token_embeddings(len(tokenizer))
                logging.info("Added %d extra special tokens", len(new_tokens))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to copy -it tokenizer: %s", exc)
    else:
        print("Tokenizer already has a chat-template.")

    print('Loading DPO dataset')
    raw_dataset = load_dataset(
        cfg.training.dpo_dataset.huggingface_dataset_id,
        split=f'{cfg.training.dpo_dataset.split}[:10%]'
    )
    print('Dataset loaded')
    # 1) HH string  →  chosen/rejected lists
    msg_dataset = raw_dataset.map(
        hh_rlhf_preprocess_to_messages,
        remove_columns=raw_dataset.column_names
    )

    # 2) drop role‑alternation violations (code from previous answer)
    msg_dataset = msg_dataset.filter(
        lambda ex: not violates_alternation(ex["chosen"])
        and not violates_alternation(ex["rejected"])
    )

    # 3) ensure at least two turns and assistant‑ending
    msg_dataset = msg_dataset.filter(
        lambda ex: is_valid_dpo_pair(ex["chosen"])
        and is_valid_dpo_pair(ex["rejected"])
    )

    logging.info("Dataset after all filters: %d rows", len(msg_dataset))

    # adds 'prompt' field expected by DPO
    msg_dataset = msg_dataset.map(extract_prompt)
    # TODO: again, why are we manually splitting if we can use the default split from huggingface?
    msg_dataset = msg_dataset.train_test_split(
        test_size=0.1, seed=cfg.seed
    )
    train_ds, eval_ds = msg_dataset["train"], msg_dataset["test"]
    logging.info(
        "Dataset after filters: %d rows",
        len(train_ds) + len(eval_ds)
    )
    logging.info(train_ds)

    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens",
            [tokenizer.eos_token]
        )[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )
    model.generation_config.eos_token_id = [
        model.generation_config.eos_token_id,
        tokenizer.convert_tokens_to_ids(eot_token),
    ]
    logging.info("EOT token set to %s", eot_token)

    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)

    # Model & tokenizer
    ref_model = merge_lora_adapter(
        cfg.training.model.model_name,
        cfg.training.adapter.checkpoint_dir,
        f'experiments/merged/{cfg.training.model.model_name}_sft',
        save_merged_model=True
    )

    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     cfg["model_name"], quantization_config=quant_cfg, device_map="auto"
    # )

    # LoRA config + record k
    lcfg = cfg.training.dpo_experiment.lora
    if lcfg.top_k_experiment:
        topk_k = lcfg.k
    else:
        topk_k = lcfg.r

    peft_cfg = LoraConfig(
        r=lcfg.r,
        lora_alpha=lcfg.alpha,
        lora_dropout=lcfg.dropout,
        bias=lcfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(lcfg.target_modules),
    )
    peft_cfg.k = topk_k  # record Top-k in adapter_config.json

    # Apply LoRA
    model = get_peft_model(model, peft_cfg)
    print(model)

    # Inject Top-k wrappers
    replaced = 0
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

    # DPO training args
    dargs = cfg.training.dpo
    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    dpo_cfg = DPOConfig(
        max_prompt_length=dargs.max_prompt_length,
        max_completion_length=dargs.max_completion_length,
        max_steps=dargs.max_steps,
        beta=dargs.beta,
        loss_type=dargs.loss_type,
        num_train_epochs=dargs.num_train_epochs,
        per_device_train_batch_size=dargs.per_device_train_batch_size,
        per_device_eval_batch_size=dargs.per_device_eval_batch_size,
        gradient_accumulation_steps=dargs.gradient_accumulation_steps,
        gradient_checkpointing=dargs.gradient_checkpointing,
        optim=dargs.optim,
        learning_rate=dargs.learning_rate,
        warmup_ratio=dargs.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler.type,
        bf16=dargs.bf16,
        fp16=dargs.fp16,
        max_grad_norm=dargs.max_grad_norm,
        logging_steps=cfg.logger.logging_steps,
        save_strategy=dargs.save_strategy,
        save_steps=dargs.save_steps,
        save_total_limit=dargs.save_total_limit,
        # eval_strategy=dargs.eval_strategy,
        # eval_steps=dargs.eval_steps,
        report_to=cfg.logger.report_to,
        output_dir=f'experiments/{model_str}_dpo',
        logging_dir=f'experiments/{model_str}_dpo/logs',
        do_eval=False,
    )

    # from pprint import pprint
    # pprint(dpo_cfg)
    # Trainer setup
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        peft_config=None,
        train_dataset=train_ds,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    # Train
    t0 = time.time()
    trainer.train()
    logging.info("Training finished in %.1f min", (time.time()-t0)/60)

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
    out_path = os.path.join(f'experiments/{model_str}_dpo', "final_adapter")
    trainer.save_model(out_path)
    logging.info("Adapter saved to %s", out_path)
    wandb.finish()

    return trainer.model

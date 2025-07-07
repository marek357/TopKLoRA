from trl import (
    SFTTrainer,
    SFTConfig,
    DPOConfig,
    DPOTrainer,
    setup_chat_format,
    extract_prompt
)
from itertools import islice
from datasets import IterableDataset
from datasets import Dataset
import gc
from peft import prepare_model_for_kbit_training
import wandb
import torch
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import peft
import time
import os
from src.models import TopKLoRALinear, MemoryClearCallback, CustomDPOTrainer
from src.utils import build_quant_config, get_conversational_dataset, hh_rlhf_preprocess_to_messages, is_valid_dpo_pair, merge_lora_adapter, preprocess_to_messages, violates_alternation
from peft import PeftModelForCausalLM, PeftModel
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


    if 'gemma' in cfg.training.model.name:
        tokenizer.padding_side = 'right'
        model = AutoModelForCausalLM.from_pretrained(
            cfg.training.model.model_name,
            attn_implementation='eager',
            # quantization doesn't work on Apple Metal
            quantization_config=quant_cfg if device != 'mps' else None,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.training.model.model_name,
            # quantization doesn't work on Apple Metal
            quantization_config=quant_cfg if device != 'mps' else None,
            device_map="auto",
            trust_remote_code=True
        )


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

    # dataset = load_dataset(
    #     cfg.training.sft_dataset.huggingface_dataset_id,
    #     split=cfg.training.sft_dataset.split
    # )

    # message_dataset = dataset.map(
    #     preprocess_to_messages,
    #     remove_columns=dataset.column_names,
    # )

    # # TODO: why train test split if we set split in load_dataset?
    # message_dataset = message_dataset.train_test_split(test_size=0.1)
    # train_dataset, val_dataset = message_dataset["train"], message_dataset["test"]

    def preprocessed_stream():
        stream = load_dataset(
            cfg.training.sft_dataset.huggingface_dataset_id,
            split=cfg.training.sft_dataset.split,
            streaming=True
        )
        for ex in stream:
            msg = preprocess_to_messages(ex)
            yield msg

    def train_gen():
        for idx, ex in enumerate(preprocessed_stream()):
            if idx % 10 != 0:
                yield ex

    def eval_gen():
        for idx, ex in enumerate(preprocessed_stream()):
            if idx % 10 == 0:
                yield ex
    from datasets import IterableDataset
    # TODO: again, why are we manually splitting if we can use the default split from huggingface?
    train_dataset = IterableDataset.from_generator(train_gen)
    eval_dataset  = IterableDataset.from_generator(eval_gen)



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
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[MemoryClearCallback()],
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
    # sample = train_dataset[0]
    # logging.info("Sample messages: %s", sample["messages"])

    # 2) One batch from the Trainer’s dataloader
    # train_loader = trainer.get_train_dataloader()
    # batch = next(iter(train_loader))
    # logging.info("Batch keys: %s", list(batch.keys()))
    # logging.info("input_ids[0]: %s", batch["input_ids"][0])
    # logging.info("attention_mask[0]: %s", batch["attention_mask"][0])
    # logging.info("labels[0]:    %s", batch["labels"][0])

    if trainer.optimizer is not None:
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

    # ------------------------------- Training ------------------------------
    start_ts = time.time()
    trainer.train()
    runtime_min = (time.time() - start_ts) / 60
    logging.info("Training finished in %.1f min", runtime_min)

    # ------------------------------- Saving -------------------------------
    out_path = os.path.join(f'experiments/{model_str}_sft', "final_adapter")
    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)
    trainer.model.save_pretrained(
        f'adapters/sft/{cfg.training.sft_experiment.lora.r}-{cfg.training.sft_experiment.lora.alpha}-'
        f'{cfg.training.sft_experiment.lora.dropout}/{cfg.training.sft_dataset.name}/'
        f'{"-".join(cfg.training.sft_experiment.lora.target_modules)}'
    )
    tokenizer.save_pretrained(
        f'adapters/sft/{cfg.training.sft_experiment.lora.r}-{cfg.training.sft_experiment.lora.alpha}-'
        f'{cfg.training.sft_experiment.lora.dropout}/{cfg.training.sft_dataset.name}/'
        f'{"-".join(cfg.training.sft_experiment.lora.target_modules)}'
    )
    logging.info("Adapter saved to %s", out_path)
    wandb.finish()

    return trainer.model



def lukas_dpo(cfg, model):
    quant_cfg = build_quant_config(
        cfg.training.quantization
    )
    # logging.info("Using quantisation: %s", quant_cfg)

    # if SFT ran before, model is not None
    if model is None:
        # otherwise, if just running DPO
        # initialise model from scratch
        model = AutoModelForCausalLM.from_pretrained(
            cfg.training.model.model_name,
            device_map="auto",
            # quantization doesn't work on Apple Metal
            quantization_config=quant_cfg if device != 'mps' else None,
        )


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
    ref_model = merge_lora_adapter(
        cfg.training.model.model_name,
        cfg.training.adapter.checkpoint_dir,
        quant_cfg,
        f'experiments/merged/{cfg.training.model.model_name}_sft',
        save_merged_model=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )#.to('cpu')
    ref_model.generation_config.eos_token_id = [
        ref_model.generation_config.eos_token_id,
        tokenizer.convert_tokens_to_ids(eot_token),
    ]
    for param in ref_model.parameters():
        param.requires_grad = False

    ref_model.eval()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_cfg)
    model.config.use_cache = False
    print(model)
    print('Loading DPO dataset')

    def preprocessed_stream():
        stream = load_dataset(
            cfg.training.dpo_dataset.huggingface_dataset_id,
            split=cfg.training.dpo_dataset.split,
            streaming=True
        )
        for ex in stream:
            msg = hh_rlhf_preprocess_to_messages(ex)
            # Skip if *either* chosen or rejected has a role-alternation violation
            if violates_alternation(msg["chosen"]) or violates_alternation(msg["rejected"]):
                continue

            # Skip if either side isn’t a valid DPO pair
            if not is_valid_dpo_pair(msg["chosen"]) or not is_valid_dpo_pair(msg["rejected"]):
                continue

            # Now it’s safe to extract and yield
            yield extract_prompt(msg)

    def train_gen():
        for idx, ex in enumerate(preprocessed_stream()):
            if idx % 10 != 0:
                yield ex

    def eval_gen():
        for idx, ex in enumerate(preprocessed_stream()):
            if idx % 10 == 0:
                yield ex
    from datasets import IterableDataset
    # TODO: again, why are we manually splitting if we can use the default split from huggingface?
    train_ds = IterableDataset.from_generator(train_gen)
    eval_ds  = IterableDataset.from_generator(eval_gen)

    logging.info("EOT token set to %s", eot_token)

    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)

    # Model & tokenizer


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
    # model.eval()

    # Inject Top-k wrappers
    # replaced = 0
    # for name, module in model.named_modules():
    #     if getattr(module, "lora_A", None) is None:
    #         continue
    #     parent = model.get_submodule(".".join(name.split(".")[:-1]))
    #     attr = name.split(".")[-1]
    #     setattr(parent, attr, TopKLoRALinear(
    #         module,
    #         layer_name=name,
    #         r=module.r,
    #         alpha=module.lora_alpha,
    #         k=topk_k,
    #     ))
    #     replaced += 1
    # logging.info("TopKLoRALinear injected in %d layers", replaced)

    # print(model)


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
        padding_value=tokenizer.pad_token_id,
        eval_strategy=dargs.eval_strategy,
        eval_steps=dargs.eval_steps,
        report_to=cfg.logger.report_to,
        output_dir=f'experiments/{model_str}_dpo',
        logging_dir=f'experiments/{model_str}_dpo/logs',
        do_eval=dargs.do_eval,
    )

    def collate_fn(batch):
        """
        Pads every example in `batch` to exactly `max_seq_len` tokens
        for prompt, chosen, and rejected separately.
        """
        B = len(batch)
        pad_id = tokenizer.pad_token_id
        max_seq_len = cfg.training.dpo.max_prompt_length + cfg.training.dpo.max_completion_length

        # allocate fixed‐size tensors
        # prompts
        prompt_ids   = torch.full((B, max_seq_len), pad_id, dtype=torch.long)
        prompt_mask  = torch.zeros((B, max_seq_len), dtype=torch.long)
        # chosen completions
        chosen_ids   = torch.full((B, max_seq_len), pad_id, dtype=torch.long)
        chosen_mask  = torch.zeros((B, max_seq_len), dtype=torch.long)
        # rejected completions
        rejected_ids  = torch.full((B, max_seq_len), pad_id, dtype=torch.long)
        rejected_mask = torch.zeros((B, max_seq_len), dtype=torch.long)

        for i, ex in enumerate(batch):
            p = ex["prompt_input_ids"]
            c = ex["chosen_input_ids"]
            r = ex["rejected_input_ids"]
            # copy and mask
            prompt_ids[i, : len(p)]   = torch.tensor(p, dtype=torch.long)
            prompt_mask[i, : len(p)]  = 1
            chosen_ids[i, : len(c)]   = torch.tensor(c, dtype=torch.long)
            chosen_mask[i, : len(c)]  = 1
            rejected_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            rejected_mask[i, : len(r)]= 1

        return {
            "prompt_input_ids":        prompt_ids,
            "prompt_attention_mask":   prompt_mask,
            "chosen_input_ids":        chosen_ids,
            "chosen_attention_mask":   chosen_mask,
            "rejected_input_ids":      rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }

    # Trainer setup
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        peft_config=None,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collate_fn,
        callbacks=[MemoryClearCallback()],
    )

    if trainer.optimizer is not None:
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

    # original_compute_ref = trainer.compute_ref_log_probs
    # def cpu_ref_log_probs(batch):
    #     batch = {
    #         k: (v.to(trainer.ref_model.device) if isinstance(v, torch.Tensor) else v)
    #         for k, v in batch.items()
    #     }
    #     return original_compute_ref(batch)
    # trainer.compute_ref_log_probs = cpu_ref_log_probs


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

def sanity_check(cfg, model, quant_cfg):

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig
    )
    import os
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from trl import DPOTrainer, DPOConfig

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

    # 4) Attach fresh LoRA everywhere
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(policy_model, lora_config)
    model.print_trainable_parameters()


    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # 6) DPOConfig (lower lr, low β, linear decay)
    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # still eff_bs=16
        learning_rate=5e-6,             # x2–3
        beta=0.01,                      # stronger KL
        lr_scheduler_type="linear",
        warmup_steps=10,                # actual warmup
        logging_steps=10,
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
        processing_class=tokenizer,
    )

    # 8) Train
    trainer.train()

# def sanity_check(cfg, model, quant_cfg):
#     import torch
#     from datasets import load_dataset
#     from transformers import (
#         AutoModelForCausalLM,
#         AutoTokenizer,
#         BitsAndBytesConfig
#     )
#     from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
#     from trl import DPOTrainer, DPOConfig
#     import wandb
#     from typing import Dict, List
#     import os


#     OUTPUT_DIR = "./sanity_check/gemma-2-2b-dpo-lora"
#     tokenizer = AutoTokenizer.from_pretrained(cfg.training.adapter.checkpoint_dir)

#     print("Loading model...")
#     # 2. Load the base model
#     base_model = AutoModelForCausalLM.from_pretrained(
#         cfg.training.model.model_name,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True, 
#         quantization_config=quant_cfg,
#     )

#     # 3. Load the LoRA adapter on top of the base model
#     model_with_lora = PeftModel.from_pretrained(
#         base_model,
#         cfg.training.adapter.checkpoint_dir, 
#         use_safetensors=True
#     )

#     # 4. Merge LoRA weights into the base model
#     merged_model = model_with_lora.merge_and_unload()
#     merged_output_dir = f'experiments/merged/{cfg.training.model.model_name}_sft'
#     # 5. Save the merged model and tokenizer
#     assert merged_output_dir is not None, 'Cannot save merged model without providing output dir'
#     merged_model.save_pretrained(merged_output_dir)
#     tokenizer.save_pretrained(merged_output_dir)

#     del merged_model, base_model

#     print(f"Merged model saved to: {merged_output_dir}")

#     # saving and loading the same model removes peft-related attributes
#     model = AutoModelForCausalLM.from_pretrained(
#         merged_output_dir
#     )

#     # Explicitly set EOT/EOS token for Gemma
#     # Gemma uses <eos> token, we need to ensure it's properly set
#     if tokenizer.eos_token is None:
#         tokenizer.add_special_tokens({'eos_token': '<eos>'})
        
#     # Set pad token to eos token
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.padding_side = "left"

#     print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
#     print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

#     # Resize token embeddings if necessary
#     model.resize_token_embeddings(len(tokenizer))

#     # Ensure model config has the correct eos_token_id
#     model.config.eos_token_id = tokenizer.eos_token_id
#     model.config.pad_token_id = tokenizer.pad_token_id

#     # Prepare model for k-bit training
#     model = prepare_model_for_kbit_training(model)

#     # LoRA configuration - targeting only layer 11
#     lora_config = LoraConfig(
#         r=16,  # Rank
#         lora_alpha=32,  # Alpha parameter for LoRA scaling
#         target_modules=[
#             "q_proj",
#             "k_proj",
#             "v_proj",
#             "o_proj",
#             "mlp.gate_proj",
#             "mlp.up_proj",
#             "mlp.down_proj",
#         ],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         modules_to_save=None,  # You can add modules like 'embed_tokens' or 'lm_head' if needed
#     )

#     # Add LoRA adapters to the model
#     print("Adding LoRA adapters...")
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()

#     # Print which specific modules are being trained
#     print("\nModules with LoRA adapters:")
#     for name, module in model.named_modules():
#         if "lora" in name:
#             print(f"  - {name}")

#     # Load and preprocess dataset
#     print("Loading dataset...")
#     dataset = load_dataset("Anthropic/hh-rlhf", split="train")


#     def format_for_dpo(example):
#         text = example["chosen"]
#         text2= example["rejected"]

#         # Extract prompt
#         if "Assistant: " not in text:
#             return None   # will be filtered out
#         before, after = text.split("Assistant: ", 1)
#         prompt = before.replace("Human:", "").strip()

#         # Choose only the *first* assistant reply
#         chosen = after.split("Human:", 1)[0].strip()

#         # Same for rejected
#         if "Assistant: " not in text2:
#             return None
#         _, rej_after = text2.split("Assistant: ", 1)
#         rejected = rej_after.split("Human:", 1)[0].strip()

#         return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

#     # Un-batched map avoids length mismatches:
#     ds1 = dataset.map(format_for_dpo, batched=False, remove_columns=dataset.column_names)
#     # Drop any Nones
#     formatted_dataset = ds1.filter(lambda ex: ex["prompt"] and ex["chosen"] and ex["rejected"])


#     # Split dataset into train and eval
#     train_test_split = formatted_dataset.train_test_split(test_size=0.1, seed=42)
#     train_dataset = train_test_split["train"]
#     eval_dataset = train_test_split["test"].take(200)

#     print(f"Training samples: {len(train_dataset)}")
#     print(f"Evaluation samples: {len(eval_dataset)}")

#     # Training arguments
#     training_args = DPOConfig(
#         output_dir=OUTPUT_DIR,
#         num_train_epochs=1,
#         per_device_train_batch_size=2,
#         per_device_eval_batch_size=4,
#         gradient_accumulation_steps=8,
#         gradient_checkpointing=True,
#         learning_rate=2e-6,
#         lr_scheduler_type="linear",
#         warmup_steps=100,
#         save_steps=500,
#         eval_strategy="steps",
#         eval_steps=100,
#         do_eval=True,
#         report_to="wandb",  # Change to "none" if not using wandb
#         run_name="gemma-2-2b-dpo-lora",
#         bf16=True,
#         push_to_hub=False,
#         remove_unused_columns=False,
#         beta=0.005,  # DPO beta parameter - controls KL penalty
#         max_length=512,
#         max_completion_length=256,
#         max_prompt_length=256,
#         logging_steps=1,
#     )

#     # DPO training arguments
#     dpo_trainer = DPOTrainer(
#         model=model,
#         ref_model=None,  # We don't need a separate reference model with LoRA
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         processing_class=tokenizer,
#     )

#     # Start training
#     print("Starting DPO training...")
#     dpo_trainer.train()

#     # Save the final model
#     print("Saving model...")
#     dpo_trainer.save_model(OUTPUT_DIR)
#     tokenizer.save_pretrained(OUTPUT_DIR)

#     # Merge LoRA weights with base model (optional)
#     print("Merging LoRA weights...")
#     merged_model = dpo_trainer.model.merge_and_unload()
#     merged_model.save_pretrained(f"{OUTPUT_DIR}-merged")
#     tokenizer.save_pretrained(f"{OUTPUT_DIR}-merged")

#     print("Training complete!")

#     # Example inference with the fine-tuned model
#     def generate_response(prompt: str, model, tokenizer, max_length: int = 256):
#         """Generate a response using the fine-tuned model."""
#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_length,
#                 temperature=0.7,
#                 do_sample=True,
#                 top_p=0.9,
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
        
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response

#     # Test the model
#     test_prompt = "Human: What are the benefits of regular exercise?\n\nAssistant:"
#     print(f"\nTest prompt: {test_prompt}")
#     response = generate_response(test_prompt, dpo_trainer.model, tokenizer)
#     print(f"Model response: {response}")

#     # Additional utility functions

#     def save_lora_only(model, output_dir: str):
#         """Save only the LoRA adapters."""
#         model.save_pretrained(output_dir)
#         print(f"LoRA adapters saved to {output_dir}")

#     def load_finetuned_model(base_model_name: str, lora_weights_path: str):
#         """Load the base model with LoRA weights."""
#         from peft import PeftModel
        
#         # Load base model
#         base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#         )
        
#         # Load LoRA weights
#         model = PeftModel.from_pretrained(base_model, lora_weights_path)
#         return model

#     # Save just the LoRA adapters (much smaller file size)
#     # save_lora_only(dpo_trainer.model, f"{OUTPUT_DIR}-lora-only")

#     print("\nTraining script completed successfully!")
#     print(f"Models saved to:")
#     print(f"  - Full model with LoRA: {OUTPUT_DIR}")
#     print(f"  - Merged model: {OUTPUT_DIR}-merged")
#     print(f"  - LoRA adapters only: {OUTPUT_DIR}-lora-only")

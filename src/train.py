import wandb
import torch
from trl import (
    SFTTrainer,
    SFTConfig,
    setup_chat_format,
    DPOConfig,
    DPOTrainer
)
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from src.utils import get_conversational_dataset
from peft import PeftModelForCausalLM
import numpy as np

device = 'cuda' if torch.cuda.is_available() \
    else 'mps' if torch.mps.is_available() else 'cpu'


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}


def run_sft(cfg, peft_config, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.huggingface_model_id
    ).to(device)

    try:
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=cfg.model.huggingface_model_id,
            # name of the adapter is the dataset name
            adapter_name=cfg.dataset_sft.name,
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
        cfg.dataset_sft.huggingface_dataset_id, tokenizer
    )
    model_str = f'{cfg.model.name}_{cfg.model.version}_{cfg.model.size}'
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
        f'adapters/sft/{cfg.experiment.lora.r}-{cfg.experiment.lora.alpha}-'
        f'{cfg.experiment.lora.dropout}/{cfg.dataset_sft.name}/'
        f'{"-".join(cfg.experiment.lora.target_modules)}'
    )

    return trainer.model


def run_dpo(cfg, peft_config, tokenizer, model):
    train_dataset = load_dataset(
        cfg.dataset_dpo.huggingface_dataset_id,
        split="train"
    )

    model_str = f'{cfg.model.name}_{cfg.model.version}_{cfg.model.size}'
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
        f'adapters/dpo/{cfg.experiment.lora.r}-{cfg.experiment.lora.alpha}-'
        f'{cfg.experiment.lora.dropout}/{cfg.dataset_sft.name}/'
        f'{"-".join(cfg.experiment.lora.target_modules)}'
    )

import wandb
import torch
import hydra
import random
import numpy as np
from trl import setup_chat_format
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.train import run_sft, run_dpo
from peft import LoraConfig, TaskType
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logger.wandb_mode  # NOTE: disabled by default
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.experiment.lora.r,
        target_modules=list(cfg.experiment.lora.target_modules),
        lora_alpha=cfg.experiment.lora.alpha,
        lora_dropout=cfg.experiment.lora.dropout
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.huggingface_model_id)

    if cfg.training.sft.enabled:
        model = run_sft(cfg, peft_config, tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.huggingface_model_id
        )
        try:
            model, tokenizer = setup_chat_format(
                model=model, tokenizer=tokenizer
            )
        except ValueError:
            pass
        model.load_adapter(
            f'adapters/sft/{cfg.experiment.lora.r}-{cfg.experiment.lora.alpha}-'
            f'{cfg.experiment.lora.dropout}/{cfg.dataset_sft.name}/'
            f'{"-".join(cfg.experiment.lora.target_modules)}'
        )

    if cfg.training.dpo.enabled:
        run_dpo(cfg, peft_config, tokenizer, model)


if __name__ == '__main__':
    main()

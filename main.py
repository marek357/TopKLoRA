from dotenv import load_dotenv
import wandb
import torch
import hydra
import random
import numpy as np
from transformers import set_seed
from src.train import lukas_dpo, lukas_sft, run_sft, run_dpo
from omegaconf import DictConfig, OmegaConf
import logging

from src.utils import merge_lora_adapter


@hydra.main(
    version_base=None,
    config_path="config/train_config",
    config_name="default"
)
def main(cfg: DictConfig):
    load_dotenv()

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.get("seed", 42))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y‑%m‑%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)

    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logger.wandb_mode  # NOTE: disabled by default
    )

    model = None

    if cfg.training.sft.enabled:
        model = lukas_sft(cfg)

    if cfg.training.dpo.enabled:
        print('Loading and merging LoRA adapter from checkpoint')
        model = merge_lora_adapter(
            cfg.training.model.model_name,
            cfg.training.adapter.checkpoint_dir,
            f'experiments/merged/{cfg.training.model.model_name}_sft',
            save_merged_model=True
        )

        model = lukas_dpo(cfg, model)

    if cfg.training.dump_trained_model:
        model.save_pretrained(cfg.training.dump_path)


if __name__ == '__main__':
    main()

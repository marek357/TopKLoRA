from dotenv import load_dotenv
import wandb
import torch
import hydra
import random
import numpy as np
from transformers import set_seed, AutoModelForCausalLM
from src.sft import run_sft
from src.dpo import run_dpo
from src.utils import build_quant_config
from omegaconf import DictConfig, OmegaConf
import logging
import os
from src.utils import merge_lora_adapter
import socket


@hydra.main(
    version_base=None,
    config_path="config/train_config",
    config_name="default"
)
def main(cfg: DictConfig):
    load_dotenv()
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(socket.gethostname())
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.get("seed", 42))
    # wandb.init(
    #     project=cfg.logger.project,
    #     entity=cfg.logger.entity,
    #     name=cfg.experiment_name,
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     mode=cfg.logger.wandb_mode  # NOTE: disabled by default
    # )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y‑%m‑%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)

    model = None

    if cfg.training.sft.enabled:
        model = run_sft(cfg)

    if cfg.training.dpo.enabled:
        print('Loading and merging LoRA adapter from checkpoint')
        quant_cfg = build_quant_config(
            cfg.training.quantization
        )

        run_dpo(cfg, quant_cfg)
        # model = lukas_dpo(cfg, model)

    # if cfg.training.dump_trained_model:
    #     model.save_pretrained(cfg.training.dump_path)


if __name__ == '__main__':
    main()

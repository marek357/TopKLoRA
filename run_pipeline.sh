#!/bin/bash

set -e

error_handler() {
  echo "Error: An unexpected error occurred on line $LINENO." >&2
  exit 1
}

trap error_handler ERR


MODEL_NAME="gemma_2b"

# /home/cvenhoff/miniconda/envs/lora_env/bin/python3 /home/cvenhoff/lora_interp/main.py training=sft training/model="$MODEL_NAME"

# echo "Step 1. complete: SFT model ($MODEL_NAME) fine-tuned!"

# /home/cvenhoff/miniconda/envs/lora_env/bin/python3 /home/cvenhoff/lora_interp/eval.py model=sft_$MODEL_NAME 

# echo "Step 2. complete: SFT model eval done!"

echo "DPO training start..."

/home/cvenhoff/miniconda/envs/lora_env/bin/python3 /home/cvenhoff/lora_interp/main.py training=dpo training/model=$MODEL_NAME training/adapter=$MODEL_NAME #logger=wandb_disabled

echo "Step 3. complete: DPO model ($MODEL_NAME) fine-tuned!"

/home/cvenhoff/miniconda/envs/lora_env/bin/python3 /home/cvenhoff/lora_interp/eval.py model=dpo_$MODEL_NAME

echo "Step 4. complete: DPO model eval done!"
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
from src.train import TopKLoRALinearSTE, TopKProgressCallback, DeadLatentsLoggerCallback
from src.utils import build_quant_config, get_conversational_dataset, hh_rlhf_preprocess_to_messages, is_valid_dpo_pair, merge_lora_adapter, preprocess_to_messages, violates_alternation
from peft import PeftModelForCausalLM, PeftModel
import numpy as np
import logging
import torch.nn.functional as F


local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


class EnhancedSFTTrainer(SFTTrainer):
    """
    Enhanced SFT Trainer with TopK monitoring and optional regularization.
    Critical fix: correctly handle `return_outputs` in compute_loss (HF eval expects (loss, outputs)).
    """

    def __init__(self, *args, **kwargs):
        self.reg_cfg = kwargs.pop('reg_cfg', {
            "log_every": 50,
            "L_DECORR": 1e-4,
            "L_MASS": 1e-3,
            "L_ENTROPY": 0.0,
            "L_ORTHO_A": 1e-4,
            "L_ORTHO_B": 1e-4,
            "ORTHO_EVERY": 4,
            "schedule_decorr": True,
            "schedule_mass": True,
            "schedule_ent": True,
            "schedule_ortho": True
        })
        # "off" | "z_only" | "z_plus_ortho"
        self.reg_mode = kwargs.pop('reg_mode', "off")
        super().__init__(*args, **kwargs)

    # ---------- helpers ----------
    def _sched_scalar(self, t: float) -> float:
        """Schedule weight from 0→1 over training progress (cubic)."""
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return t ** 3

    def _active(self, L: float, scheduled_flag: bool, w_sched: float) -> bool:
        """Is a regularizer active under current schedule/weight?"""
        if L <= 0.0:
            return False
        if not scheduled_flag:
            return True
        return w_sched > 0.0

    def _clear_topk_caches(self, model):
        # Drop live caches to avoid cross-step graph retention
        from types import SimpleNamespace  # just to reduce overhead if absent
        for m in model.modules():
            # isinstance check is cheap; avoids hasattr chain when irrelevant
            if m.__class__.__name__ == "TopKLoRALinearSTE":
                if hasattr(m, "_z_live"):
                    m._z_live = None
                if hasattr(m, "_g_soft_live"):
                    m._g_soft_live = None

    # ---------- main ----------
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        HF expects:
          - when return_outputs=False → a scalar loss tensor
          - when return_outputs=True  → (loss, outputs)
        """
        # Get base loss/outputs from TRL/HF
        base = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if return_outputs:
            base_loss, base_outputs = base
        else:
            base_loss, base_outputs = base, None

        step = int(self.state.global_step or 0)
        max_steps = int(self.state.max_steps or 1)

        # --- TopK gate stats logging (lightweight, first matching layer) ---
        log_every = int(self.reg_cfg.get("log_every", 50))
        if log_every > 0 and (step % log_every == 0):
            for name, m in model.named_modules():
                if m.__class__.__name__ == "TopKLoRALinearSTE":
                    st = getattr(m, "get_gate_stats", lambda: {})()
                    if st:
                        self.log({
                            f"{name}.k": st.get("k", 0),
                            f"{name}.tau": st.get("tau", 0.0),
                            f"{name}.frac_active_vs_target": st.get("frac_active_vs_target", 0.0),
                        })
                        break

        # If regularization is off, just clear caches and return the base
        if self.reg_mode == "off":
            self._clear_topk_caches(model)
            return (base_loss, base_outputs) if return_outputs else base_loss

        # -------- Regularization path --------
        reg = base_loss.new_tensor(0.0)
        do_log = (log_every > 0) and (step % log_every == 0)
        acc = {
            "reg/decorr": 0.0,
            "reg/mass": 0.0,
            "reg/entropy": 0.0,
            "reg/ortho_A": 0.0,
            "reg/ortho_B": 0.0,
            "reg/sched_w": 0.0,
        }
        n_layers = 0

        # Pull config once
        L_DECORR = float(self.reg_cfg.get("L_DECORR", 0.0))
        L_MASS   = float(self.reg_cfg.get("L_MASS",   0.0))
        L_ENTROPY= float(self.reg_cfg.get("L_ENTROPY",0.0))
        L_ORTHO_A= float(self.reg_cfg.get("L_ORTHO_A",0.0))
        L_ORTHO_B= float(self.reg_cfg.get("L_ORTHO_B",0.0))
        ORTHO_EVERY = int(self.reg_cfg.get("ORTHO_EVERY", 0))

        try:
            for m in model.modules():
                if m.__class__.__name__ != "TopKLoRALinearSTE":
                    continue

                # Live tensors from forward pass
                z_live = getattr(m, "_z_live", None)
                if z_live is None:
                    continue

                # Progress & schedule weight
                try:
                    p_layer = float(getattr(m, "progress", 0.0))
                except Exception:
                    p_layer = step / max(1, max_steps)
                w_sched = self._sched_scalar(p_layer)

                need_decorr = self._active(L_DECORR, self.reg_cfg.get("schedule_decorr", True), w_sched)
                need_mass   = self._active(L_MASS,   self.reg_cfg.get("schedule_mass", True),   w_sched)
                need_ent    = self._active(L_ENTROPY,self.reg_cfg.get("schedule_ent", True),    w_sched)

                need_orthoA = (
                    self.reg_mode == "z_plus_ortho" and ORTHO_EVERY > 0 and (step % ORTHO_EVERY == 0) and
                    self._active(L_ORTHO_A, self.reg_cfg.get("schedule_ortho", True), w_sched)
                )
                need_orthoB = (
                    self.reg_mode == "z_plus_ortho" and ORTHO_EVERY > 0 and (step % ORTHO_EVERY == 0) and
                    self._active(L_ORTHO_B, self.reg_cfg.get("schedule_ortho", True), w_sched)
                )

                all_sched = (
                    self.reg_cfg.get("schedule_decorr", True) and
                    self.reg_cfg.get("schedule_mass", True) and
                    self.reg_cfg.get("schedule_ent", True) and
                    (self.reg_mode != "z_plus_ortho" or self.reg_cfg.get("schedule_ortho", True))
                )
                if all_sched and not (need_decorr or need_mass or need_ent or need_orthoA or need_orthoB):
                    if do_log:
                        acc["reg/sched_w"] += w_sched
                        n_layers += 1
                    continue

                # Prepare gates (reuse live soft gate when available)
                k_now = getattr(m, "_current_k")()
                tau   = getattr(m, "_tau")()
                g_soft_live = getattr(m, "_g_soft_live", None)

                if g_soft_live is None:
                    # Local import to avoid circulars (matches your original code pattern)
                    from src.train import _soft_topk_mass
                    g_soft = _soft_topk_mass(z_live, k_now, tau)
                else:
                    g_soft = g_soft_live

                # ---- Z-based regularizers ----
                if need_decorr:
                    Z = z_live.reshape(-1, z_live.size(-1)).float()
                    Z = Z - Z.mean(dim=0, keepdim=True)
                    C = (Z.T @ Z) / (Z.size(0) + 1e-6)
                    off = C - torch.diag(torch.diag(C))
                    r_decorr = (off ** 2).mean().to(base_loss.dtype)
                    if self.reg_cfg.get("schedule_decorr", True):
                        r_decorr = r_decorr * r_decorr.new_tensor(w_sched)
                    reg = reg + L_DECORR * r_decorr
                    if do_log:
                        acc["reg/decorr"] += float(r_decorr.detach().cpu())

                if need_mass or need_ent:
                    if need_mass:
                        r_mass = (g_soft.sum(dim=-1) - k_now).pow(2).mean()
                        if self.reg_cfg.get("schedule_mass", True):
                            r_mass = r_mass * r_mass.new_tensor(w_sched)
                        reg = reg + L_MASS * r_mass
                        if do_log:
                            acc["reg/mass"] += float(r_mass.detach().cpu())

                    if need_ent:
                        g_safe = g_soft.clamp_min(1e-8)
                        r_ent = -(g_safe * g_safe.log()).sum(dim=-1).mean()
                        if self.reg_cfg.get("schedule_ent", True):
                            r_ent = r_ent * r_ent.new_tensor(w_sched)
                        reg = reg + L_ENTROPY * r_ent
                        if do_log:
                            acc["reg/entropy"] += float(r_ent.detach().cpu())

                # ---- Orthogonality (only in z_plus_ortho) ----
                if need_orthoA:
                    Aw = m.A_module.weight
                    A_rows = F.normalize(Aw.float(), p=2, dim=1)
                    GA = A_rows @ A_rows.T
                    GA_off = GA - torch.diag(torch.diag(GA))
                    r_oa = (GA_off ** 2).mean().to(base_loss.dtype)
                    if self.reg_cfg.get("schedule_ortho", True):
                        r_oa = r_oa * r_oa.new_tensor(w_sched)
                    reg = reg + L_ORTHO_A * r_oa
                    if do_log:
                        acc["reg/ortho_A"] += float(r_oa.detach().cpu())

                if need_orthoB:
                    Bw = m.B_module.weight
                    B_cols = F.normalize(Bw.float(), p=2, dim=0)
                    GB = B_cols.T @ B_cols
                    GB_off = GB - torch.diag(torch.diag(GB))
                    r_ob = (GB_off ** 2).mean().to(base_loss.dtype)
                    if self.reg_cfg.get("schedule_ortho", True):
                        r_ob = r_ob * r_ob.new_tensor(w_sched)
                    reg = reg + L_ORTHO_B * r_ob
                    if do_log:
                        acc["reg/ortho_B"] += float(r_ob.detach().cpu())

                if do_log:
                    acc["reg/sched_w"] += w_sched
                    n_layers += 1

                # Extra gentle priors
                L1 = 1e-5
                reg = reg + L1 * z_live.abs().mean()
                usage = g_soft.mean(dim=(0, 1))  # [r]
                cov = ((usage - usage.mean()) ** 2).mean()
                reg = reg + 1e-4 * cov

        finally:
            # Always clear caches
            self._clear_topk_caches(model)

        total_loss = base_loss + reg

        # Log aggregate regs + one layer’s gate stats
        if do_log and n_layers > 0:
            for k in acc:
                acc[k] /= n_layers
            # attach some gate stats for one layer
            for name, m in model.named_modules():
                if m.__class__.__name__ == "TopKLoRALinearSTE":
                    st = getattr(m, "get_gate_stats", lambda: {})()
                    if st:
                        acc.update({
                            f"gate/{name}.active_latents": st.get("active_latents", 0),
                            f"gate/{name}.dead_latents":   st.get("dead_latents", 0),
                            f"gate/{name}.avg_usage":      st.get("avg_usage", 0.0),
                        })
                        break
            self.log(acc)

        return (total_loss, base_outputs) if return_outputs else total_loss

def enable_topk_lora_grads(model):
    # mark only A/B weights trainable; freeze everything else
    ab_ids = set()
    for mod in model.modules():
        if isinstance(mod, TopKLoRALinearSTE):
            if hasattr(mod.A_module, "weight"):
                mod.A_module.weight.requires_grad_(True)
                ab_ids.add(id(mod.A_module.weight))
            if getattr(mod.A_module, "bias", None) is not None:
                mod.A_module.bias.requires_grad_(True)
                ab_ids.add(id(mod.A_module.bias))
            if hasattr(mod.B_module, "weight"):
                mod.B_module.weight.requires_grad_(True)
                ab_ids.add(id(mod.B_module.weight))
            if getattr(mod.B_module, "bias", None) is not None:
                mod.B_module.bias.requires_grad_(True)
                ab_ids.add(id(mod.B_module.bias))

    # freeze everything not in A/B
    for p in model.parameters():
        if id(p) not in ab_ids:
            p.requires_grad_(False)


def count_trainables(model, label=""):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    T = sum(p.numel() for p in model.parameters())
    print(f"[raw] trainable params {label}: {t} / {T}")
    return t


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

def count_params(m):
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    print(f"[raw] trainable params: {trainable} / {total}")
    return trainable



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
            # device_map="auto",
            device_map={"": local_rank},
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.training.model.model_name,
            # quantization doesn't work on Apple Metal
            quantization_config=quant_cfg if device != 'mps' else None,
            # device_map="auto",
            device_map={"": local_rank},
            trust_remote_code=True
        )

    print("Model loaded")
    raw_trainables = count_params(model)

    peft_config = LoraConfig(
        r=cfg.training.sft_experiment.lora.r,
        lora_alpha=cfg.training.sft_experiment.lora.alpha,
        lora_dropout=cfg.training.sft_experiment.lora.dropout,
        # bias=cfg.training.sft_experiment.lora.bias, # getting NotImplementedError when set (?)
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=list(cfg.training.sft_experiment.lora.target_modules),
    )


    model = prepare_model_for_kbit_training(model)
    print("Model prepared for kbit training")
    raw_trainables = count_params(model)
    model.enable_input_require_grads()  # QLoRA needs this
    model = get_peft_model(model, peft_config)  # inject LoraLinear now
    print("PEFT model created")
    raw_trainables = count_params(model)



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

    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens", [tokenizer.eos_token])[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )

    # Convert to ID
    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)

    # Get the base EOS token ID
    base_eos_token_id = tokenizer.eos_token_id

    # Update generation config with both EOS and EOT tokens
    if hasattr(model.generation_config, 'eos_token_id'):
        # Create a list of both tokens
        eos_token_ids = []

        # Add base EOS token
        if isinstance(base_eos_token_id, list):
            eos_token_ids.extend(base_eos_token_id)
        else:
            eos_token_ids.append(base_eos_token_id)

        # Add EOT token if it's different
        if eot_token_id not in eos_token_ids:
            eos_token_ids.append(eot_token_id)

        model.generation_config.eos_token_id = eos_token_ids
    else:
        model.generation_config.eos_token_id = [
            base_eos_token_id, eot_token_id]

    # Log the configuration
    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS token ID(s): {model.generation_config.eos_token_id}")

    # Check if enhanced datasets are enabled
    if getattr(cfg.training.sft_dataset, 'use_enhanced_datasets', False):
        # Use enhanced dataset system from src/datasets.py
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils import build_sft_datasets
        
        train_dataset, eval_dataset = build_sft_datasets(
            datasets_to_use=cfg.training.sft_dataset.datasets_to_use,
            tokenizer=tokenizer,
            max_length=cfg.training.sft_dataset.max_length,
            eval_holdout_ratio=cfg.training.sft_dataset.eval_holdout_ratio,
            seed=cfg.training.sft_dataset.seed,
            pack_sequences=cfg.training.sft_dataset.pack_sequences,
            use_cache=cfg.training.sft_dataset.use_cache,
            streaming=cfg.training.sft_dataset.streaming,
        )
        logging.info("Using enhanced datasets with %d datasets: %s", 
                    len(cfg.training.sft_dataset.datasets_to_use),
                    cfg.training.sft_dataset.datasets_to_use)
    else:
        # Use legacy streaming approach
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
        train_dataset = IterableDataset.from_generator(train_gen)
        eval_dataset  = IterableDataset.from_generator(eval_gen)
        logging.info("Using legacy streaming datasets")


    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    training_args = SFTConfig(
        packing=cfg.training.sft.packing,
        # changes the tokenizers eos token to eot and the google gemma-2b-it doesn't have that will default to the list [...] in the tokenizer bos and end of turn
        eos_token=eot_token,
        completion_only_loss=cfg.training.sft.completion_only_loss,
        max_length=cfg.training.sft.max_seq_length,
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

    # Prepare regularization config for enhanced trainer
    reg_cfg = {
        "log_every": 50,
        "L_DECORR": 1e-4,
        "L_MASS": 1e-3,
        "L_ENTROPY": 0.0,
        "L_ORTHO_A": 1e-4,
        "L_ORTHO_B": 1e-4,
        "ORTHO_EVERY": 4,
        "schedule_decorr": True,
        "schedule_mass": True,
        "schedule_ent": True,
        "schedule_ortho": True
    }
    
    # Use enhanced trainer if TopK is enabled, otherwise regular trainer
    use_topk = getattr(cfg.training.sft_experiment.lora, 'use_topk', False)
    reg_mode = "z_only" if use_topk else "off"

    # ----------------------- TopK Injection (Enhanced) -----------------------
    # Check if TopK is enabled in the configuration
    if getattr(cfg.training.sft_experiment.lora, 'use_topk', False):
        logging.info("🔥 Injecting TopKLoRALinearSTE wrappers...")
        targets = []
        for name, module in model.named_modules():
            if getattr(module, "lora_A", None) is not None:
                targets.append(name)

        replaced = 0
        for name in targets:
            peft_layer = model.get_submodule(name)
            parent = model.get_submodule(".".join(name.split(".")[:-1])) if "." in name else model
            attr = name.split(".")[-1]
            setattr(parent, attr, TopKLoRALinearSTE(
                base=peft_layer,
                layer_name=name,
                k=cfg.training.sft_experiment.lora.k,
                temperature=getattr(cfg.training.sft_experiment.lora, 'temperature', 1.0),
                temperature_schedule=getattr(cfg.training.sft_experiment.lora, 'temperature_schedule', 'constant'),
                k_schedule=getattr(cfg.training.sft_experiment.lora, 'k_schedule', 'constant'),
                k_final=getattr(cfg.training.sft_experiment.lora, 'k_final', cfg.training.sft_experiment.lora.k),
                hard_eval=True, relu_latents=True, alpha_over_r=True,
                temperature_final=getattr(cfg.training.sft_experiment.lora, 'temperature_final', None),
            ))
            replaced += 1
        logging.info(f"✅ Injected TopK STE wrappers in {replaced} layers")
        enable_topk_lora_grads(model)
        print("Model after TopK injection")
        count_params(model)

    else:
        logging.info("⚪ TopK disabled - using standard LoRA training")

    from torch.optim import AdamW
    
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=cfg.training.sft.lr, weight_decay=cfg.training.sft.weight_decay)

    
    trainer = EnhancedSFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
        callbacks=[MemoryClearCallback()],
        reg_cfg=reg_cfg,
        reg_mode=reg_mode,
        # reg_mode="off",  # Temporarily disable regularization to debug TopK issues
        optimizers=(optimizer, None),
    )

    ft_model = trainer.model
    ft_model.print_trainable_parameters()
    print("Model inside trainer")
    count_params(ft_model)

    # Some launchers flip flags during init; enforce again just before training:
    enable_topk_lora_grads(trainer.model)
    count_trainables(trainer.model, "right before train()")

    # (Optional) sanity: ensure the optimizer really has params
    num_opt_params = sum(p.numel() for g in trainer.optimizer.param_groups for p in g["params"])
    assert num_opt_params > 0, "Optimizer has no params; LoRA A/B were not captured."



    if cfg.training.sft_experiment.lora.use_topk:
        topk_callbacks = [
            MemoryClearCallback(),
            TopKProgressCallback(),
        ]
        
        # Add dead latent logging if enabled
        if getattr(cfg.training.sft_experiment.lora, 'log_dead_latents', False):
            dead_latents_log_every = getattr(cfg.training.sft_experiment.lora, 'dead_latents_log_every', 500)
            topk_callbacks.append(DeadLatentsLoggerCallback(log_every=dead_latents_log_every))
            logging.info(f"📊 Added DeadLatentsLoggerCallback (log_every={dead_latents_log_every})")
        
        # Update trainer callbacks
        # trainer.callback_handler.callbacks = topk_callbacks
        for callback in topk_callbacks:
            trainer.add_callback(callback)
            callback.trainer = trainer
        
        logging.info("🔧 Updated trainer callbacks for TopK monitoring")

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
    
    
    # ----------------------- Enhanced Logging & Output Structure -----------------------
    import subprocess
    import sys
    import json
    
    # Create structured output directory similar to DPO
    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    base_output_dir = f'experiments/{model_str}_sft'
    
    # Collect hyperparameters
    hparams = {
        "model": cfg.training.model.model_name,
        "tokenizer": getattr(tokenizer, 'name_or_path', 'unknown'),
        "sft_config": {
            "max_seq_length": cfg.training.sft.max_seq_length,
            "num_epochs": cfg.training.sft.num_epochs,
            "batch_size_train": cfg.training.sft.batch_size_train,
            "gradient_accumulation_steps": cfg.training.sft.gradient_accumulation_steps,
            "learning_rate": cfg.training.sft.lr,
            "warmup_ratio": cfg.training.sft.warmup_ratio,
            "weight_decay": cfg.training.sft.weight_decay,
            "max_steps": cfg.training.sft.max_steps,
        },
        "lora_config": {
            "r": cfg.training.sft_experiment.lora.r,
            "alpha": cfg.training.sft_experiment.lora.alpha,
            "dropout": cfg.training.sft_experiment.lora.dropout,
            "target_modules": list(cfg.training.sft_experiment.lora.target_modules),
            "use_topk": getattr(cfg.training.sft_experiment.lora, 'use_topk', False),
        }
    }
    
    # Add TopK parameters if enabled
    if getattr(cfg.training.sft_experiment.lora, 'use_topk', False):
        hparams["topk_config"] = {
            "k": cfg.training.sft_experiment.lora.k,
            "k_final": getattr(cfg.training.sft_experiment.lora, 'k_final', cfg.training.sft_experiment.lora.k),
            "temperature": getattr(cfg.training.sft_experiment.lora, 'temperature', 1.0),
            "temperature_final": getattr(cfg.training.sft_experiment.lora, 'temperature_final', None),
            "temperature_schedule": getattr(cfg.training.sft_experiment.lora, 'temperature_schedule', 'constant'),
            "k_schedule": getattr(cfg.training.sft_experiment.lora, 'k_schedule', 'constant'),
        }
    
    # Add dataset info if available
    if hasattr(train_dataset, '__len__'):
        hparams["dataset"] = {
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset) if eval_dataset else 0,
        }
        if hasattr(cfg.training, 'sft_dataset'):
            hparams["dataset"].update({
                "name": getattr(cfg.training.sft_dataset, 'name', 'unknown'),
                "datasets_to_use": getattr(cfg.training.sft_dataset, 'datasets_to_use', []),
                "max_length": getattr(cfg.training.sft_dataset, 'max_length', cfg.training.sft.max_seq_length),
            })
    
    # Save hyperparameters
    os.makedirs(base_output_dir, exist_ok=True)
    with open(os.path.join(base_output_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2, default=str)
    
    # Save full config as YAML for reproducibility
    try:
        from omegaconf import OmegaConf
        with open(os.path.join(base_output_dir, "cfg.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
    except Exception as e:
        logging.warning(f"Could not serialize cfg to YAML: {e}")
    
    # Capture environment snapshots
    try:
        env_dir = os.path.join(base_output_dir, "env")
        os.makedirs(env_dir, exist_ok=True)
        
        # pip freeze
        try:
            frz = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(os.path.join(env_dir, "requirements_freeze.txt"), "wb") as f:
                f.write(frz.stdout)
        except Exception:
            pass
            
        # nvidia-smi
        try:
            smi = subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(os.path.join(env_dir, "nvidia-smi.txt"), "wb") as f:
                f.write(smi.stdout or smi.stderr)
        except Exception:
            pass
    except Exception as e:
        logging.warning(f"Failed to capture environment info: {e}")
    
    # Human-readable summary
    try:
        summary = []
        summary.append(f"model: {cfg.training.model.model_name}")
        summary.append(f"training: SFT with {cfg.training.sft_experiment.lora.r}r LoRA")
        if hasattr(train_dataset, '__len__'):
            summary.append(f"dataset: {len(train_dataset)} train samples")
        if getattr(cfg.training.sft_experiment.lora, 'use_topk', False):
            summary.append(f"topk: k={cfg.training.sft_experiment.lora.k}, temp={getattr(cfg.training.sft_experiment.lora, 'temperature', 1.0)}")
        summary.append(f"sft: lr={cfg.training.sft.lr}, steps={cfg.training.sft.max_steps}, bs={cfg.training.sft.batch_size_train}x{cfg.training.sft.gradient_accumulation_steps}")
        
        with open(os.path.join(base_output_dir, "README.txt"), "w") as f:
            f.write("\\n".join(summary) + "\\n")
    except Exception as e:
        logging.warning(f"Failed to write summary README.txt: {e}")
    
    # Update WandB config if enabled
    if getattr(cfg.logger, "report_to", None) and "wandb" in cfg.logger.report_to:
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(hparams, allow_val_change=True)
                if not getattr(cfg, "experiment_name", None):
                    wandb.run.name = os.path.basename(base_output_dir)
        except Exception as e:
            logging.warning(f"Could not update wandb config: {e}")
    
    logging.info(f"📊 Enhanced logging setup complete. Output dir: {base_output_dir}")
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

    # if trainer.optimizer is not None:
    #     for state in trainer.optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.cpu()

    # training_args.gradient_checkpointing = False


    # # force a tiny run
    # training_args = SFTConfig(
    #     **{**training_args.to_dict(),
    #     "eval_strategy": "no",
    #     "do_eval": False,
    #     "max_steps": 10,
    #     "logging_steps": 1,
    #     "save_strategy": "no",
    #     "report_to": "none",
    #     "gradient_checkpointing": True,
    #     "optim": "paged_adamw_32bit",        # prevent VRAM thrash
    #     "dataloader_num_workers": 0,         # avoid worker fork issues
    #     "disable_tqdm": False}
    # )

    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     processing_class=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=None,
    #     peft_config=None,                 # PEFT already applied
    #     callbacks=[Heartbeat()],          # only heartbeat for now
    #     optimizers=(optimizer, None),
    # )

    opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    for n, m in model.named_modules():
        if isinstance(m, TopKLoRALinearSTE):
            print(n, "A in optimizer:", id(m.A_module.weight) in opt_ids,
                    "B in optimizer:", id(m.B_module.weight) in opt_ids)


    # enable_topk_lora_grads(trainer.model)

    # raw_trainables = count_params(trainer.model)
    
    # --- one-batch A/B grad + update smoke test ---
    # model.train()
    # dl = trainer.get_train_dataloader()
    # batch = next(iter(dl))
    # batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}

    # # find your TopK wrappers and grab A/B tensors
    # wrappers = [m for m in model.modules() if isinstance(m, TopKLoRALinearSTE)]
    # assert wrappers, "No TopKLoRALinearSTE modules found"
    # A = [w.A_module.weight for w in wrappers]
    # B = [w.B_module.weight for w in wrappers]

    # # keep pre-step copies to measure updates
    # with torch.no_grad():
    #     A0 = [p.detach().clone() for p in A]
    #     B0 = [p.detach().clone() for p in B]

    # # build a tiny optimizer over LoRA params (just for this test)
    # from torch.optim import AdamW
    # ab_params = [*A, *B]
    # opt = AdamW(ab_params, lr=1e-3)

    # # forward/backward
    # out = model(input_ids=batch["input_ids"],
    #             attention_mask=batch.get("attention_mask"),
    #             labels=batch["labels"])
    # loss = out.loss
    # loss.backward()

    # # report grad norms
    # print("== Grad norms ==")
    # for i, p in enumerate(A):
    #     print(f"A[{i}] grad_norm:", float(p.grad.norm()) if p.grad is not None else None)
    # for i, p in enumerate(B):
    #     print(f"B[{i}] grad_norm:", float(p.grad.norm()) if p.grad is not None else None)

    # # do one step and measure parameter change
    # opt.step(); opt.zero_grad(set_to_none=True)

    # with torch.no_grad():
    #     print("== Update norms ==")
    #     for i, p in enumerate(A):
    #         print(f"A[{i}] Δ:", float((p - A0[i]).norm()))
    #     for i, p in enumerate(B):
    #         print(f"B[{i}] Δ:", float((p - B0[i]).norm()))
    # # ------------------------------------------------


    from transformers import TrainerCallback

    class ABProbe(TrainerCallback):
        def __init__(self, every=10):
            self.every = every
            self.prev = {}

        def _targets(self, model):
            t = []
            for name, mod in model.named_modules():
                if isinstance(mod, TopKLoRALinearSTE):
                    t.append((f"{name}.A", mod.A_module.weight))
                    t.append((f"{name}.B", mod.B_module.weight))
            return t

        def on_step_begin(self, args, state, control, model=None, **kw):
            if (state.global_step % self.every) == 0 and model is not None:
                self.prev = {name: p.detach().clone().cpu() for name, p in self._targets(model)}

        def on_step_end(self, args, state, control, model=None, **kw):
            if (state.global_step % self.every) != 0 or model is None:
                return
            logs = {}
            for name, p in self._targets(model):
                g = p.grad
                if g is not None:
                    logs[f"grad_norm/{name}"] = float(g.norm().detach().cpu())
                if name in self.prev:
                    with torch.no_grad():
                        logs[f"update_norm/{name}"] = float((p.detach().cpu() - self.prev[name]).norm())
            wandb.log(logs, step=state.global_step)
            print(logs)

    # add it
    trainer.add_callback(ABProbe(every=10))
    print('REG MODE:', trainer.reg_mode)



    # ------------------------------- Training ------------------------------
    start_ts = time.time()
    trainer.train()
    runtime_min = (time.time() - start_ts) / 60
    logging.info("Training finished in %.1f min", runtime_min)

    # ------------------------------- Saving -------------------------------
    # Use the structured output directory for consistency
    final_path = os.path.join(base_output_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Also save to legacy path for compatibility
    legacy_path = (
        f'adapters/sft/{cfg.training.sft_experiment.lora.r}-{cfg.training.sft_experiment.lora.alpha}-'
        f'{cfg.training.sft_experiment.lora.dropout}/'
        f'{getattr(cfg.training.sft_dataset, "name", "enhanced_dataset")}/'
        f'{"-".join(cfg.training.sft_experiment.lora.target_modules)}'
    )
    trainer.model.save_pretrained(legacy_path)
    tokenizer.save_pretrained(legacy_path)
    
    logging.info("✅ Adapter saved to: %s", final_path)
    logging.info("📂 Legacy path: %s", legacy_path)
    
    # Unwrap TopK wrappers before finishing (like in DPO)
    if getattr(cfg.training.sft_experiment.lora, 'use_topk', False):
        logging.info("🔧 Unwrapping TopK wrappers...")
        unwrapped = 0
        for name, module in ft_model.named_modules():
            if isinstance(module, TopKLoRALinearSTE):
                parent = ft_model.get_submodule(".".join(name.split(".")[:-1]))
                attr = name.split(".")[-1]
                setattr(parent, attr, module.lora_module)
                unwrapped += 1
        logging.info(f"✅ Reverted {unwrapped} TopK wrappers")
    
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

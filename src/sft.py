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
from src.models import TopKLoRALinear, MemoryClearCallback, CustomDPOTrainer, TopKLoRALinearSTE
from src.models import TopKProgressCallback, DeadLatentsLoggerCallback
from src.utils import build_quant_config, get_conversational_dataset, hh_rlhf_preprocess_to_messages, is_valid_dpo_pair, merge_lora_adapter, preprocess_to_messages, violates_alternation
from peft import PeftModelForCausalLM, PeftModel
import numpy as np
import logging
from peft import get_peft_model_state_dict
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
        L_MASS = float(self.reg_cfg.get("L_MASS",   0.0))
        L_ENTROPY = float(self.reg_cfg.get("L_ENTROPY", 0.0))
        L_ORTHO_A = float(self.reg_cfg.get("L_ORTHO_A", 0.0))
        L_ORTHO_B = float(self.reg_cfg.get("L_ORTHO_B", 0.0))
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

                need_decorr = self._active(L_DECORR, self.reg_cfg.get(
                    "schedule_decorr", True), w_sched)
                need_mass = self._active(L_MASS,   self.reg_cfg.get(
                    "schedule_mass", True),   w_sched)
                need_ent = self._active(L_ENTROPY, self.reg_cfg.get(
                    "schedule_ent", True),    w_sched)

                need_orthoA = (
                    self.reg_mode == "z_plus_ortho" and ORTHO_EVERY > 0 and (step % ORTHO_EVERY == 0) and
                    self._active(L_ORTHO_A, self.reg_cfg.get(
                        "schedule_ortho", True), w_sched)
                )
                need_orthoB = (
                    self.reg_mode == "z_plus_ortho" and ORTHO_EVERY > 0 and (step % ORTHO_EVERY == 0) and
                    self._active(L_ORTHO_B, self.reg_cfg.get(
                        "schedule_ortho", True), w_sched)
                )

                all_sched = (
                    self.reg_cfg.get("schedule_decorr", True) and
                    self.reg_cfg.get("schedule_mass", True) and
                    self.reg_cfg.get("schedule_ent", True) and
                    (self.reg_mode != "z_plus_ortho" or self.reg_cfg.get(
                        "schedule_ortho", True))
                )
                if all_sched and not (need_decorr or need_mass or need_ent or need_orthoA or need_orthoB):
                    if do_log:
                        acc["reg/sched_w"] += w_sched
                        n_layers += 1
                    continue

                # Prepare gates (reuse live soft gate when available)
                k_now = getattr(m, "_current_k")()
                tau = getattr(m, "_tau")()
                g_soft_live = getattr(m, "_g_soft_live", None)

                if g_soft_live is None:
                    # Local import to avoid circulars (matches your original code pattern)
                    from src.dpo import _soft_topk_mass
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


def count_params(m):
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f"[raw] trainable params: {trainable} / {total}")
    return trainable


def run_sft(cfg):
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
    count_params(model)

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
    count_params(model)
    model.enable_input_require_grads()  # QLoRA needs this
    model = get_peft_model(model, peft_config)  # inject LoraLinear now
    print("PEFT model created")
    count_params(model)

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
        sys.path.append(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
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
        eval_dataset = IterableDataset.from_generator(eval_gen)
        logging.info("Using legacy streaming datasets")

    model_str = f'{cfg.training.model.name}_{cfg.training.model.version}_{cfg.training.model.size}'
    base_output_dir = f'experiments/{model_str}_sparse_sft'
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
        output_dir=base_output_dir,
        eval_strategy=cfg.training.sft.eval_strategy,
        eval_steps=cfg.training.sft.eval_steps,
        logging_dir=f'{base_output_dir}/logs',
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
            parent = model.get_submodule(
                ".".join(name.split(".")[:-1])) if "." in name else model
            attr = name.split(".")[-1]
            wrapped = TopKLoRALinearSTE(
                base=peft_layer,
                layer_name=name,
                k=cfg.training.sft_experiment.lora.k,
                temperature=getattr(
                    cfg.training.sft_experiment.lora, 'temperature', 1.0),
                temperature_schedule=getattr(
                    cfg.training.sft_experiment.lora, 'temperature_schedule', 'constant'),
                k_schedule=getattr(
                    cfg.training.sft_experiment.lora, 'k_schedule', 'constant'),
                k_final=getattr(cfg.training.sft_experiment.lora,
                                'k_final', cfg.training.sft_experiment.lora.k),
                hard_eval=True, relu_latents=True, alpha_over_r=True,
                temperature_final=getattr(
                    cfg.training.sft_experiment.lora, 'temperature_final', None),
            )
            wrapped.train()
            setattr(parent, attr, wrapped)
            replaced += 1
        logging.info(f"✅ Injected TopK STE wrappers in {replaced} layers")
        enable_topk_lora_grads(model)
        print("Model after TopK injection")
        count_params(model)

    else:
        logging.info("⚪ TopK disabled - using standard LoRA training")

    from torch.optim import AdamW

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=cfg.training.sft.lr,
                      weight_decay=cfg.training.sft.weight_decay)

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
    num_opt_params = sum(p.numel()
                         for g in trainer.optimizer.param_groups for p in g["params"])
    assert num_opt_params > 0, "Optimizer has no params; LoRA A/B were not captured."

    if cfg.training.sft_experiment.lora.use_topk:
        topk_callbacks = [
            MemoryClearCallback(),
            TopKProgressCallback(),
        ]

        # Add dead latent logging if enabled
        if getattr(cfg.training.sft_experiment.lora, 'log_dead_latents', False):
            dead_latents_log_every = getattr(
                cfg.training.sft_experiment.lora, 'dead_latents_log_every', 500)
            topk_callbacks.append(DeadLatentsLoggerCallback(
                log_every=dead_latents_log_every))
            logging.info(
                f"📊 Added DeadLatentsLoggerCallback (log_every={dead_latents_log_every})")

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
        summary.append(
            f"training: SFT with {cfg.training.sft_experiment.lora.r}r LoRA")
        if hasattr(train_dataset, '__len__'):
            summary.append(f"dataset: {len(train_dataset)} train samples")
        if getattr(cfg.training.sft_experiment.lora, 'use_topk', False):
            summary.append(
                f"topk: k={cfg.training.sft_experiment.lora.k}, temp={getattr(cfg.training.sft_experiment.lora, 'temperature', 1.0)}")
        summary.append(
            f"sft: lr={cfg.training.sft.lr}, steps={cfg.training.sft.max_steps}, bs={cfg.training.sft.batch_size_train}x{cfg.training.sft.gradient_accumulation_steps}")

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

    logging.info(
        f"📊 Enhanced logging setup complete. Output dir: {base_output_dir}")

    opt_ids = {id(p)
               for g in trainer.optimizer.param_groups for p in g["params"]}
    for n, m in model.named_modules():
        if isinstance(m, TopKLoRALinearSTE):
            print(n, "A in optimizer:", id(m.A_module.weight) in opt_ids,
                  "B in optimizer:", id(m.B_module.weight) in opt_ids)

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
                self.prev = {name: p.detach().clone().cpu()
                             for name, p in self._targets(model)}

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
                        logs[f"update_norm/{name}"] = float(
                            (p.detach().cpu() - self.prev[name]).norm())
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
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    adapter_state_dict = get_peft_model_state_dict(
        model,
        state_dict=model.state_dict(),
        adapter_name="default"
    )
    torch.save(adapter_state_dict, f"{final_path}/adapter_model.bin")
    # or use safetensors
    from safetensors.torch import save_file
    save_file(adapter_state_dict, f"{final_path}/adapter_model.safetensors")

    logging.info("✅ Adapter saved to: %s", final_path)
    logging.info("📂 Legacy path: %s", legacy_path)

    from safetensors.torch import load_file
    saved = load_file(f"{final_path}/adapter_model.safetensors")
    print("Saved keys sample:")
    for key in list(saved.keys())[:5]:
        print(f"  {key}")

    wandb.finish()

    return trainer.model

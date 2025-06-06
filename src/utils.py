from typing import Dict, Any, List, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import apply_chat_template
from datasets import load_dataset
from peft import PeftModel
import torch
import heapq


def get_conversational_dataset(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name)

    if dataset_name == 'xlangai/spider':
        return get_spider_dataset(dataset_name, tokenizer)

    train, test = dataset['train'], dataset['test']

    train_dataset = train.map(
        apply_chat_template, fn_kwargs={'tokenizer': tokenizer}
    )
    test_dataset = test.map(
        apply_chat_template, fn_kwargs={'tokenizer': tokenizer}
    )

    return train_dataset, test_dataset


def get_spider_dataset(dataset_name, tokenizer):
    # source: https://medium.com/%40shekhars271991/finetuning-llama-3-2-eef3114b5f6c
    def format_entries(entry):
        # Format conversations as list of dictionaries with alternating user/assistant messages
        conversations = [
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": query}
            ]
            for question, query in zip(entry["question"], entry["query"])
        ]
        # Apply chat template to each conversation
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False)
            for convo in conversations
        ]
        return {"text": texts}

    train = load_dataset(dataset_name, split='train')
    test = load_dataset(dataset_name, split='validation[:10%]')
    train_dataset = train.map(format_entries, batched=True)
    test_dataset = test.map(format_entries, batched=True)
    return train_dataset, test_dataset


def merge_lora_adapter(
    base_model_dir: str,
    lora_checkpoint_dir: str,
    merged_output_dir: str,
    tokenizer_dir: str = None,
    torch_dtype="auto",
    device_map="auto"
):
    """
    Load a base model and its tokenizer (optionally from a separate directory),
    merge LoRA adapter weights, and save the merged model to the specified output directory.

    Args:
        base_model_dir (str): Path to the directory containing the base model files.
        lora_checkpoint_dir (str): Path to the directory containing LoRA adapter files 
                                   (e.g., adapter_config.json, lora_adapters.pt).
        merged_output_dir (str): Path to the directory where the merged model will be saved.
        tokenizer_dir (str, optional): Path to the directory from which to load the tokenizer.
                                       If None, defaults to `base_model_dir`.
        torch_dtype (str or torch.dtype, optional): Data type to load the model with. 
                                                   Defaults to 'auto'.
        device_map (str or dict, optional): Device map for loading the model. Defaults to 'auto'.

    Returns:
        None
    """
    # If no separate tokenizer directory is given, use the base_model_dir
    if tokenizer_dir is None:
        tokenizer_dir = base_model_dir

    # 1. Load the tokenizer from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 2. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    # 3. Load the LoRA adapter on top of the base model
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_dir
    )

    # 4. Merge LoRA weights into the base model
    merged_model = model_with_lora.merge_and_unload()

    # 5. Save the merged model and tokenizer
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    print(f"Merged model saved to: {merged_output_dir}")


# -----------------------------------------------------------------------------
# Pre‑processing
# -----------------------------------------------------------------------------

def preprocess_to_messages(example: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Convert Alpaca record to ChatML messages expected by TRL‑SFT."""
    instruction = example["instruction"].strip()
    user_input = example.get("input", "").strip()
    response = example["output"].strip()
    user_content = f"{instruction}\n\n{user_input}" if user_input else instruction
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


# -----------------------------------------------------------------------------
# Quantisation helper
# -----------------------------------------------------------------------------

def build_quant_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    """Return a BitsAndBytesConfig from YAML sub‑dict."""
    qtype = cfg["type"].lower()
    if qtype == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.get("compute_dtype", "bfloat16")),
        )
    if qtype == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=cfg.get("llm_int8_threshold", 6.0),
        )
    raise ValueError(f"Unsupported quantisation type: {qtype}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def violates_alternation(msgs: List[Dict[str, str]]) -> bool:
    """
    True  → conversation breaks the template rule
    False → conversation is OK
    Rules:
    • first turn must be 'user' or 'system'
    • thereafter roles must strictly alternate user↔assistant
    """
    if not msgs:                                      # empty conversation
        return True

    # ── first speaker
    if msgs[0]["role"] not in {"user", "system"}:
        return True

    # ── alternation check
    for prev, curr in zip(msgs, msgs[1:]):
        if prev["role"] == curr["role"]:              # same role twice
            return True
        # user/system must be followed by assistant, and vice‑versa
        if prev["role"] in {"user", "system"} and curr["role"] != "assistant":
            return True
        if prev["role"] == "assistant" and curr["role"] not in {"user", "system"}:
            return True

    return False


def is_valid_dpo_pair(msgs):
    """True → OK for DPO; False → drop."""
    if len(msgs) < 2:
        return False
    if msgs[-1]["role"] != "assistant":   # must end with assistant answer
        return False
    return True


_TAG_RE = __import__("re").compile(r"(Human|Assistant):")
_ROLE_MAP = {"Human": "user", "Assistant": "assistant"}


def hh_string_to_messages(text: str) -> List[Dict[str, str]]:
    """
    Convert a raw Anthropic HH conversation string into Chat‑ML messages.
    Example:
        "Human: Hi. Assistant: Hello!"  →
        [{"role":"user","content":"Hi."},
         {"role":"assistant","content":"Hello!"}]
    """
    parts, msgs = _TAG_RE.split(text), []
    for i in range(1, len(parts), 2):
        role_tag, content = parts[i].strip(), parts[i + 1].strip()
        if content:
            msgs.append({"role": _ROLE_MAP[role_tag], "content": content})
    return msgs


def preprocess_to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map HH‑RLHF record → {'chosen': [...], 'rejected': [...]} chat lists."""
    return {
        "chosen":   hh_string_to_messages(example["chosen"]),
        "rejected": hh_string_to_messages(example["rejected"]),
    }


def build_quant_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    """Return a BitsAndBytesConfig from YAML sub‑dict."""
    qtype = cfg["type"].lower()
    if qtype == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.get("compute_dtype", "bfloat16")),
        )
    if qtype == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=cfg.get("llm_int8_threshold", 6.0),
        )
    raise ValueError(f"Unsupported quantisation type: {qtype}")


def make_topk_hook(
    module_name: str, k_heap: int, ax_topk: int,
    topk_store: Dict[str, Dict[int, Dict[str, List[Tuple[float, int, int, float]]]]],
):
    """
    Build a forward hook for a PEFT lora.Linear layer that
      • computes h = A·x
      • optionally selects the Top-AX_TOPK neurons (|h|) per token
      • stores top-k_heap examples per (layer, neuron, sign) in `topk_store`.
    """
    def _hook(module, inp, _):
        # global: dataset row index of batch[0]
        ex_offset = CURRENT_EX_OFFSET
        mask = CURRENT_PAD_MASK      # (B, L) bool, True = real token

        # ---------- resolve which adapter key to use ----------------------
        adapter = module.active_adapter or next(iter(module.lora_A))
        if isinstance(adapter, (list, tuple)):
            adapter = adapter[0]

        # ---------- compute A·x -------------------------------------------
        x = inp[0]                        # (B, L, D_hidden)
        h = module.lora_A[adapter](x)     # (B, L, r)
        if h.ndim == 2:                   # some PEFT ops flatten B·L
            B, L = x.shape[:2]
            h = h.view(B, L, -1)

        B, L, R = h.shape
        if mask is None:
            mask = h.new_ones((B, L), dtype=torch.bool)

        # ---------- iterate over real tokens ------------------------------
        for b in range(B):
            ex_idx = ex_offset + b
            valid_len = int(mask[b].sum())
            pad_left = L - valid_len

            for pos in range(L):
                if not mask[b, pos]:
                    continue   # skip padding
                true_pos = pos - pad_left
                act_vec = h[b, pos]                      # (R,)

                # --- select which neuron indices to store ----------------
                if ax_topk is None or ax_topk >= R:
                    store_idx = range(R)                  # keep all
                else:
                    store_idx = act_vec.abs().topk(ax_topk).indices.tolist()

                for n_idx in store_idx:
                    raw_val = act_vec[n_idx].item()
                    sign = "pos" if raw_val >= 0 else "neg"
                    # TODO: make topk_store part of local frame
                    #       let's not rely on global state
                    heap = topk_store[module_name][n_idx][sign]

                    mag = abs(raw_val)
                    item = (mag, ex_idx, true_pos, raw_val)

                    if len(heap) < k_heap:
                        heapq.heappush(heap, item)
                    elif mag > heap[0][0]:
                        heapq.heapreplace(heap, item)

    return _hook

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build chat messages for apply_chat_template
# ─────────────────────────────────────────────────────────────────────────────


def build_metrics_eval_messages(question: str, reply_a: str, reply_b: str) -> List[Dict]:
    user = (
        f"{question}\n\n"
        f"### Reply A:\n{reply_a}\n\n"
        f"### Reply B:\n{reply_b}\n\n"
        "Which reply is better? Answer with A or B only."
    )
    return [
        {"role": "user",   "content": user},
        # The assistant role is left blank; the tokenizer adds the tag.
    ]

from __future__ import annotations
import json
from transformers import PreTrainedTokenizerBase
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
from typing import List, Dict, Tuple
import heapq
from peft import PeftModel
from datasets import load_dataset, load_from_disk
from trl import apply_chat_template
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from tqdm import tqdm
from typing import Dict, Any, List, Mapping, Optional, Tuple
import hashlib
import pickle
import re
import random
# --- UltraChat token streaming utilities for TopKLoRA autointerp ---
import os
import torch
from typing import Tuple


def ultrachat_to_flat_tokens(
    tokenizer,
    splits=("train_sft",),
    add_eos_between=True,
    eos_id=None,
    num_proc=None,
    render_batch_size=256,
    encode_batch_size=1024,
    token_budget=None,  # int or None
):
    """
    Returns a 1-D torch.LongTensor of token ids from UltraChat, normalized and rendered with chat template.
    """
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    from src.utils import load_ultrachat, normalize_ultrachat_messages
    ds = load_ultrachat()

    def render_batch(batch):
        texts = []
        for msgs in batch["messages"]:
            fixed, need_gen = normalize_ultrachat_messages(msgs)
            txt = tokenizer.apply_chat_template(
                fixed,
                tokenize=False,
                add_generation_prompt=need_gen
            )
            texts.append(txt)
        return {"text": texts}

    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])
    ds = ds.map(
        render_batch,
        batched=True,
        batch_size=render_batch_batch_size if render_batch_size is None else render_batch_size,
        num_proc=(num_proc or os.cpu_count() or 4),
        desc="Render chat templates",
    )
    texts = ds["text"]

    pieces = []
    total = 0
    for i in range(0, len(texts), encode_batch_size):
        chunk = texts[i:i + encode_batch_size]
        enc = tokenizer(
            chunk,
            add_special_tokens=False,
            return_attention_mask=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )["input_ids"]

        for ids in enc:
            if add_eos_between and eos_id is not None:
                ids = ids + [eos_id]
            t = torch.tensor(ids, dtype=torch.long)
            if token_budget is None:
                pieces.append(t)
            else:
                if total >= token_budget:
                    break
                need = token_budget - total
                if t.numel() <= need:
                    pieces.append(t)
                    total += t.numel()
                else:
                    pieces.append(t[:need])
                    total += need
                    break
        if token_budget is not None and total >= token_budget:
            break

    if not pieces:
        return torch.empty(0, dtype=torch.long)

    return torch.cat(pieces, dim=0)


def pack_1d_stream(tokens_1d: torch.Tensor, seq_len: int) -> torch.Tensor:
    usable = (tokens_1d.numel() // seq_len) * seq_len
    if usable == 0:
        raise ValueError("Not enough tokens to form a single window.")
    return tokens_1d.narrow(0, 0, usable).view(-1, seq_len)


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
    quantization_config,
    merged_output_dir: Optional[str] = None,
    save_merged_model: bool = False,
    tokenizer_dir: str = None,
    torch_dtype="auto",
    device_map="auto",
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
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    # 3. Load the LoRA adapter on top of the base model
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_dir,
        use_safetensors=True
    )

    # 4. Merge LoRA weights into the base model
    merged_model = model_with_lora.merge_and_unload()

    if save_merged_model:
        # 5. Save the merged model and tokenizer
        assert merged_output_dir is not None, 'Cannot save merged model without providing output dir'
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

        print(f"Merged model saved to: {merged_output_dir}")

        # saving and loading the same model removes peft-related attributes
        merged_model = AutoModelForCausalLM.from_pretrained(
            merged_output_dir
        )

    return merged_model

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

# def build_quant_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
#     """Return a BitsAndBytesConfig from YAML sub‑dict."""
#     qtype = cfg["type"].lower()
#     if qtype == "4bit":
#         return BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
#             bnb_4bit_compute_dtype=getattr(
#                 torch, cfg.get("compute_dtype", "bfloat16")),
#         )
#     if qtype == "8bit":
#         return BitsAndBytesConfig(
#             load_in_8bit=True,
#             llm_int8_threshold=cfg.get("llm_int8_threshold", 6.0),
#         )
#     raise ValueError(f"Unsupported quantisation type: {qtype}")


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


def hh_rlhf_preprocess_to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map HH‑RLHF record → {'chosen': [...], 'rejected': [...]} chat lists."""
    return {
        "chosen":   hh_string_to_messages(example["chosen"]),
        "rejected": hh_string_to_messages(example["rejected"]),
    }


def build_quant_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    """Return a BitsAndBytesConfig from YAML sub‑dict."""
    qtype = cfg.type.lower()
    if qtype == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.compute_dtype
            ),
        )
    if qtype == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=cfg.get("llm_int8_threshold", 6.0),
        )
    raise ValueError(f"Unsupported quantisation type: {qtype}")


"""
Autointerp eval helper functions
"""


def autointerp_make_topk_hook(
    module_name: str, cfg,
    topk_store: Dict[str, Dict[int, Dict[str, List[Tuple[float, int, int, float]]]]],
    current_vals
):
    """
    Build a forward hook for a PEFT lora.Linear layer that
      • computes h = A·x
      • optionally selects the Top-AX_TOPK neurons (|h|) per token
      • stores top-k_heap examples per (layer, neuron, sign) in `topk_store`.
    """
    def _hook(module, inp, _):
        # print('hook')
        # global: dataset row index of batch[0]
        ex_offset = current_vals['current_ex_offset']
        # (B, L) bool, True = real token
        mask = current_vals['current_pad_mask']

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
                # TODO: study critical error here
                # NOTE: this error is caused by
                #       setting pad_left and truncate_left
                #       in the tokenizer for LoRA fine tuning
                true_pos = pos - pad_left
                # NOTE: removing the subtraction of pad_left
                # NOTE: this is very dangerous without understanding
                #       the underlying logic
                # true_pos = pos

                act_vec = h[b, pos]                      # (R,)

                # --- select which neuron indices to store ----------------
                if cfg.evals.auto_interp.ax_topk is None or \
                        cfg.evals.auto_interp.ax_topk >= R:
                    store_idx = range(R)                  # keep all
                else:
                    store_idx = act_vec.abs().topk(
                        cfg.evals.auto_interp.ax_topk
                    ).indices.tolist()

                for n_idx in store_idx:
                    raw_val = act_vec[n_idx].item()
                    sign = "pos" if raw_val >= 0 else "neg"
                    # TODO: make topk_store part of local frame
                    #       let's not rely on global state
                    # print('here')
                    heap = topk_store[module_name][n_idx][sign]

                    mag = abs(raw_val)
                    item = (mag, ex_idx, true_pos, raw_val)

                    if len(heap) < cfg.evals.auto_interp.k_heap:
                        heapq.heappush(heap, item)
                    elif mag > heap[0][0]:
                        heapq.heapreplace(heap, item)

    return _hook


def autointerp_hh_string_to_messages(text: str) -> List[Dict[str, str]]:
    parts, msgs = _TAG_RE.split(text), []
    for i in range(1, len(parts), 2):
        role, content = parts[i].strip(), parts[i+1].strip()
        if content:
            msgs.append({"role": _ROLE_MAP[role], "content": content})
    return msgs


def autointerp_preprocess_to_messages(ex: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chosen":   hh_string_to_messages(ex["chosen"]),
        "rejected": hh_string_to_messages(ex["rejected"]),
    }


def autointerp_violates_alternation(msgs: List[Dict[str, str]]) -> bool:
    if not msgs:
        return True
    if msgs[0]["role"] not in {"user", "system"}:
        return True
    for prev, curr in zip(msgs, msgs[1:]):
        if prev["role"] == curr["role"]:
            return True
        if prev["role"] in {"user", "system"} and curr["role"] != "assistant":
            return True
        if prev["role"] == "assistant" and curr["role"] not in {"user", "system"}:
            return True
    return False


def autointerp_is_valid_dpo_pair(msgs):
    return len(msgs) >= 2 and msgs[-1]["role"] == "assistant"


class AutointerpChatCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        dialogs = [ex["input"] for ex in batch]

        # `ids` is a tensor (B, L)
        ids = self.tokenizer.apply_chat_template(
            dialogs,
            add_generation_prompt=False,
            padding=True,            # pad to longest in *this* batch
            return_tensors="pt"
        )

        mask = (ids != self.tokenizer.pad_token_id).long()   # (B, L)

        return {
            "input_ids":      ids.to(self.device),
            "attention_mask": mask.to(self.device)
        }


def autointerp_collapse_heaps(topk_store):
    """
    adapter → neuron → rank → (ex_idx, true_pos, raw_val)
    Positive ranks 1..k, negative ranks k+1..2k (both sorted by |val| desc).
    """
    out = {}
    for module, n_map in topk_store.items():
        n_out = {}
        for n_idx, heaps in n_map.items():
            combined = sorted(
                heaps["pos"] + heaps["neg"],
                key=lambda t: t[0],
                reverse=True
            )
            n_out[n_idx] = {
                rank + 1: (ex, pos, raw)
                for rank, (_, ex, pos, raw) in enumerate(combined)
            }
        out[module] = n_out
    return out


def autointerp_extract_windows_from_dataset(
    topk_map: Dict[int, Tuple[int, int]],
    dataset: Any,
    tokenizer,
    window: int = 7
) -> Dict[int, str]:
    """
    Return a dict mapping rank → context snippet, where the target token
    at `tok_pos` is wrapped in << >> within a ±window token window.
    """
    out: Dict[int, str] = {}

    for rank, (ex_idx, tok_pos, val) in topk_map.items():
        # Re-tokenize the single example (no padding)
        ids = tokenizer.apply_chat_template(
            dataset[ex_idx]["input"],
            add_generation_prompt=False,
            padding=False,
            return_tensors="pt"
        )[0]  # shape (L,)

        # Determine window slice
        start = max(0, tok_pos - window)
        end = min(ids.size(0), tok_pos + window + 1)
        snippet_ids = ids[start:end].tolist()

        # Decode one token at a time to preserve alignment
        toks = [
            tokenizer.decode([tid], skip_special_tokens=False)
            for tid in snippet_ids
        ]

        # Wrap the center token
        center = tok_pos - start
        toks[center] = f" <<{toks[center]}>> "

        # Re-join into a single string (preserving any newlines)
        snippet = "".join(toks)

        out[rank] = snippet

    return out


def autointerp_build_activation_prompt(
    windows_dict: Dict[int, Dict[str, str]],
    topk_map: Mapping[int, Tuple[int, int]],
    *,
    newline: str = "\n",            # "\n" for plain text, "<br>" for Markdown
    context_newline: bool = False,  # convert "⏎" back to real newlines?
) -> str:
    """
    Build a prompt like

        Example 1: … token …  
        Activations: ("token", 42)

    Parameters
    ----------
    newline
        The line-separator to use between lines.  Change to "<br>" if you plan
        to render in Markdown/HTML and don’t want actual carriage returns.
    context_newline
        If True, every literal "⏎" that appears inside the context slice is
        replaced by the chosen `newline` character so you regain the original
        formatting of multi-line examples.
    """
    blocks = []
    for rank in sorted(windows_dict):
        token = windows_dict[rank]["token"]
        context = windows_dict[rank]["context"]
        pos = topk_map[rank][1]
        val = topk_map[rank][2]

        if context_newline:
            context = context.replace("⏎", newline)

        block = (
            f"Example {rank}:{newline}"
            f"{context}{newline}"
            f"Activations: (\"{token.strip()}\", {val})"
        )
        blocks.append(block)

    # Two line-breaks between blocks so each example is visually separated
    return (newline * 2).join(blocks)


def autointerp_token_windows_dict(
    # {rank: (ex_idx, tok_pos, val)}
    topk_map: Mapping[int, Tuple[int, int, float]],
    dataset: Any,
    tokenizer,
    window: int = 7,
    # new: only keep ranks ≤ topk
    topk: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Return a dict of the form
        { rank: { "token": <str>, "context": <str>, "value": <float> }, ... }

    * `window` controls how many tokens are shown on each side.
    * `topk`, if set, limits to only ranks 1..topk.
    * Newlines inside the context are replaced with the glyph '⏎'
      so the string stays single-line (handy for printing or logging).
    """
    # from pprint import pprint
    # pprint(topk_map)
    # assert False
    result: Dict[int, Dict[str, Any]] = {}
    for rank, (ex_idx, tok_pos, val) in sorted(topk_map.items()):
        # stop if we've reached the desired topk
        if topk is not None and int(rank) > topk:
            break

        # Build the full token-id sequence for this example
        ids = tokenizer.apply_chat_template(
            dataset[ex_idx]["input"],
            add_generation_prompt=False,
            padding=False,
            return_tensors="pt",
        )[0]  # shape (L,)

        # Slice out a small window around the target position
        start = max(0, tok_pos - window)
        end = min(ids.size(0), tok_pos + window + 1)
        ids_slice = ids[start:end].tolist()

        # Decode one token at a time to keep byte-level alignment
        toks = [
            tokenizer.decode([tid], skip_special_tokens=False)
            for tid in ids_slice
        ]

        center = tok_pos - start          # position of the “hot” token
        token_text = toks[center]         # raw decoded token

        # Wrap the center token in << ... >>
        toks[center] = f"<<{toks[center]}>>"

        # Join and replace newlines for single-line context
        # TODO: do we actually want to do this?
        #       we can use repr(string) for logging
        #       because this unicode character might
        #       bias results (?)
        snippet = "".join(toks).replace("\n", "⏎")

        result[rank] = {
            "token": token_text,
            "context": snippet,
            "value": val,
        }

    return result


def autointerp_generate_system_prompt(
    *,
    include_cot: bool = False,
    include_few_shot: bool = False,
) -> str:
    """
    Return the complete system prompt used for the activation-analysis task.

    Parameters
    ----------
    include_cot
        If True, the prompt explicitly encourages the assistant to think
        through the three analysis stages (1…3).
    include_few_shot
        If True, append illustrative few-shot examples.  When both flags are
        True the COT-annotated few-shot block is used; otherwise a plain,
        interpretation-only block is shown.
    """
    base_guidelines = """You are a meticulous AI researcher conducting an important
investigation into patterns found in language. Your task is to
analyze text and provide an interpretation that thoroughly
encapsulates possible patterns found in it.
Guidelines:
You will be given a list of text examples on which special words
are selected and between delimiters like << this >>.
If a sequence of consecutive tokens all are important,
the entire sequence of tokens will be contained between
delimiters <<just like this>>. How important each token is for
the behavior is listed after each example in parentheses.
- Try to produce a concise final description. Simply describe
  the text latents that are common in the examples, and what
  patterns you found.
- If the examples are uninformative, you don’t need to mention
  them. Don’t focus on giving examples of important tokens,
  but try to summarize the patterns found in the examples.
- Do not mention the marker tokens ($<<$ $>>$) in your interpretation.
- Do not make lists of possible interpretations.
  Keep your interpretations short and concise.
- The last line of your response must be the formatted
  interpretation, using [interpretation]:"""

    cot_instructions = """
To better find the interpretation for the language patterns,
go through the following stages:
1. Find the special words that are selected in the examples and list
   a couple of them (no more than five). Search for patterns in these words.
2. Write down general shared latents of the text examples.
   This could be related to the full sentence or to the words
   surrounding the marked words.
3. Formulate a hypothesis and write down the final interpretation
   using [interpretation]:"""

    # ---------- few-shot templates ----------
    few_shot_plain = """
Example 1: and he was <<over the moon>> to find
Activations: (“over", 5), (“ the", 6), (“ moon", 9)

Example 2: we'll be laughing <<till the cows come home>>!
Activations: (“till", 5), (“ the", 5), (“ cows", 8),
(“ come", 8), (“ home", 8)

Example 3: thought Scotland was boring, but really there’s
more <<than meets the eye>>!
Activations: (“than", 5), (“ meets", 7), (“ the", 6), (“ eye", 8)

[interpretation]: Common idioms in text conveying positive sentiment.
"""

    few_shot_cot = """
Example 1: and he was <<over the moon>> to find
Activations: (“over", 5), (“ the", 6), (“ moon", 9)

Example 2: we'll be laughing <<till the cows come home>>!
Activations: (“till", 5), (“ the", 5), (“ cows", 8),
(“ come", 8), (“ home", 8)

Example 3: thought Scotland was boring, but really there’s
more <<than meets the eye>>!
Activations: (“than", 5), (“ meets", 7), (“ the", 6), (“ eye", 8)

ACTIVATING TOKENS: “over the moon”, “than meets the eye”.
SURROUNDING TOKENS: No interesting patterns.

Step 1.
- The activating tokens are parts of common idioms.
- The surrounding tokens have nothing in common.

Step 2.
- The examples contain common idioms.
- Some activating tokens are followed by an exclamation mark.

Step 3.
- The activation values are highest for the more common idioms
  in examples 1 and 3.
Let me think carefully … Did I miss any patterns?
Yes: all examples convey positive sentiment.

[interpretation]: Common idioms in text conveying positive sentiment.
"""

    # ---------- assemble the prompt ----------
    parts = [base_guidelines]

    if include_cot:
        parts.append(cot_instructions)

    if include_few_shot:
        parts.append(few_shot_cot if include_cot else few_shot_plain)

    return "\n\n".join(parts)


def autointerp_extract_interpretation(response_text: str) -> Optional[str]:
    """
    Extract the final interpretation from a model response string using the
    `[interpretation]:` marker.

    Returns the interpretation string (stripped), or None if no match is found.
    """
    match = re.search(r"\[interpretation\]\s*:\s*(.+)",
                      response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def autointerp_build_lora_json_with_responses(
    adapters_pos_map: Dict[str, Mapping[int, Tuple[int, int, float]]],
    dataset: Any,
    tokenizer,
    *,
    window: int = 32,
    model: str = "gpt-4o",
    include_cot: bool = False,
    include_few_shot: bool = False,
    temperature: float = 0.0,
    max_tokens: int | None = 512,
    outfile: str = "temp/lora_neuron_info.json",
    client: OpenAI | None = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generates `outfile` in the format needed by the Gradio app,
    using the Responses API (`client.responses.create`).

    The output JSON will now include:
      * interpretation: str
      * top_activations: List[str]   (contexts)
      * values:          List[float] (activation values)
    """
    if client is None:
        client = OpenAI()

    system_prompt = autointerp_generate_system_prompt(
        include_cot=include_cot, include_few_shot=include_few_shot
    )

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for adapter, neuron_maps in tqdm(
        adapters_pos_map.items(), desc="Adapters"
    ):
        adapter_block: Dict[str, Dict[str, Any]] = {}
        for n_idx, pos_map in tqdm(
            neuron_maps.items(),
            desc=f"  {adapter}", leave=False
        ):
            # 1) build example windows (each entry also has a "value" field)

            pos_map = {int(key): val for key, val in pos_map.items()}

            windows_dict = autointerp_token_windows_dict(
                pos_map, dataset, tokenizer,
                window=window,
                topk=40
            )

            # 2) craft user prompt
            prompt = autointerp_build_activation_prompt(
                windows_dict,
                pos_map,
                newline="\n",
                context_newline=True,
            )
            # 3) model call via Responses API
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=prompt,
                temperature=temperature,
            )
            answer_text = response.output_text
            interpretation = autointerp_extract_interpretation(
                answer_text) or "N/A"

            # 4) collect the top‐k contexts *and* their activation values
            ranks = sorted(windows_dict)
            topk_snippets = [windows_dict[r]["context"] for r in ranks]
            topk_values = [windows_dict[r]["value"] for r in ranks]

            adapter_block[str(n_idx)] = {
                "interpretation":  interpretation,
                "top_activations": topk_snippets,
                "values":          topk_values,
            }

        results[adapter] = adapter_block

    # 5) write to disk
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"✓ Saved {outfile}  "
        f"({len(results)} adapters × {len(next(iter(results.values())))} neurons)"
    )
    return results


def autointerp_build_lora_json_with_responses_local(
    adapters_pos_map: Dict[str, Mapping[int, Tuple[int, int, float]]],
    dataset: Any,
    tokenizer_name_or_obj,
    *,
    window: int = 32,
    model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    include_cot: bool = False,
    include_few_shot: bool = False,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    device: str = "cuda",
    outfile: str = "temp/lora_neuron_info_local.json"
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Same as before, but uses a local HF‐style model.generate instead of OpenAI’s API.
    """

    # 1) load model & tokenizer (once)
    if isinstance(tokenizer_name_or_obj, str):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_obj, use_fast=True)
    else:
        tokenizer = tokenizer_name_or_obj

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",      # or model.to(device)
        torch_dtype=torch.float16,  # if your GPU supports it
        # low_cpu_mem_usage=True,
        # trust_remote_code=True,     # if the repo needs it
    )
    model.eval()

    # build system prompt just once
    system_prompt = autointerp_generate_system_prompt(
        include_cot=include_cot, include_few_shot=include_few_shot
    )

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for adapter, neuron_maps in tqdm(adapters_pos_map.items(), desc="Adapters"):
        adapter_block: Dict[str, Dict[str, Any]] = {}
        for n_idx, pos_map in tqdm(neuron_maps.items(), desc=f"  {adapter}", leave=False):
            pos_map = {int(k): v for k, v in pos_map.items()}
            windows_dict = autointerp_token_windows_dict(
                pos_map, dataset, tokenizer, window=window, topk=40
            )
            prompt = autointerp_build_activation_prompt(
                windows_dict,
                pos_map,
                newline="\n",
                context_newline=True,
            )

            # 2) Prepare the full text we feed the model
            full_input = system_prompt + "\n\n" + prompt

            # 3) Tokenize + generate
            inputs = tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=4096,     # adjust to your model’s context window
            ).to(model.device)

            with torch.no_grad():
                generation_output = model.generate(
                    **inputs,
                    do_sample=temperature > 0.0,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 4) Decode just the **new** tokens
            generated = generation_output[0][inputs["input_ids"].shape[-1]:]
            answer_text = tokenizer.decode(generated, skip_special_tokens=True)

            interpretation = autointerp_extract_interpretation(
                answer_text) or "N/A"

            ranks = sorted(windows_dict)
            topk_snippets = [windows_dict[r]["context"] for r in ranks]
            topk_values = [windows_dict[r]["value"] for r in ranks]

            adapter_block[str(n_idx)] = {
                "interpretation":  interpretation,
                "top_activations": topk_snippets,
                "values":          topk_values,
            }

        results[adapter] = adapter_block

    # 5) write to disk
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"✓ Saved {outfile}  ({len(results)} adapters × {len(next(iter(results.values())))} neurons)")
    return results


TOKENIZER_CACHE: Dict[str, Any] = {}


def autointerp_tok(model):
    if model not in TOKENIZER_CACHE:
        TOKENIZER_CACHE[model] = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True)
    return TOKENIZER_CACHE[model]


def autointerp_token_window(dataset, tokenizer, ex_idx: int, pos: int, window: int = 7):
    """Return ±`window` tokens around *pos* with center wrapped in <<…>>."""
    ids = tokenizer.apply_chat_template(
        dataset[ex_idx]["input"],
        add_generation_prompt=False,
        padding=False,
        return_tensors="pt"
    )[0]
    seq_len = ids.size(0)

    if pos < 0 or pos >= seq_len:
        # Token position is invalid — skip the example
        return None

    s = max(0, pos - window)
    e = min(seq_len, pos + window + 1)
    toks = [tokenizer.decode([tid], skip_special_tokens=False)
            for tid in ids[s:e]]

    center_idx = pos - s
    if 0 <= center_idx < len(toks):
        toks[center_idx] = f"<<{toks[center_idx]}>>"
        return "".join(toks).replace("\n", "⏎")
    else:
        # Token not in window (some rare edge case) — skip
        return None


def autointerp_ctx_lorainfo(layer, nid, rank, lora_info, model, tok_id: Optional[int]):
    tops = lora_info.get(layer, {}).get(nid, {}).get("top_activations", [])
    if 0 < rank <= len(tops):
        return tops[rank-1].replace("<<", "").replace(">>", "").strip()
    if tok_id is not None:
        return autointerp_tok(model).decode([tok_id]).strip()
    return "[UNK]"


def autointerp_clean(s):
    return re.sub(r"\s+", " ", re.sub(r"<<\s*|\s*>>", "", s)).strip()


def autointerp_build_dataset(act_dict, lora_info, examples, model, window=7, k_skip=40,
                             n_examples=10, n_neg=4, seed=42):
    random.seed(seed)
    tok = autointerp_tok(model)

    # Build: (layer,nid) -> rows[(tid?,pos,val,rank,ex_idx)]
    rows_by_neuron: Dict[Tuple[str, str],
                         List[Tuple[Optional[int], int, float, int, int]]] = {}
    for layer, ns in act_dict.items():
        for nid, acts in ns.items():
            rows = []
            for rk_str, trip in acts.items():
                if len(trip) == 3:
                    ex_idx, pos, val = trip
                    tid = None
                elif len(trip) == 4:
                    ex_idx, tid, pos, val = trip
                else:
                    raise ValueError("Activation entry must have 3 or 4 items")
                rows.append((tid, pos, val, int(rk_str), ex_idx))
            rows.sort(key=lambda t: -t[2])
            rows_by_neuron[(layer, nid)] = rows

    all_keys = list(rows_by_neuron.keys())
    dataset = []

    def ctx(layer, nid, tid, pos, rank, ex_idx):
        if ex_idx is not None and ex_idx < len(examples):
            return autointerp_token_window(examples, tok, ex_idx, pos, window)
        return autointerp_ctx_lorainfo(layer, nid, rank, lora_info, model, tid)

    invalid_examples = 0
    for (layer, nid), rows in rows_by_neuron.items():
        if len(rows) <= k_skip:
            continue

        pos_rows = rows[k_skip:]
        positives = random.choices(pos_rows, k=n_examples)
        pos_positions = {p for _tid, p, _v, _r, _e in rows}

        for tid, pos, _v, rank, ex_idx in positives:
            negs, tries = [], 0

            while len(negs) < n_neg and tries < 100:
                tries += 1
                nl, nn = random.choice(all_keys)
                if (nl, nn) == (layer, nid):
                    continue

                ntid, npos, _nval, nrank, nex = random.choice(
                    rows_by_neuron[(nl, nn)][:k_skip])
                if npos in pos_positions:
                    continue

                # ➕ INSERT THIS SAFETY CHECK:
                neg_ctx = ctx(nl, nn, ntid, npos, nrank, nex)
                if neg_ctx is not None:
                    negs.append(neg_ctx)

            if len(negs) < n_neg:
                continue  # Skip this positive example — not enough negatives

            # ➕ INSERT THIS SAFETY CHECK:
            pos_ctx = ctx(layer, nid, tid, pos, rank, ex_idx)
            if pos_ctx is None:
                invalid_examples += 1
                continue  # Skip if we can't build the positive context

            sents = negs + [pos_ctx]
            random.shuffle(sents)

            dataset.append({
                "layer": layer,
                "neuron": nid,
                "sentences": sents,
                "answer": sents.index(pos_ctx) + 1,  # 1-based index
                "interpretation": lora_info[layer][str(nid)]["interpretation"],
            })

    print(f'Skipped {invalid_examples} invalid examples')
    return dataset


def autointerp_build_prompt(item):
    lines = [
        "You are given **five** independent sentences. Exactly **one** of them matches the neuron interpretation below.",
        "Respond **only** with its number (1-5).",
        "",
        f"Neuron interpretation: {item['interpretation']}",
        "",
    ]
    for i, s in enumerate(item['sentences'], 1):
        lines.append(f"{i}. {autointerp_clean(s)}")
    return "\n".join(lines)


def autointerp_evaluate(preds, ds):
    acc = sum(p == ex['answer'] for p, ex in zip(preds, ds))/len(ds)
    return {k: acc for k in ("accuracy", "precision", "recall")}


def autointerp_extract_digit(text: str) -> int:
    """Return first digit 1‑5 found in *text* (or 0 if none)."""
    m = re.compile(r"[1-5]").search(text)
    return int(m.group()) if m else 0


"""
Metrics eval helper functions
"""


def build_metrics_eval_messages(question: str, reply_a: str, reply_b: str) -> List[Dict]:
    user = (
        f"{question}\n\n"
        f"### Reply A:\n{reply_a}\n\n"
        f"### Reply B:\n{reply_b}\n\n"
        "Which reply is better? Answer with A or B only."
    )
    return [
        {"role": "user",   "content": f"{question}\n\n"},
        # The assistant role is left blank; the tokenizer adds the tag.
    ]


def preprocess_to_perspective_message(example):
    return {
        "messages": [{
            "role": "user",
            "content": "Complete the following sentence: " + example["prompt"]["text"]
        }]
    }


def analyze_text_toxicity_eval(text, requested_attributes, client):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': requested_attributes,
        'languages': ['en']
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return response


Message = Dict[str, str]  # {"role": "user"|"assistant", "content": str}


def setup_tokenizer_for_chat(tokenizer):
    """Setup tokenizer with proper chat template and padding token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load chat template from IT version if not present
    if tokenizer.chat_template is None:
        from transformers import AutoTokenizer
        it_tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-2b-it", use_fast=True)
        if it_tokenizer.chat_template is not None:
            tokenizer.chat_template = it_tokenizer.chat_template
            print("Loaded chat template from google/gemma-2-2b-it")
        else:
            raise Exception("Could not load IT tokenizer chat template")

    return tokenizer


def _make_cache_key(
    tokenizer_name: str,
    max_length: int,
    datasets_to_use: Tuple[str, ...],
    eval_holdout_ratio: float,
    seed: int,
    pack_sequences: bool,
) -> str:
    """Create a deterministic cache key based on dataset parameters."""
    key_dict = {
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "datasets_to_use": sorted(datasets_to_use),  # Sort for consistency
        "eval_holdout_ratio": eval_holdout_ratio,
        "seed": seed,
        "pack_sequences": pack_sequences,
        "version": "v1",  # Increment this if data processing changes
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> str:
    """Get the cache file path for a given cache key."""
    cache_dir = os.path.join(os.path.expanduser(
        "~"), ".cache", "topk_lora_datasets")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"sft_datasets_{cache_key}.pkl")


def _save_datasets_to_cache(train_ds: HFDataset, eval_ds: HFDataset, cache_path: str) -> None:
    """Save datasets to cache file efficiently with disk-based format for streaming."""
    print(f"💾 Saving datasets to cache: {cache_path}")

    # Save in multiple formats for flexibility
    cache_data = {
        "train": {
            "data": train_ds.to_list(),
            "features": train_ds.features,
        },
        "eval": {
            "data": eval_ds.to_list(),
            "features": eval_ds.features,
        }
    }

    # Save pickle format (legacy compatibility)
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save disk format for efficient streaming
    train_cache_dir = cache_path.replace('.pkl', '_train_dir')
    eval_cache_dir = cache_path.replace('.pkl', '_eval_dir')

    try:
        train_ds.save_to_disk(train_cache_dir)
        eval_ds.save_to_disk(eval_cache_dir)
        print(f"💾 Also saved disk format for efficient streaming")
    except Exception as e:
        print(f"⚠️ Failed to save disk format: {e}")

    print(
        f"✅ Datasets cached successfully ({len(train_ds)} train, {len(eval_ds)} eval)")


def _load_datasets_from_cache(cache_path: str, streaming: bool = True) -> Tuple[HFDataset, HFDataset]:
    """Load datasets from cache file with true streaming support."""
    print(f"📁 Loading datasets from cache: {cache_path}")

    # Check if we have disk-based cache directories (best for streaming)
    train_cache_dir = cache_path.replace('.pkl', '_train_dir')
    eval_cache_dir = cache_path.replace('.pkl', '_eval_dir')

    if os.path.exists(train_cache_dir) and os.path.exists(eval_cache_dir):
        print(f"🔄 Loading datasets from disk cache with streaming={streaming}")
        from datasets import load_from_disk
        train_ds = load_from_disk(train_cache_dir)
        eval_ds = load_from_disk(eval_cache_dir)

        if streaming:
            # Only convert training dataset to streaming - keep eval as regular dataset
            # This avoids issues with evaluation loops that expect finite datasets
            train_ds = train_ds.to_iterable_dataset()
            print(
                f"✅ Training dataset streaming enabled, eval dataset kept as regular dataset")
        else:
            print(
                f"✅ Datasets loaded from disk cache ({len(train_ds)} train, {len(eval_ds)} eval)")
        return train_ds, eval_ds

    # Check if we have Arrow format files (fallback)
    train_cache_path = cache_path.replace('.pkl', '_train.arrow')
    eval_cache_path = cache_path.replace('.pkl', '_eval.arrow')

    if os.path.exists(train_cache_path) and os.path.exists(eval_cache_path):
        # Load from Arrow format - still loads to memory but more efficient
        train_ds = HFDataset.from_file(train_cache_path)
        eval_ds = HFDataset.from_file(eval_cache_path)

        if streaming:
            # Only convert training dataset to streaming
            train_ds = train_ds.to_iterable_dataset()
            print(
                f"✅ Training dataset streaming enabled from Arrow cache, eval dataset regular")
        else:
            print(
                f"✅ Datasets loaded from Arrow cache ({len(train_ds)} train, {len(eval_ds)} eval)")
        return train_ds, eval_ds

    # Fallback to pickle format (legacy)
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    train_ds = HFDataset.from_list(
        cache_data["train"]["data"], features=cache_data["train"]["features"])
    eval_ds = HFDataset.from_list(
        cache_data["eval"]["data"], features=cache_data["eval"]["features"])

    # Save in disk format for future efficient streaming
    try:
        train_ds.save_to_disk(train_cache_dir)
        eval_ds.save_to_disk(eval_cache_dir)
        print(f"💾 Converted cache to disk format for future streaming")
    except Exception as e:
        print(f"⚠️ Failed to convert to disk format: {e}")

    if streaming:
        # Only convert training dataset to streaming
        train_ds = train_ds.to_iterable_dataset()
        print(
            f"✅ Training dataset streaming enabled from pickle cache, eval dataset regular")
    else:
        print(
            f"✅ Datasets loaded from pickle cache ({len(train_ds)} train, {len(eval_ds)} eval)")
    return train_ds, eval_ds


def _ensure_roles(messages: List[Message]) -> List[Message]:
    """Ensure roles are valid and alternating for Gemma chat template."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "assistant" if role == "system" else "user"
        out.append({"role": role, "content": m.get("content", "")})

    # Ensure alternating user/assistant pattern required by Gemma-IT
    if not out:
        return out

    # Fix alternation: must start with user and alternate
    cleaned = []
    expected_role = "user"

    for msg in out:
        if msg["role"] == expected_role:
            cleaned.append(msg)
            expected_role = "assistant" if expected_role == "user" else "user"
        elif expected_role == "assistant" and msg["role"] == "assistant":
            # This is good, add it
            cleaned.append(msg)
            expected_role = "user"
        # Skip messages that break the alternating pattern

    # Ensure we end with an assistant message for training
    if cleaned and cleaned[-1]["role"] == "user":
        # Remove the last user message if there's no assistant response
        cleaned = cleaned[:-1]

    return cleaned


def _encode_with_assistant_mask(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Message],
    max_length: int,
) -> Dict[str, List[int]]:
    """
    Apply Gemma chat template and create labels with -100 on non-assistant tokens.
    """
    messages = _ensure_roles(messages)
    if not messages:
        # Return empty sequence for empty messages
        return {"input_ids": [], "labels": [], "attention_mask": []}

    full_ids: List[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    labels = [-100] * len(full_ids)

    # Mark assistant spans
    for j, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        # Handle empty prefix case
        prefix_ids: List[int] = []
        if j > 0:
            prefix_ids = tokenizer.apply_chat_template(
                messages[:j], tokenize=True, add_generation_prompt=True
            )
        upto_ids: List[int] = tokenizer.apply_chat_template(
            messages[: j + 1], tokenize=True, add_generation_prompt=False
        )
        start = len(prefix_ids)
        end = min(len(upto_ids), len(full_ids))
        for t in range(start, end):
            labels[t] = full_ids[t]

    # Left trim to max_length (keep rightmost)
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    attn_mask = [1] * len(full_ids)
    return {"input_ids": full_ids, "labels": labels, "attention_mask": attn_mask}


def load_dolly(split: str = "train") -> HFDataset:
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)

    def to_messages(ex):
        instr = (ex.get("instruction") or "").strip()
        ctx = (ex.get("context") or "").strip()
        user = instr if not ctx else f"{instr}\n\nContext:\n{ctx}"
        resp = (ex.get("response") or "").strip()
        return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": resp}]}
    # type: ignore
    return ds.map(to_messages, remove_columns=[c for c in ds.column_names if c != "messages"])


def load_ultrachat(split: str = "train_sft") -> HFDataset:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    def clean(ex):
        msgs = ex.get("messages") or []
        cleaned = []
        for m in msgs:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (m.get("content") or "").strip()
            if content:
                cleaned.append({"role": role, "content": content})
        has_assistant = any(m["role"] == "assistant" for m in cleaned)
        return {"messages": cleaned if has_assistant else None}
    ds = ds.map(clean)
    ds = ds.filter(lambda ex: ex["messages"] is not None)
    return ds  # type: ignore


def load_oasst1(split: str = "train") -> HFDataset:
    ds = load_dataset("OpenAssistant/oasst1", split=split)

    def keep(ex):
        if ex.get("deleted", False):
            return False
        if ex.get("lang") not in (None, "en"):
            return False
        role = ex.get("role")
        if role not in ("prompter", "assistant"):
            return False
        txt = ex.get("text") or ""
        return len(txt.strip()) > 0
    ds = ds.filter(keep)
    rows = ds.to_list()
    by_id = {r["message_id"]: r for r in rows}
    from collections import defaultdict
    children = defaultdict(list)
    for r in rows:
        pid = r.get("parent_id")
        if pid in by_id:
            children[pid].append(r["message_id"])
    leaves = [mid for mid in by_id.keys() if len(children.get(mid, [])) == 0]
    conversations: List[Dict[str, List[Message]]] = []
    for leaf in leaves:
        path = []
        cur = leaf
        seen = set()
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            path.append(by_id[cur])
            cur = by_id[cur].get("parent_id")
        path.reverse()
        msgs: List[Message] = []
        for node in path:
            role = node.get("role")
            r = "user" if role == "prompter" else (
                "assistant" if role == "assistant" else None)
            if r is None:
                continue
            content = (node.get("text") or "").strip()
            if content:
                msgs.append({"role": r, "content": content})
        if any(m["role"] == "assistant" for m in msgs):
            conversations.append({"messages": msgs})
    return HFDataset.from_list(conversations)


def build_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 8192,
    datasets_to_use: Tuple[str, ...] = ("oasst1", "dolly", "ultrachat"),
) -> HFDataset:
    pieces = []
    if "oasst1" in datasets_to_use:
        pieces.append(load_oasst1("train"))
    if "dolly" in datasets_to_use:
        pieces.append(load_dolly("train"))
    if "ultrachat" in datasets_to_use:
        pieces.append(load_ultrachat("train_sft"))
    if not pieces:
        raise ValueError("No datasets selected.")
    mixed = concatenate_datasets(pieces)

    # Filter out empty conversations
    def is_valid(ex):
        messages = ex.get("messages", [])
        if not messages:
            return False
        # Must have at least one assistant message
        has_assistant = any(
            m.get("role") == "assistant" for m in messages if m.get("content", "").strip())
        return has_assistant

    mixed = mixed.filter(is_valid)

    def tokenize_example(ex):
        messages: List[Message] = ex["messages"]
        return _encode_with_assistant_mask(tokenizer, messages, max_length)
    tokenized = mixed.map(tokenize_example, remove_columns=[
                          c for c in mixed.column_names if c != "messages"])

    # Filter out sequences that became empty after tokenization
    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)
    return tokenized


def pack_tokenized_dataset(
    tokenized: HFDataset,
    *,
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> HFDataset:
    """Pack multiple tokenized examples into <=max_length sequences. Insert EOS between examples and set its label to -100."""
    buffers = {"input_ids": [], "labels": [], "attention_mask": []}
    packed = []

    def flush():
        if not buffers["input_ids"]:
            return
        packed.append({k: v[:] for k, v in buffers.items()})
        for k in buffers:
            buffers[k].clear()
    cur_len = 0
    for ex in tokenized:
        ids = list(ex["input_ids"])
        labs = list(ex["labels"])
        attn = list(ex["attention_mask"])
        need = len(ids) + (1 if cur_len > 0 else 0)
        if cur_len + need > max_length:
            flush()
            cur_len = 0
        if cur_len > 0:
            buffers["input_ids"].append(eos_token_id)
            buffers["labels"].append(-100)
            buffers["attention_mask"].append(1)
            cur_len += 1
        if len(ids) > max_length:
            ids, labs, attn = ids[-max_length:], labs[-max_length:], attn[-max_length:]
        buffers["input_ids"].extend(ids)
        buffers["labels"].extend(labs)
        buffers["attention_mask"].extend(attn)
        cur_len += len(ids)
        if cur_len >= max_length:
            flush()
            cur_len = 0
    flush()
    return HFDataset.from_list(packed)


def build_sft_datasets(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 8192,
    datasets_to_use: Tuple[str, ...] = ("oasst1", "dolly", "ultrachat"),
    eval_holdout_ratio: float = 0.01,
    seed: int = 42,
    pack_sequences: bool = True,
    use_cache: bool = True,
    streaming: bool = True,  # New parameter for streaming
) -> Tuple[HFDataset, HFDataset]:
    """
    Build SFT datasets with caching and streaming support.

    Args:
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        datasets_to_use: Tuple of dataset names to include
        eval_holdout_ratio: Fraction of data to use for evaluation
        seed: Random seed for train/eval split
        pack_sequences: Whether to pack sequences together
        use_cache: Whether to use cached datasets if available
        streaming: Whether to return streaming datasets (memory efficient)

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Create cache key based on all parameters that affect the output
    tokenizer_name = getattr(tokenizer, 'name_or_path', 'unknown_tokenizer')
    cache_key = _make_cache_key(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        datasets_to_use=datasets_to_use,
        eval_holdout_ratio=eval_holdout_ratio,
        seed=seed,
        pack_sequences=pack_sequences,
    )

    cache_path = _get_cache_path(cache_key)

    # Try to load from cache first
    if use_cache and os.path.exists(cache_path):
        try:
            return _load_datasets_from_cache(cache_path, streaming=streaming)
        except Exception as e:
            print(
                f"⚠️ Failed to load from cache ({e}), rebuilding datasets...")
            # Remove corrupted cache file
            # try:
            #     os.remove(cache_path)
            # except:
            #     pass

    # Build datasets from scratch
    print("🔨 Building datasets from scratch (this may take a while)...")
    full = build_sft_dataset(
        tokenizer, max_length=max_length, datasets_to_use=datasets_to_use)
    split = full.train_test_split(test_size=eval_holdout_ratio, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    # Safety check: ensure eval dataset is not empty
    if len(eval_ds) == 0:
        print("⚠️ Evaluation dataset is empty, adjusting holdout ratio...")
        # Use a minimum of 10 examples for evaluation or 1% of data, whichever is larger
        min_eval_size = max(10, int(len(full) * 0.01))
        adjusted_ratio = min(min_eval_size / len(full), 0.1)  # Cap at 10%
        split = full.train_test_split(test_size=adjusted_ratio, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]
        print(
            f"📊 Adjusted eval dataset size: {len(eval_ds)} examples ({adjusted_ratio:.3f} ratio)")

    if pack_sequences:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        eos_id = tokenizer.eos_token_id or pad_id
        print("📦 Packing training sequences...")
        train_ds = pack_tokenized_dataset(
            train_ds, max_length=max_length, pad_token_id=pad_id, eos_token_id=eos_id)
        print("📦 Packing evaluation sequences...")
        eval_ds = pack_tokenized_dataset(
            eval_ds, max_length=max_length, pad_token_id=pad_id, eos_token_id=eos_id)

    # Save to cache for future use
    if use_cache:
        try:
            _save_datasets_to_cache(train_ds, eval_ds, cache_path)
        except Exception as e:
            print(
                f"⚠️ Failed to save to cache ({e}), continuing without caching...")

    # Convert to streaming if requested (only for training dataset)
    if streaming:
        train_ds = train_ds.to_iterable_dataset()
        # Keep eval_ds as regular dataset for compatibility with evaluation loops
        print(
            f"✅ Training dataset converted to streaming format, eval dataset kept regular")

    return train_ds, eval_ds


def normalize_ultrachat_messages(msgs):
    """
    Preserve content while enforcing alternation:
      • keep only non-empty user/assistant turns
      • merge consecutive same-role turns (concat with blank line)
      • if first turn is assistant → prepend a minimal user stub ' '
      • if last turn is user → we'll set add_generation_prompt=True (no deletion)
    Returns (fixed_messages, add_gen_prompt)
    """
    # 1) keep only user/assistant with text
    cleaned = [{"role": m["role"], "content": (m.get("content") or "").strip()}
               for m in (msgs or [])
               if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]

    if not cleaned:
        # fallback: single empty user so the template is satisfiable
        return [{"role": "user", "content": " "}], True

    # 2) merge consecutive same-role
    merged = []
    for m in cleaned:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(m)

    # 3) ensure we start with user
    if merged[0]["role"] != "user":
        merged = [{"role": "user", "content": " "}] + merged

    # 4) enforce alternation (after merging it’s rare to fail; still be safe)
    alternated = [merged[0]]
    expect = "assistant"
    for m in merged[1:]:
        if m["role"] == expect:
            alternated.append(m)
            expect = "user" if expect == "assistant" else "assistant"
        else:
            # If out-of-order, insert a stub to keep content (no drops)
            alternated.append({"role": expect, "content": " "})
            expect = "user" if expect == "assistant" else "assistant"
            if m["role"] == expect:
                alternated.append(m)
                expect = "assistant" if expect == "user" else "user"

    # 5) generation prompt if last is user (don’t delete their turn)
    add_gen = alternated[-1]["role"] == "user"
    return alternated, add_gen

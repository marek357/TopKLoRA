import json
import random
import re
from typing import Dict, Any, List, Mapping, Optional, Tuple
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from openai import OpenAI

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
    merged_output_dir: Optional[str] = None,
    save_merged_model: bool = False,
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
    print(example)
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
        # if center >= len(toks):
        #     print(center, tok_pos, start)
        #     print(center, tok_pos, start)
        #     print(len(toks))
        #     print(len(toks))
        #     assert False
        # print(tok_pos, start, center, len(toks))
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
    model: str = "gpt-4o-mini",
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
    # s, e = max(0, pos-window), min(ids.size(0), pos+window+1)
    # toks = [
    #     tokenizer.decode([tid], skip_special_tokens=False)
    #     for tid in ids[s:e]
    # ]
    # print(len(toks), pos-s, pos, s)
    # center_idx = pos - s
    # # if 0 <= center_idx < len(toks):
    # #     toks[center_idx] = f"<<{toks[center_idx]}>>"

    # toks[pos-s] = f"<<{toks[pos-s]}>>"
    # return "".join(toks).replace("\n", "⏎")
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

    # for (layer, nid), rows in rows_by_neuron.items():
    #     if len(rows) <= k_skip:
    #         continue
    #     pos_rows = rows[k_skip:]
    #     positives = random.choices(pos_rows, k=n_examples)
    #     pos_positions = {p for _tid, p, _v, _r, _e in rows}

    #     for tid, pos, _v, rank, ex_idx in positives:
    #         negs, tries = [], 0
    #         while len(negs) < n_neg and tries < 100:
    #             tries += 1
    #             nl, nn = random.choice(all_keys)
    #             if (nl, nn) == (layer, nid):
    #                 continue
    #             ntid, npos, _nval, nrank, nex = random.choice(
    #                 rows_by_neuron[(nl, nn)][:k_skip])
    #             if npos in pos_positions:
    #                 continue
    #             negs.append(ctx(nl, nn, ntid, npos, nrank, nex))
    #         if len(negs) < n_neg:
    #             continue
    #         pos_ctx = ctx(layer, nid, tid, pos, rank, ex_idx)
    #         sents = negs+[pos_ctx]
    #         random.shuffle(sents)
    #         # print(lora_info[layer].keys())
    #         dataset.append({
    #             "layer": layer, "neuron": nid,
    #             "sentences": sents,
    #             "answer": sents.index(pos_ctx)+1,
    #             "interpretation": lora_info[layer][str(nid)]["interpretation"],
    #         })
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
        {"role": "user",   "content": user},
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

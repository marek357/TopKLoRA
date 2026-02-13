"""
Backend for the TopKLoRA dashboard.

Module-level singleton holds the loaded model (not JSON-serializable,
too large for gr.State).  All heavy lifting lives here; the Gradio UI
in app.py only calls these functions.
"""

import glob
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.utils import wrap_topk_lora_modules
from src.models import _hard_topk_mask
from src.steering import FeatureSteeringContext, list_available_adapters

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
CACHE_DIR = Path("delphi_cache")


# ---------------------------------------------------------------------------
# Singleton model state
# ---------------------------------------------------------------------------
class ModelState:
    model = None
    tokenizer = None
    wrapped_modules: dict = {}
    device: str = "cpu"
    # Cache for visualization (avoid re-computing activations)
    viz_cache: dict = {
        "text": None,
        "hookpoint": None,
        "tokens": None,
        "activations": None,  # [seq_len, r]
    }


state = ModelState()


# ---------------------------------------------------------------------------
# Local model / adapter discovery
# ---------------------------------------------------------------------------
def discover_base_models() -> list[str]:
    """Return paths to local base models (directories containing config.json
    directly under ``models/``, excluding adapter-only dirs).

    Uses ``glob.glob`` instead of ``Path.iterdir`` to follow symlinks.
    """
    if not MODELS_DIR.is_dir():
        return []
    return sorted(
        str(Path(p).parent) for p in glob.glob(str(MODELS_DIR / "*/config.json"))
    )


def discover_adapters() -> list[dict]:
    """Walk ``models/`` and find every adapter checkpoint.

    An adapter directory is identified by containing ``adapter_config.json``.
    Returns a list of dicts with keys:
        path, display_name, r, k_final (from the directory structure or config).
    """
    if not MODELS_DIR.is_dir():
        return []

    results = []
    for adapter_cfg_path in sorted(
        Path(p)
        for p in glob.glob(str(MODELS_DIR / "**/adapter_config.json"), recursive=True)
    ):
        adapter_dir = adapter_cfg_path.parent
        # Build a human-friendly display name from relative path
        rel = adapter_dir.relative_to(MODELS_DIR)
        display = str(rel)

        # Read r from adapter_config.json
        try:
            cfg = json.loads(adapter_cfg_path.read_text())
            r = cfg.get("r", None)
        except Exception:
            logger.warning("Failed to read %s", adapter_cfg_path, exc_info=True)
            r = None

        # Try to read k_final from hparams.json (more accurate than dir name)
        k_final = None
        hparams_path = adapter_dir / "hparams.json"
        if hparams_path.exists():
            try:
                hp = json.loads(hparams_path.read_text())
                k_final = hp.get("lora_topk", {}).get("k_final", None)
            except Exception:
                logger.warning("Failed to read %s", hparams_path, exc_info=True)

        # Fallback: parse k from directory path (e.g. "dpo/mlp/k64/...")
        if k_final is None:
            for part in rel.parts:
                if part.startswith("k") and part[1:].isdigit():
                    k_final = int(part[1:])
                    break

        results.append(
            {
                "path": str(adapter_dir),
                "display": display,
                "r": r,
                "k_final": k_final,
            }
        )
    return results


def discover_cached_adapters() -> list[str]:
    """Return paths to adapters with cached activations in delphi_cache/."""
    if not CACHE_DIR.is_dir():
        return []
    return sorted(
        Path(p).name
        for p in glob.glob(str(CACHE_DIR / "*/"))
        if not Path(p).name.startswith(".")
    )


def get_cached_hookpoints(adapter_name: str) -> list[str]:
    """Return hookpoint names for a cached adapter by reading config.json files."""
    adapter_dir = CACHE_DIR / adapter_name
    if not adapter_dir.is_dir():
        return []

    hookpoints = []
    for child in sorted(Path(p) for p in glob.glob(str(adapter_dir / "*/"))):
        if not child.name.startswith(".") and child.name != "stats":
            config_path = child / "config.json"
            if config_path.exists():
                try:
                    # Verify it's valid JSON
                    config = json.loads(config_path.read_text())
                    if isinstance(config, dict):
                        hookpoints.append(child.name)
                except Exception:
                    logger.warning("Failed to read %s", config_path, exc_info=True)
    return hookpoints


def get_cached_hookpoint_config(adapter_name: str, hookpoint: str) -> dict:
    """Read config.json for a cached hookpoint."""
    config_path = CACHE_DIR / adapter_name / hookpoint / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except Exception:
        logger.warning("Failed to read %s", config_path, exc_info=True)
        return {}


def get_cached_latent_choices(
    adapter_name: str, hookpoint: str
) -> list[tuple[str, int]]:
    """Get list of (display_name, latent_id) tuples for a hookpoint with p_active from stats.

    Returns:
        List of tuples: [("Latent 0 (p_active=0.100)", 0), ...]
    """
    stats_path = CACHE_DIR / adapter_name / "stats" / "latent_stats.jsonl"
    offsets_path = CACHE_DIR / adapter_name / "stats" / "hookpoint_offsets.json"

    if not stats_path.exists() or not offsets_path.exists():
        return []

    try:
        # Load hookpoint offsets to determine latent_id range
        offsets_data = json.loads(offsets_path.read_text())
        if hookpoint not in offsets_data["offsets"]:
            return []

        start_id = offsets_data["offsets"][hookpoint]
        width = offsets_data["widths"][hookpoint]
        end_id = start_id + width

        # Read latent stats and filter by latent_id range
        choices = []
        with open(stats_path, "r") as f:
            for line in f:
                stat = json.loads(line)
                latent_id = stat["latent_id"]
                if start_id <= latent_id < end_id:
                    p_active = stat["p_active"]
                    # Convert to local index (relative to this hookpoint)
                    local_idx = latent_id - start_id
                    display = f"Latent {local_idx} (p_active={p_active:.3f})"
                    choices.append((display, local_idx))

        return choices
    except Exception:
        logger.warning(
            "Failed to load latent choices for %s/%s",
            adapter_name,
            hookpoint,
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _infer_k(adapter_path: str) -> int:
    """Infer k_final from the adapter's metadata.

    Checks (in order):
      1. ``hparams.json`` → ``lora_topk.k_final``
      2. Directory path components matching ``k<digits>``
    """
    adapter_dir = Path(adapter_path)

    # 1. hparams.json
    hparams_path = adapter_dir / "hparams.json"
    if hparams_path.exists():
        try:
            hp = json.loads(hparams_path.read_text())
            k = hp.get("lora_topk", {}).get("k_final")
            if k is not None:
                return int(k)
        except Exception:
            logger.warning("Failed to read %s", hparams_path, exc_info=True)

    # 2. Directory name convention (e.g. ".../k64/...")
    for part in adapter_dir.parts:
        if part.startswith("k") and part[1:].isdigit():
            return int(part[1:])

    raise Exception("Couldn't infer k_final")


def load_model(
    base_model_path: str,
    adapter_path: str,
    device_str: str,
) -> tuple[str, str]:
    """Load base model + LoRA adapter and wrap with TopK.

    ``k`` is inferred automatically from the adapter's metadata.
    Mirrors ``src/evals.py:init_model_tokenizer_fixed``.
    Returns (status_message, hookpoint_config_html).
    """
    try:
        k = _infer_k(adapter_path)

        # 1. Tokenizer from the adapter checkpoint (has chat template)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)

        # 2. Base model on CPU first
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype="auto", device_map="cpu"
        )

        # 3. Load PEFT adapter
        model = PeftModel.from_pretrained(
            model, adapter_path, device_map="cpu", use_safetensors=True
        )

        # 4. Wrap LoRA layers with TopK
        replaced, wrapped_modules = wrap_topk_lora_modules(
            model,
            k=k,
            temperature=0.0,
            temperature_schedule="constant",
            k_schedule="constant",
            k_final=k,
            temperature_final=0.0,
            is_topk_experiment=True,
            set_train=False,
        )

        # 5. Move to target device and eval
        model.to(device_str)
        model.eval()

        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Store in singleton
        state.model = model
        state.tokenizer = tokenizer
        state.wrapped_modules = wrapped_modules
        state.device = device_str

        # Build status message
        r_val = list(wrapped_modules.values())[0].r if wrapped_modules else "N/A"
        status = (
            f"✅ Loaded successfully on {device_str} | "
            f"dtype: {model.dtype} | r={r_val}, k={k} | "
            f"{replaced} hookpoints"
        )

        # Build hookpoint configuration HTML
        hookpoint_items = "".join(
            f"<li><code>{name}</code></li>" for name in sorted(wrapped_modules.keys())
        )
        hookpoint_html = (
            f"<div style='font-family: system-ui; padding: 12px;'>"
            f"<h3 style='margin-top: 0;'>Available Hookpoints ({replaced})</h3>"
            f"<ul style='line-height: 1.6; margin: 0; padding-left: 20px;'>"
            f"{hookpoint_items}"
            f"</ul>"
            f"</div>"
        )

        return status, hookpoint_html

    except Exception as exc:
        logger.error("Error loading model: %s", exc, exc_info=True)
        return f"❌ Error loading model: {exc}", ""


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------
def get_adapter_choices() -> list[str]:
    """Return hookpoint names for all TopKLoRALinearSTE modules."""
    if state.model is None:
        return []
    info = list_available_adapters(state.model, verbose=False)
    return list(info.keys())


def get_adapter_info(hookpoint: str) -> dict:
    """Return ``{"r": int, "k": int}`` for *hookpoint*."""
    if hookpoint in state.wrapped_modules:
        m = state.wrapped_modules[hookpoint]
        return {"r": m.r, "k": m.topk.k}
    return {"r": 0, "k": 0}


# ---------------------------------------------------------------------------
# Activation visualisation
# ---------------------------------------------------------------------------
def _render_activation_html(
    tokens: list, activations: torch.Tensor, hookpoint: str, latent_idx: int
) -> str:
    """Render activation heatmap HTML from pre-computed activations.

    Args:
        tokens: List of token strings
        activations: [seq_len] tensor of activation values for one latent
        hookpoint: Hookpoint name (for display)
        latent_idx: Latent index (for display)
    """
    # Validate inputs
    if len(tokens) == 0:
        return "<p>⚠️ No tokens to display.</p>"

    if activations is None or activations.numel() == 0:
        return "<p>⚠️ No activations available for this latent.</p>"

    acts = activations.float().cpu()

    # Check for mismatched lengths
    if len(acts) != len(tokens):
        return (
            f"<p>⚠️ Length mismatch: {len(tokens)} tokens but {len(acts)} activations. "
            "This may indicate a model/tokenizer issue.</p>"
        )

    # Compute statistics
    a_min = acts.min().item()
    a_max = acts.max().item()
    a_mean = acts.mean().item()

    # Check if latent is effectively dead (all near-zero)
    if abs(a_max) < 1e-6 and abs(a_min) < 1e-6:
        # Render as gray with warning
        spans = []
        for tok in tokens:
            display_tok = (
                tok.replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
            )
            spans.append(
                f'<span class="token-span" data-activation="0.0000 (dead)" '
                f'style="background:#f0f0f0;padding:2px 4px;margin:1px;'
                f'border-radius:3px;display:inline-block;font-family:monospace;color:#999;position:relative;cursor:help;">'
                f"{display_tok}</span>"
            )

        html = (
            "<style>"
            ".token-span::after { content: attr(data-activation); position: absolute; bottom: 100%; left: 50%; "
            "transform: translateX(-50%); background: #1f2937; color: #fff; padding: 4px 8px; border-radius: 4px; "
            "font-size: 11px; white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 0.2s; "
            "margin-bottom: 4px; z-index: 1000; }"
            ".token-span:hover::after { opacity: 1; }"
            '.token-span::before { content: ""; position: absolute; bottom: 100%; left: 50%; '
            "transform: translateX(-50%); border: 4px solid transparent; border-top-color: #1f2937; "
            "opacity: 0; pointer-events: none; transition: opacity 0.2s; z-index: 1000; }"
            ".token-span:hover::before { opacity: 1; }"
            "</style>"
            '<div style="line-height:2.2;padding:8px;">'
            + " ".join(spans)
            + "</div>"
            + f"<p style='margin-top:8px;font-size:0.85em;color:#f59e0b;'>"
            f"⚠️ <strong>Dead latent</strong> — all activations ≈ 0 | hookpoint={hookpoint} | latent={latent_idx}</p>"
        )
        return html

    # Normalise to [0, 1] for colouring
    span = max(a_max - a_min, 1e-8)
    normed = ((acts - a_min) / span).tolist()

    # Build HTML spans
    spans = []
    for tok, val, raw in zip(tokens, normed, acts.tolist()):
        # white (0) → red (1)
        r_col = 255
        g_col = int(255 * (1 - val))
        b_col = int(255 * (1 - val))
        bg = f"rgb({r_col},{g_col},{b_col})"
        # Use white text for high intensity (dark backgrounds), dark text for low intensity
        text_color = "#fff" if val > 0.5 else "#1f2937"
        display_tok = (
            tok.replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
        )
        spans.append(
            f'<span class="token-span" data-activation="{raw:.4f}" '
            f'style="background:{bg};color:{text_color};padding:2px 4px;margin:1px;'
            f'border-radius:3px;display:inline-block;font-family:monospace;position:relative;cursor:help;">'
            f"{display_tok}</span>"
        )

    html = (
        "<style>"
        ".token-span::after { content: attr(data-activation); position: absolute; bottom: 100%; left: 50%; "
        "transform: translateX(-50%); background: #1f2937; color: #fff; padding: 4px 8px; border-radius: 4px; "
        "font-size: 11px; white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 0.2s; "
        "margin-bottom: 4px; z-index: 1000; }"
        ".token-span:hover::after { opacity: 1; }"
        '.token-span::before { content: ""; position: absolute; bottom: 100%; left: 50%; '
        "transform: translateX(-50%); border: 4px solid transparent; border-top-color: #1f2937; "
        "opacity: 0; pointer-events: none; transition: opacity 0.2s; z-index: 1000; }"
        ".token-span:hover::before { opacity: 1; }"
        "</style>"
        '<div style="line-height:2.2;padding:8px;">'
        + " ".join(spans)
        + "</div>"
        + f"<p style='margin-top:8px;font-size:0.85em;color:#666;'>"
        f"min={a_min:.4f}, max={a_max:.4f}, mean={a_mean:.4f} | hookpoint={hookpoint} | latent={latent_idx}</p>"
    )
    return html


def compute_token_activations(
    text: str,
    hookpoint: str,
    latent_idx: int,
) -> str:
    """Run a forward pass and cache activations for ALL latents, then render
    the heatmap for the selected latent.

    Subsequent calls to ``render_latent_from_cache`` can instantly switch
    latents without re-computing.
    """
    if state.model is None:
        return "<p>No model loaded.</p>"
    if hookpoint not in state.wrapped_modules:
        return f"<p>Hookpoint <code>{hookpoint}</code> not found.</p>"

    module = state.wrapped_modules[hookpoint]
    if latent_idx < 0 or latent_idx >= module.r:
        return f"<p>Latent index must be in [0, {module.r - 1}].</p>"

    # Check cache: if text + hookpoint match, skip forward pass
    if (
        state.viz_cache["text"] == text
        and state.viz_cache["hookpoint"] == hookpoint
        and state.viz_cache["activations"] is not None
    ):
        # Use cached activations
        return _render_activation_html(
            state.viz_cache["tokens"],
            state.viz_cache["activations"][:, latent_idx],
            hookpoint,
            latent_idx,
        )

    # Run forward pass and cache
    # Use tokenize=True to get correct token IDs directly (avoids double BOS)
    messages = [{"role": "user", "content": text}]
    input_ids = state.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = (
            torch.tensor([input_ids])
            if isinstance(input_ids, list)
            else input_ids["input_ids"]
        )
    input_ids = input_ids.to(state.device)
    enc = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    tokens = state.tokenizer.convert_ids_to_tokens(input_ids[0])

    # _last_z stores pre-TopK activations (models.py:292); we apply the
    # hard TopK mask here so the dashboard only shows the k active latents.
    with torch.no_grad():
        state.model(**enc)

    z = module._last_z  # [batch, seq_len, r]  (pre-TopK)
    if z is None:
        return "<p>No activations captured (try a different hookpoint).</p>"

    # Apply hard TopK mask so only the k active latents are non-zero
    k = module.topk.k
    z_masked = z * _hard_topk_mask(z, k)  # [batch, seq_len, r]

    # Cache the full activation tensor [seq_len, r]
    state.viz_cache["text"] = text
    state.viz_cache["hookpoint"] = hookpoint
    state.viz_cache["tokens"] = tokens
    state.viz_cache["activations"] = z_masked[0].cpu()  # [seq_len, r]

    # Render for the selected latent
    return _render_activation_html(
        tokens, z_masked[0, :, latent_idx], hookpoint, latent_idx
    )


def render_latent_from_cache(latent_idx: int) -> str:
    """Render activation heatmap from cached data (instant, no forward pass).

    Returns error message if cache is empty or latent index is out of bounds.
    """
    if state.viz_cache["activations"] is None:
        return "<p>No cached activations. Click 'Visualize' first.</p>"

    tokens = state.viz_cache["tokens"]
    activations = state.viz_cache["activations"]  # [seq_len, r]
    hookpoint = state.viz_cache["hookpoint"]

    if latent_idx < 0 or latent_idx >= activations.shape[1]:
        return f"<p>Latent index must be in [0, {activations.shape[1] - 1}].</p>"

    return _render_activation_html(
        tokens, activations[:, latent_idx], hookpoint, latent_idx
    )


# ---------------------------------------------------------------------------
# Cached activation explorer
# ---------------------------------------------------------------------------
def load_top_activating_examples(
    adapter_name: str,
    hookpoint: str,
    latent_idx: int,
    n_examples: int = 10,
) -> str:
    """Load and render top activating examples for a latent from delphi_cache.

    Returns HTML showing the top N examples with token-level heatmaps.
    """
    from safetensors import safe_open

    cache_dir = CACHE_DIR / adapter_name / hookpoint
    if not cache_dir.exists():
        return f"<p>Cache directory not found: {cache_dir}</p>"

    # Try to load tokenizer from the corresponding adapter
    # Adapter name pattern: mlp_k64_r8192_layer18 → models/dpo/mlp/k64/r8192/layer18
    tokenizer = None
    try:
        # Try to find corresponding adapter in models/dpo
        adapter_parts = adapter_name.split("_")
        # Expected pattern: {module_type}_k{k}_r{r}_layer{layer}
        if len(adapter_parts) >= 4:
            module_type = adapter_parts[0]  # e.g., "mlp"
            k_part = next((p for p in adapter_parts if p.startswith("k")), None)
            r_part = next(
                (
                    p
                    for p in adapter_parts
                    if p.startswith("r") and not p.startswith("reg")
                ),
                None,
            )
            layer_part = next((p for p in adapter_parts if p.startswith("layer")), None)

            if k_part and r_part and layer_part:
                # Check for optional reg mode directory (regon/regoff)
                reg_part = next(
                    (p for p in adapter_parts if p in ("regon", "regoff")),
                    None,
                )
                if reg_part:
                    adapter_path = (
                        MODELS_DIR
                        / "dpo"
                        / module_type
                        / k_part
                        / r_part
                        / reg_part
                        / layer_part
                    )
                else:
                    adapter_path = (
                        MODELS_DIR / "dpo" / module_type / k_part / r_part / layer_part
                    )
                if adapter_path.exists():
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(adapter_path), use_fast=True
                    )
    except Exception:
        logger.warning(
            "Failed to load tokenizer for cached adapter %s",
            adapter_name,
            exc_info=True,
        )

    # Find which safetensors file contains this latent
    safetensors_files = sorted(cache_dir.glob("*.safetensors"))
    if not safetensors_files:
        return f"<p>No safetensors files found in {cache_dir}</p>"

    # Determine which file contains the latent
    target_file = None
    for f in safetensors_files:
        parts = f.stem.split("_")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start, end = int(parts[0]), int(parts[1])
            if start <= latent_idx <= end:
                target_file = f
                break

    if target_file is None:
        return f"<p>No safetensors file contains latent {latent_idx}</p>"

    # Load the safetensors file
    try:
        with safe_open(target_file, framework="pt", device="cpu") as f:
            locations = f.get_tensor("locations")  # [n_activations, 3]
            activations_all = f.get_tensor("activations")  # [n_activations]
            tokens_all = f.get_tensor("tokens")  # [batch, sequence]
    except Exception as e:
        return f"<p>Error loading {target_file}: {e}</p>"

    # Convert locations to int64 to support boolean indexing
    locations = locations.to(torch.int64)

    # Filter to this specific latent (accounting for file offset)
    parts = target_file.stem.split("_")
    file_start = int(parts[0])
    local_latent_idx = latent_idx - file_start

    mask = locations[:, 2] == local_latent_idx
    if not mask.any():
        return f"<p>Latent {latent_idx} has no activations in the cache.</p>"

    latent_locs = locations[mask]  # [n_matches, 3]
    latent_acts = activations_all[mask]  # [n_matches]

    # Sort by activation (descending)
    sorted_indices = torch.argsort(latent_acts, descending=True)
    top_indices = sorted_indices[: n_examples * 3]  # Get more to account for duplicates

    # Group activations by batch_idx to avoid duplicate prompts
    from collections import defaultdict

    batch_activations = defaultdict(
        list
    )  # batch_idx -> [(seq_idx, activation_val), ...]

    for idx in top_indices:
        batch_idx, seq_idx, _ = latent_locs[idx].tolist()
        activation_val = latent_acts[idx].item()
        batch_activations[batch_idx].append((seq_idx, activation_val))

    # Sort batches by their maximum activation
    sorted_batches = sorted(
        batch_activations.items(),
        key=lambda x: max(act for _, act in x[1]),
        reverse=True,
    )[:n_examples]  # Take top N unique prompts

    # Build HTML for top examples
    tokenizer_status = (
        "✓ Tokenizer loaded"
        if tokenizer
        else "⚠ Showing token IDs (tokenizer not found)"
    )
    html_parts = [
        "<style>"
        ".token-span::after { content: attr(data-activation); position: absolute; bottom: 100%; left: 50%; "
        "transform: translateX(-50%); background: #1f2937; color: #fff; padding: 4px 8px; border-radius: 4px; "
        "font-size: 11px; white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 0.2s; "
        "margin-bottom: 4px; z-index: 1000; }"
        ".token-span:hover::after { opacity: 1; }"
        '.token-span::before { content: ""; position: absolute; bottom: 100%; left: 50%; '
        "transform: translateX(-50%); border: 4px solid transparent; border-top-color: #1f2937; "
        "opacity: 0; pointer-events: none; transition: opacity 0.2s; z-index: 1000; }"
        ".token-span:hover::before { opacity: 1; }"
        "</style>"
        f"<div style='font-family: system-ui; padding: 12px;'>"
        f"<h3 style='margin-top: 0;'>Top {len(sorted_batches)} Activating Examples</h3>"
        f"<p style='color: #666; font-size: 0.9em;'>Adapter: {adapter_name} | Hookpoint: {hookpoint} | "
        f"Latent: {latent_idx} | {tokenizer_status}</p>"
    ]

    for rank, (batch_idx, positions) in enumerate(sorted_batches):
        # Sort positions by activation within this batch
        positions.sort(key=lambda x: x[1], reverse=True)
        max_activation = positions[0][1]
        position_indices = {seq_idx: act for seq_idx, act in positions}

        # Get the token sequence for this batch
        token_ids = tokens_all[batch_idx]

        # Convert to token strings
        if tokenizer:
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
        else:
            tokens = [f"tok_{tid}" for tid in token_ids.tolist()]

        # Build highlighted HTML with all activating positions
        highlighted_seq = []
        for i, tok in enumerate(tokens):
            display_tok = (
                str(tok)
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ", "&nbsp;")
            )

            if i in position_indices:
                # Highlight this activating token
                activation_val = position_indices[i]
                intensity = min(activation_val / 5.0, 1.0)
                r = 255
                g = int(255 * (1 - intensity))
                b = int(255 * (1 - intensity))
                bg = f"rgb({r},{g},{b})"
                # Use white text for high intensity (dark backgrounds), dark text for low intensity
                text_color = "#fff" if intensity > 0.5 else "#1f2937"
                highlighted_seq.append(
                    f'<span class="token-span" data-activation="{activation_val:.4f}" '
                    f'style="background:{bg};color:{text_color};padding:2px 4px;margin:1px;border-radius:3px;'
                    f'display:inline-block;font-family:monospace;font-weight:bold;position:relative;cursor:help;">{display_tok}</span>'
                )
            else:
                highlighted_seq.append(
                    f'<span style="padding:2px 4px;margin:1px;display:inline-block;'
                    f'font-family:monospace;color:#999;">{display_tok}</span>'
                )

        # Create position summary
        pos_summary = ", ".join(
            [f"pos {seq_idx} ({act:.4f})" for seq_idx, act in positions]
        )

        html_parts.append(
            f"<div style='margin: 12px 0; padding: 8px; background: #f9f9f9; border-radius: 4px;'>"
            f"<div style='font-size: 0.85em; color: #666; margin-bottom: 4px;'>"
            f"Rank #{rank + 1} | Max: {max_activation:.4f} | Batch {batch_idx} | {len(positions)} activation(s): {pos_summary}</div>"
            f"<div style='line-height: 1.8;'>{' '.join(highlighted_seq)}</div>"
            f"</div>"
        )

    html_parts.append("</div>")
    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------
def generate_steered(
    prompt: str,
    steering_data: list[list],
    amplification: float,
    max_new_tokens: int,
) -> tuple[str, str, str]:
    """Generate baseline vs steered completions with activation visualization.

    *steering_data* is a list of rows ``[hookpoint, latent_idx, effect]``
    coming from the Gradio Dataframe component.

    Returns:
        (baseline_html, steered_html, stats_html): HTML-formatted outputs with
        color-coded tokens and activation statistics.
    """
    if state.model is None:
        return "No model loaded.", "", ""

    tokenizer = state.tokenizer
    model = state.model

    # Format as chat and tokenize
    # Use tokenize=True to get correct token IDs directly (avoids double BOS)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = (
            torch.tensor([input_ids])
            if isinstance(input_ids, list)
            else input_ids["input_ids"]
        )
    input_ids = input_ids.to(state.device)
    enc = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    prompt_length = input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )

    # Parse steering configuration
    feature_dict: dict[str, list[tuple[int, str]]] = {}
    has_data = False
    if steering_data is not None:
        if hasattr(steering_data, "empty"):
            has_data = not steering_data.empty
            if has_data:
                steering_data = steering_data.values.tolist()
        else:
            has_data = len(steering_data) > 0

    if has_data:
        for row in steering_data:
            if len(row) < 3:
                continue
            hp, idx_str, effect = row[0], row[1], row[2]
            if not hp or not str(idx_str).strip():
                continue
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                continue
            effect = str(effect).strip().lower()
            if effect not in ("enable", "disable", "isolate"):
                effect = "enable"
            feature_dict.setdefault(hp, []).append((idx, effect))

    if not feature_dict:
        # No steering - just run baseline
        with torch.no_grad():
            baseline_ids = model.generate(**enc, **gen_kwargs)
        full_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
        return (
            f"<div style='font-family:monospace;white-space:pre-wrap;padding:12px;'>{full_text}</div>",
            "<div style='color:#999;padding:12px;'>No steering rows provided</div>",
            "",
        )

    # --- Baseline generation (no activation tracking during generation) ---
    with torch.no_grad():
        baseline_ids = model.generate(**enc, **gen_kwargs)

    # --- Steered generation (re-use same enc from above) ---
    with FeatureSteeringContext(
        model, feature_dict, verbose=False, amplification=amplification
    ):
        with torch.no_grad():
            steered_ids = model.generate(**enc, **gen_kwargs)

    # --- Capture activations with single forward pass on complete sequences ---
    # Use the generated IDs directly to avoid tokenization mismatches
    baseline_activations = {}
    steered_activations = {}

    # Baseline activations: forward pass on baseline output
    # _last_z stores pre-TopK activations; apply hard TopK mask to extract
    # only the k active latents.
    baseline_input = {
        "input_ids": baseline_ids,
        "attention_mask": torch.ones_like(baseline_ids),
    }

    with torch.no_grad():
        model(**baseline_input)

    for hp, latent_list in feature_dict.items():
        if hp not in state.wrapped_modules:
            continue
        module = state.wrapped_modules[hp]
        if module._last_z is not None:
            k = module.topk.k
            z_masked = module._last_z * _hard_topk_mask(module._last_z, k)
            if hp not in baseline_activations:
                baseline_activations[hp] = {}
            for latent_idx, _ in latent_list:
                baseline_activations[hp][latent_idx] = (
                    z_masked[0, :, latent_idx].cpu().clone()
                )

    # Steered activations: forward pass on steered output
    # Run WITHOUT steering context — we want to see the model's natural
    # activations on the steered text (what does the model "see"?)
    steered_input = {
        "input_ids": steered_ids,
        "attention_mask": torch.ones_like(steered_ids),
    }

    with torch.no_grad():
        model(**steered_input)

    for hp, latent_list in feature_dict.items():
        if hp not in state.wrapped_modules:
            continue
        module = state.wrapped_modules[hp]
        if module._last_z is not None:
            k = module.topk.k
            z_masked = module._last_z * _hard_topk_mask(module._last_z, k)
            if hp not in steered_activations:
                steered_activations[hp] = {}
            for latent_idx, _ in latent_list:
                steered_activations[hp][latent_idx] = (
                    z_masked[0, :, latent_idx].cpu().clone()
                )

    # Generate HTML outputs
    baseline_html = _render_generation_with_activations(
        tokenizer, baseline_ids[0], prompt_length, baseline_activations, "Baseline"
    )
    steered_html = _render_generation_with_activations(
        tokenizer, steered_ids[0], prompt_length, steered_activations, "Steered"
    )
    stats_html = _render_activation_stats(
        baseline_activations, steered_activations, amplification
    )

    return baseline_html, steered_html, stats_html


def _render_generation_with_activations(
    tokenizer,
    token_ids: torch.Tensor,
    prompt_length: int,
    activations: dict,
    label: str,
) -> str:
    """Render generated tokens with color-coding based on activations."""
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Get special token IDs to mask them out
    special_token_ids = set(tokenizer.all_special_ids)

    # Compute aggregate activation per token (max across all steered latents)
    token_activations = torch.zeros(len(tokens))
    for hp_acts in activations.values():
        for latent_acts in hp_acts.values():
            if len(latent_acts) >= len(tokens):
                token_activations = torch.maximum(
                    token_activations, latent_acts[: len(tokens)]
                )

    # Mask out special tokens
    for i, token_id in enumerate(token_ids):
        if token_id.item() in special_token_ids:
            token_activations[i] = 0.0

    # Render HTML
    spans = []
    for i, (tok, act) in enumerate(zip(tokens, token_activations)):
        display_tok = (
            str(tok).replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
        )

        # Color based on activation
        if act > 1e-4:
            intensity = min(act.item() / 5.0, 1.0)
            r = 255
            g = int(255 * (1 - intensity))
            b = int(255 * (1 - intensity))
            bg = f"rgb({r},{g},{b})"
            # Use white text for high intensity (dark backgrounds), dark text for low intensity
            text_color = "#fff" if intensity > 0.5 else "#1f2937"
        else:
            bg = "#fff"
            text_color = "#1f2937"

        # Mark prompt vs completion
        if i < prompt_length:
            border = "border-left: 3px solid #3b82f6;"
            title_extra = " (PROMPT)"
        else:
            border = ""
            title_extra = ""

        spans.append(
            f'<span class="token-span" data-activation="{act:.4f}{title_extra}" '
            f'style="background:{bg};color:{text_color};padding:2px 4px;margin:1px;{border}'
            f'border-radius:3px;display:inline-block;font-family:monospace;position:relative;cursor:help;">'
            f"{display_tok}</span>"
        )

    return (
        "<style>"
        ".token-span::after { content: attr(data-activation); position: absolute; bottom: 100%; left: 50%; "
        "transform: translateX(-50%); background: #1f2937; color: #fff; padding: 4px 8px; border-radius: 4px; "
        "font-size: 11px; white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 0.2s; "
        "margin-bottom: 4px; z-index: 1000; }"
        ".token-span:hover::after { opacity: 1; }"
        '.token-span::before { content: ""; position: absolute; bottom: 100%; left: 50%; '
        "transform: translateX(-50%); border: 4px solid transparent; border-top-color: #1f2937; "
        "opacity: 0; pointer-events: none; transition: opacity 0.2s; z-index: 1000; }"
        ".token-span:hover::before { opacity: 1; }"
        "</style>"
        f"<div style='padding:8px;'>"
        f"<div style='font-weight:bold;margin-bottom:8px;color:#666;'>{label}</div>"
        f"<div style='line-height:2.2;'>{' '.join(spans)}</div>"
        f"</div>"
    )


def _render_activation_stats(
    baseline_acts: dict,
    steered_acts: dict,
    amplification: float,
) -> str:
    """Render activation statistics comparing baseline vs steered."""
    if not baseline_acts and not steered_acts:
        return "<div style='padding:8px;color:#999;'>No activation data</div>"

    stats_rows = []
    all_hps = set(baseline_acts.keys()) | set(steered_acts.keys())

    for hp in sorted(all_hps):
        baseline_latents = baseline_acts.get(hp, {})
        steered_latents = steered_acts.get(hp, {})
        all_latents = set(baseline_latents.keys()) | set(steered_latents.keys())

        for latent_idx in sorted(all_latents):
            baseline_vals = baseline_latents.get(latent_idx)
            steered_vals = steered_latents.get(latent_idx)

            baseline_mean = (
                baseline_vals.mean().item() if baseline_vals is not None else 0.0
            )
            steered_mean = (
                steered_vals.mean().item() if steered_vals is not None else 0.0
            )

            delta = steered_mean - baseline_mean
            delta_pct = (
                (delta / (baseline_mean + 1e-8)) * 100 if baseline_mean > 1e-8 else 0
            )

            stats_rows.append(
                f"<tr>"
                f"<td style='padding:4px 8px;'><code>{hp.split('.')[-1]}</code></td>"
                f"<td style='padding:4px 8px;'>{latent_idx}</td>"
                f"<td style='padding:4px 8px;text-align:right;'>{baseline_mean:.4f}</td>"
                f"<td style='padding:4px 8px;text-align:right;'>{steered_mean:.4f}</td>"
                f"<td style='padding:4px 8px;text-align:right;color:{'#16a34a' if delta > 0 else '#dc2626'};'>"
                f"{delta:+.4f} ({delta_pct:+.1f}%)</td>"
                f"</tr>"
            )

    return (
        f"<div style='padding:8px;'>"
        f"<div style='font-weight:bold;margin-bottom:8px;color:#666;'>Activation Statistics</div>"
        f"<div style='font-size:0.85em;color:#999;margin-bottom:8px;'>Amplification: {amplification}×</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
        f"<thead><tr style='background:#f3f4f6;'>"
        f"<th style='padding:6px 8px;text-align:left;color:#1f2937;font-weight:600;'>Hookpoint</th>"
        f"<th style='padding:6px 8px;text-align:left;color:#1f2937;font-weight:600;'>Latent</th>"
        f"<th style='padding:6px 8px;text-align:right;color:#1f2937;font-weight:600;'>Baseline Mean</th>"
        f"<th style='padding:6px 8px;text-align:right;color:#1f2937;font-weight:600;'>Steered Mean</th>"
        f"<th style='padding:6px 8px;text-align:right;color:#1f2937;font-weight:600;'>Change</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(stats_rows)}</tbody>"
        f"</table>"
        f"</div>"
    )

"""
TopKLoRA Interactive Dashboard.

Three tabs:
  1. Model Loading — load base model + adapter (auto-discovered from models/)
  2. Activation Visualization — per-token latent heatmap
  3. Interactive Steering — side-by-side baseline vs steered generation

Run from the repo root:
    python -m dashboard.app
"""

import gradio as gr

from dashboard.backend import (
    compute_token_activations,
    discover_adapters,
    discover_base_models,
    discover_cached_adapters,
    generate_steered,
    get_adapter_choices,
    get_adapter_info,
    get_cached_hookpoints,
    get_cached_hookpoint_config,
    get_cached_latent_choices,
    load_model,
    load_top_activating_examples,
    render_latent_from_cache,
)

# ------------------------------------------------------------------
# Discover what's on disk at startup
# ------------------------------------------------------------------
_base_models = discover_base_models()
_adapters = discover_adapters()
_cached_adapters = discover_cached_adapters()

# Build lookup: adapter display name -> adapter info dict
_adapter_lookup: dict[str, dict] = {a["display"]: a for a in _adapters}


# ------------------------------------------------------------------
# Helpers wired to Gradio events
# ------------------------------------------------------------------
def _on_adapter_select(adapter_display):
    """When an adapter is selected, resolve its filesystem path."""
    info = _adapter_lookup.get(adapter_display, {})
    return info.get("path", "")


def _on_load_model(base_path, adapter_path, device):
    status, hookpoint_html = load_model(base_path, adapter_path, device)
    choices = get_adapter_choices()
    return (
        status,
        hookpoint_html,
        gr.update(choices=choices, value=choices[0] if choices else None),
        gr.update(choices=choices, value=choices[0] if choices else None),
    )


def _on_hookpoint_change(hookpoint):
    info = get_adapter_info(hookpoint)
    return f"r = {info['r']},  k = {info['k']}"


def _on_visualize(text, hookpoint, latent_idx):
    if latent_idx is None:
        return "<p>Enter a latent index to visualize.</p>"
    return compute_token_activations(text, hookpoint, int(latent_idx))


def _on_latent_change(latent_idx):
    """Instantly re-render from cached activations when latent index changes."""
    if latent_idx is None:
        return "<p>Enter a latent index to visualize.</p>"
    return render_latent_from_cache(int(latent_idx))


def _on_generate(prompt, steering_data, amplification, max_new_tokens):
    rows = steering_data if steering_data is not None else []
    baseline_html, steered_html, stats_html = generate_steered(
        prompt, rows, float(amplification), int(max_new_tokens)
    )
    return baseline_html, steered_html, stats_html


def _on_add_steering_row(current_df, hookpoint, latent_idx, effect):
    """Add a new steering row to the dataframe."""
    # Convert DataFrame to list of lists if needed
    if current_df is None or (hasattr(current_df, "empty") and current_df.empty):
        rows = []
    elif hasattr(current_df, "values"):
        # It's a pandas DataFrame
        rows = current_df.values.tolist()
    else:
        rows = list(current_df)

    new_row = [hookpoint, int(latent_idx), effect]
    rows.append(new_row)
    return rows


def _on_cached_adapter_select(adapter_name):
    """When a cached adapter is selected, populate hookpoint and latent dropdowns."""
    if not adapter_name:
        return gr.update(choices=[]), gr.update(choices=[])

    hookpoints = get_cached_hookpoints(adapter_name)
    hookpoint_update = gr.update(
        choices=hookpoints, value=hookpoints[0] if hookpoints else None
    )

    # Also populate latent dropdown for first hookpoint
    if hookpoints:
        latent_choices = get_cached_latent_choices(adapter_name, hookpoints[0])
        choices_display = [display for display, _ in latent_choices]
        choices_values = [idx for _, idx in latent_choices]
        latent_update = gr.update(
            choices=list(zip(choices_display, choices_values)),
            value=choices_values[0] if choices_values else None
        )
    else:
        latent_update = gr.update(choices=[])

    return hookpoint_update, latent_update


def _on_cached_hookpoint_select(adapter_name, hookpoint):
    """When a cached hookpoint is selected, show its config and populate latent dropdown."""
    if not adapter_name or not hookpoint:
        return "Select an adapter and hookpoint", gr.update(choices=[])

    config = get_cached_hookpoint_config(adapter_name, hookpoint)
    width = config.get("width", "unknown")
    config_text = f"Width (r): {width}"

    # Load latent choices with p_active
    latent_choices = get_cached_latent_choices(adapter_name, hookpoint)
    choices_display = [display for display, _ in latent_choices]
    choices_values = [idx for _, idx in latent_choices]

    return config_text, gr.update(
        choices=list(zip(choices_display, choices_values)),
        value=choices_values[0] if choices_values else None
    )


def _on_show_cached_examples(adapter_name, hookpoint, latent_idx, n_examples):
    """Show top activating examples for a cached latent."""
    if latent_idx is None:
        return "<p>Enter a latent index.</p>"
    if not adapter_name or not hookpoint:
        return "<p>Select an adapter and hookpoint first.</p>"
    return load_top_activating_examples(
        adapter_name, hookpoint, int(latent_idx), int(n_examples)
    )


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
with gr.Blocks(title="TopKLoRA Dashboard") as demo:
    gr.Markdown("# TopKLoRA Dashboard")

    # ---- Tab 1: Model Loading ----
    with gr.Tab("Model Loading"):
        with gr.Row():
            base_model_dd = gr.Dropdown(
                label="Base model",
                choices=_base_models,
                value=_base_models[0] if _base_models else None,
                allow_custom_value=True,
            )
            adapter_dd = gr.Dropdown(
                label="Adapter",
                choices=list(_adapter_lookup.keys()),
                value=list(_adapter_lookup.keys())[0] if _adapter_lookup else None,
            )
        # Hidden textbox carries the resolved adapter path
        adapter_path_tb = gr.Textbox(visible=False)
        with gr.Row():
            device_dd = gr.Dropdown(
                label="Device",
                choices=["cpu", "cuda", "mps"],
                value="cpu",
            )
        load_btn = gr.Button("Load Model", variant="primary")
        status_tb = gr.Textbox(label="Status", interactive=False)
        hookpoint_config = gr.HTML(label="Adapter Configuration")

        # Wire adapter dropdown to resolve path
        adapter_dd.change(
            fn=_on_adapter_select,
            inputs=[adapter_dd],
            outputs=[adapter_path_tb],
        )

        # Initialise adapter path from the default selection
        if _adapter_lookup:
            adapter_path_tb.value = list(_adapter_lookup.values())[0]["path"]

    # ---- Tab 2: Activation Visualization ----
    with gr.Tab("Activation Visualization"):
        with gr.Row():
            viz_text = gr.Textbox(
                label="Input text",
                placeholder="Enter text to visualize activations for...",
                lines=3,
            )
        with gr.Row():
            viz_hookpoint = gr.Dropdown(label="Hookpoint", choices=[])
            viz_latent = gr.Number(label="Latent index", value=0, precision=0)
        hookpoint_info = gr.Textbox(label="Adapter info", interactive=False)
        viz_btn = gr.Button("Visualize", variant="primary")
        viz_html = gr.HTML(label="Activation Heatmap")

    # ---- Tab 3: Interactive Steering ----
    with gr.Tab("Interactive Steering"):
        steer_prompt = gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt...",
            lines=3,
        )
        with gr.Row():
            amp_num = gr.Number(label="Amplification", value=5.0)
            tokens_num = gr.Number(label="Max new tokens", value=128, precision=0)

        gr.Markdown("### Add Steering Rule")
        with gr.Row():
            steer_hookpoint_dd = gr.Dropdown(
                label="Hookpoint",
                choices=[],
                allow_custom_value=True,
            )
            steer_latent_num = gr.Number(label="Latent Index", value=0, precision=0)
            steer_effect_dd = gr.Dropdown(
                label="Effect",
                choices=["enable", "disable", "isolate"],
                value="enable",
            )
            add_row_btn = gr.Button("Add Row", size="sm")

        steer_df = gr.Dataframe(
            headers=["Hookpoint", "Latent Index", "Effect"],
            datatype=["str", "number", "str"],
            row_count=(1, "dynamic"),
            col_count=(3, "fixed"),
            label="Steering Configuration",
            interactive=True,
        )

        gen_btn = gr.Button("Generate", variant="primary")

        steer_stats_html = gr.HTML(label="Activation Statistics")

        with gr.Row():
            baseline_out = gr.HTML(label="Baseline Output")
            steered_out = gr.HTML(label="Steered Output")

    # ---- Tab 4: Cached Activation Explorer ----
    with gr.Tab("Cached Activations"):
        gr.Markdown(
            "Explore top activating examples from cached activations in `delphi_cache/`. "
            "No model loading required."
        )
        with gr.Row():
            cached_adapter_dd = gr.Dropdown(
                label="Cached Adapter",
                choices=_cached_adapters,
                value=_cached_adapters[0] if _cached_adapters else None,
            )
            # Initialize hookpoints from the first adapter
            _initial_hookpoints = (
                get_cached_hookpoints(_cached_adapters[0]) if _cached_adapters else []
            )
            cached_hookpoint_dd = gr.Dropdown(
                label="Hookpoint",
                choices=_initial_hookpoints,
                value=_initial_hookpoints[0] if _initial_hookpoints else None,
            )
        cached_config_info = gr.Textbox(label="Hookpoint Config", interactive=False)

        with gr.Row():
            # Initialize latent choices from first hookpoint
            _initial_latent_choices = []
            if _cached_adapters and _initial_hookpoints:
                _initial_latent_choices = get_cached_latent_choices(
                    _cached_adapters[0], _initial_hookpoints[0]
                )
            _latent_choices_display = [display for display, _ in _initial_latent_choices]
            _latent_choices_values = [idx for _, idx in _initial_latent_choices]

            cached_latent_dd = gr.Dropdown(
                label="Latent",
                choices=list(zip(_latent_choices_display, _latent_choices_values)) if _latent_choices_values else [],
                value=_latent_choices_values[0] if _latent_choices_values else None,
            )
            cached_n_examples = gr.Number(
                label="Number of Examples", value=10, precision=0
            )

        show_examples_btn = gr.Button("Show Top Examples", variant="primary")
        cached_examples_html = gr.HTML(label="Top Activating Examples")

    # ---- Wiring ----
    load_btn.click(
        fn=_on_load_model,
        inputs=[base_model_dd, adapter_path_tb, device_dd],
        outputs=[status_tb, hookpoint_config, viz_hookpoint, steer_hookpoint_dd],
    )

    add_row_btn.click(
        fn=_on_add_steering_row,
        inputs=[steer_df, steer_hookpoint_dd, steer_latent_num, steer_effect_dd],
        outputs=[steer_df],
    )

    viz_hookpoint.change(
        fn=_on_hookpoint_change,
        inputs=[viz_hookpoint],
        outputs=[hookpoint_info],
    )

    viz_btn.click(
        fn=_on_visualize,
        inputs=[viz_text, viz_hookpoint, viz_latent],
        outputs=[viz_html],
    )

    viz_latent.change(
        fn=_on_latent_change,
        inputs=[viz_latent],
        outputs=[viz_html],
    )

    gen_btn.click(
        fn=_on_generate,
        inputs=[steer_prompt, steer_df, amp_num, tokens_num],
        outputs=[baseline_out, steered_out, steer_stats_html],
    )

    cached_adapter_dd.change(
        fn=_on_cached_adapter_select,
        inputs=[cached_adapter_dd],
        outputs=[cached_hookpoint_dd, cached_latent_dd],
    )

    cached_hookpoint_dd.change(
        fn=_on_cached_hookpoint_select,
        inputs=[cached_adapter_dd, cached_hookpoint_dd],
        outputs=[cached_config_info, cached_latent_dd],
    )

    show_examples_btn.click(
        fn=_on_show_cached_examples,
        inputs=[
            cached_adapter_dd,
            cached_hookpoint_dd,
            cached_latent_dd,
            cached_n_examples,
        ],
        outputs=[cached_examples_html],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

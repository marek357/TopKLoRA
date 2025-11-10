# Feature Steering for TopKLoRALinearSTE Adapters

This directory contains configuration files for steering specific features (latents) in trained TopKLoRALinearSTE adapters.

## Overview

Feature steering allows you to enable or disable specific latents during inference, giving you fine-grained control over model behavior. This is particularly useful for:

- Testing interpretability hypotheses (e.g., "Does feature 217 really control question-asking behavior?")
- Ablation studies (disable features to see their effect)
- Safety interventions (disable features associated with harmful content)
- Behavior modification (enable features to encourage specific responses)

## Quick Start

### 1. Basic Usage

```bash
# Use default configuration
python steer.py

# Use a specific config file
python steer.py --config-name dpo_512_4_example

# Override specific settings
python steer.py model.adapter_path=path/to/adapter steering.verbose=true
```

### 2. Programmatic Usage (from Python)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.steering import FeatureSteeringContext

# Load model and adapter
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
model = PeftModel.from_pretrained(model, "path/to/adapter")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Define features to steer
feature_dict = {
    "base_model.model.model.layers.11.self_attn.q_proj.topk": [
        (217, "enable"),   # Enable feature 217
        (45, "disable"),   # Disable feature 45
    ]
}

# Generate with steering
with FeatureSteeringContext(model, feature_dict):
    inputs = tokenizer("How do I make a cake?", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0]))
```

## Configuration Files

### `default.yaml`
Base configuration with all available options documented. Start here to understand the config structure.

### `dpo_512_4_example.yaml`
Example configuration for steering features discovered in the DPO model with r=512, k=4. Based on `assess_autointerp_dpo.py` results showing features related to harmful queries.

### `dpo_4096_32_example.yaml`
Example configuration for steering features from the larger DPO model (r=4096, k=32), focusing on politeness and response style features.

## Configuration Structure

```yaml
experiment_name: "my_steering_experiment"

model:
  model_name: "google/gemma-2-2b-it"
  adapter_path: "path/to/trained/adapter"
  device: "cuda"
  dtype: "bfloat16"

steering:
  features:
    # Full adapter name from model.named_modules()
    "base_model.model.model.layers.11.self_attn.q_proj.topk":
      - feature: 217
        effect: "enable"  # or "disable"
        description: "Feature description for documentation"
      
      - feature: 45
        effect: "disable"
  
  verbose: true
  list_adapters: true  # Print all available adapters before steering

prompts:
  - "How do I make a cake?"
  - "What is the capital of France?"

output:
  output_dir: "outputs/steering"
  save_outputs: true
  compare_baseline: true  # Compare with non-steered outputs
```

## Finding Adapter Names

Not sure what adapter names to use? Run with `steering.list_adapters: true` to see all available adapters:

```bash
python steer.py steering.list_adapters=true
```

Or use the utility function:

```python
from src.steering import list_available_adapters

adapters = list_available_adapters(model, verbose=True)
# Prints all adapter names with their properties (r, k, temperature)
```

## Finding Feature Numbers

Feature numbers come from your auto-interpretation results. Use `assess_autointerp_dpo.py` to discover interesting features:

```bash
python scripts/assess_autointerp_dpo.py
```

This will show high-scoring features with their descriptions, e.g.:
```
0.79 Pronouns and auxiliary verbs in question structures
     Matrix: self_attn (q_proj)
     Feature: 217
```

## Advanced Usage

### Custom Steering Logic

The `src/steering.py` module provides low-level control:

```python
from src.steering import steer_features, remove_steering_hooks

# Register hooks manually
hooks_info = steer_features(model, feature_dict, verbose=True)

# Run inference
outputs = model.generate(...)

# Clean up
remove_steering_hooks(hooks_info["hooks"])
```

### Batch Steering from Auto-Interpretation Results

```python
from src.steering import create_steering_dict_from_script_output

# Parse your auto-interpretation results
results = [
    {"adapter_name": "base_model.model.model.layers.11.self_attn.q_proj.topk",
     "feature_num": 217, "effect": "enable"},
    {"adapter_name": "base_model.model.model.layers.11.self_attn.q_proj.topk",
     "feature_num": 45, "effect": "disable"}
]

steering_dict = create_steering_dict_from_script_output(results)
steer_features(model, steering_dict)
```

## Output Structure

When you run steering with `save_outputs: true`, you'll get:

```
outputs/steering/YYYYMMDD_HHMMSS/
├── config.yaml          # Full configuration used
├── results.json         # Machine-readable results
└── comparison.txt       # Human-readable comparison of baseline vs steered
```

### `results.json` structure:
```json
{
  "experiment_name": "...",
  "model": "...",
  "adapter": "...",
  "steering_config": {...},
  "prompts": [...],
  "outputs": {
    "baseline": [
      {"prompt": "...", "output": "..."},
      ...
    ],
    "steered": [
      {"prompt": "...", "output": "..."},
      ...
    ]
  }
}
```

## Tips & Best Practices

1. **Start Simple**: Test steering with a single feature on a few prompts before scaling up.

2. **Compare Baselines**: Always use `compare_baseline: true` to see the effect of steering.

3. **Verify Adapter Names**: Use `list_adapters: true` to ensure you're targeting the right modules.

4. **Document Features**: Add descriptions to your feature configs for future reference.

5. **Version Control**: Keep your steering configs in version control to track experiments.

6. **Iterate**: Start with features that have high auto-interpretation scores (>0.7) for clearer effects.

## Troubleshooting

### "Adapter not found in model"
- Check adapter names with `list_adapters: true`
- Use the full module path from `model.named_modules()`
- The steering code supports partial matching, but exact names are more reliable

### "Feature index out of bounds"
- Check the `r` value of your adapter (how many latents it has)
- Feature numbers should be in range [0, r-1]
- The code will log a warning and skip invalid indices

### Hook not affecting outputs
- Verify the adapter is actually being used (not bypassed)
- Check that you're using the wrapper's forward hook correctly
- Ensure the model is in eval mode: `model.eval()`

## Related Files

- `src/steering.py` - Core steering implementation
- `steer.py` - Main entrypoint script
- `scripts/assess_autointerp_dpo.py` - Find interesting features to steer

## Examples from Research

Based on auto-interpretation results, here are some interesting features to try:

**DPO Model (512_4):**
- Feature 217: "Key words in user queries seeking instructions for illegal or harmful activities" (score: 0.78)
- Feature 128: "Words central to social identity and sensitive topics" (score: 0.74)

**DPO Model (4096_32):**
- Feature ???: "Comma following 'Sorry' in chatbot responses" (score: 0.89)
- Feature ???: "Direct 'Yes' responses to questions" (score: 0.82)

*(Note: Replace ??? with actual feature numbers from your auto-interpretation results)*

## Citation

If you use this steering functionality in your research, please cite the SparseLoRA project.

#!/usr/bin/env python3
"""
Layer attribution analysis for toxicity using gradient × activation.

This script identifies which transformer layers have the highest impact on
toxic vs safe outputs using gradient-based attribution. This helps determine
which single layer to target for toxicity-reducing LoRA adapters.

Usage:
    python find_toxicity_critical_layer.py --model google/gemma-2b --num_samples 100
    
The script will:
1. Load toxicity prompts from RealToxicityPrompts
2. Generate toxic and safe completions
3. Compute gradient × activation attribution for each layer
4. Rank layers by their impact on toxicity
5. Output recommendations for which layer(s) to target
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt


def load_toxicity_prompts(num_samples: int = 100) -> List[str]:
    """Load prompts from RealToxicityPrompts dataset."""
    print(f"Loading {num_samples} prompts from RealToxicityPrompts...")

    dataset = load_dataset(
        "allenai/real-toxicity-prompts",
        split="train",
        cache_dir="./cache"
    )

    # Extract prompts with continuation text
    prompts = []
    for item in dataset:
        if item.get('prompt') and item['prompt'].get('text'):
            prompt_text = item['prompt']['text'].strip()
            if len(prompt_text) > 10:  # Filter very short prompts
                prompts.append(prompt_text)

        if len(prompts) >= num_samples:
            break

    print(f"Loaded {len(prompts)} prompts")
    return prompts[:num_samples]


def generate_with_toxicity_score(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    toxic_tokens: set = None
) -> Tuple[str, List[int]]:
    """
    Generate text and track which tokens are likely toxic.
    Returns (generated_text, toxic_token_positions)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Simple heuristic: tokens with high probability on toxic words
    # (In practice, you'd use Perspective API, but this is for attribution)
    toxic_positions = []
    if toxic_tokens:
        for i, token_id in enumerate(generated_ids[input_len:].tolist()):
            if token_id in toxic_tokens:
                toxic_positions.append(i)

    return generated_text, toxic_positions


def get_toxic_token_ids(tokenizer) -> set:
    """
    Get a set of token IDs that are commonly associated with toxic content.
    This is a simple heuristic - in production, use Perspective API.
    """
    # Common toxic/offensive words (keeping this family-friendly for the code)
    toxic_words = [
        "kill", "hate", "stupid", "idiot", "dumb", "damn", "hell",
        "terrible", "awful", "worst", "suck", "crap", "shit", "fuck",
        "ass", "bitch", "dick", "cock", "pussy", "bastard", "asshole"
    ]

    toxic_ids = set()
    for word in toxic_words:
        # Try different casings and with spaces
        for variant in [word, word.upper(), word.capitalize(), f" {word}", f"{word} "]:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            toxic_ids.update(token_ids)

    print(f"Tracking {len(toxic_ids)} potentially toxic token IDs")
    return toxic_ids


def compute_layer_attribution(
    model,
    tokenizer,
    prompt: str,
    target_token_position: int,
    target_is_toxic: bool = True
) -> Dict[int, float]:
    """
    Compute gradient × activation attribution for each layer.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Input prompt
        target_token_position: Position of token to attribute (after prompt)
        target_is_toxic: If True, compute gradients for toxic direction

    Returns:
        Dictionary mapping layer_idx -> attribution_score
    """
    model.zero_grad()

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Storage for activations and gradients
    layer_activations = {}
    layer_gradients = {}

    def get_forward_hook(layer_idx):
        def hook(module, input, output):
            # output is tuple (hidden_states, ...) for decoder layers
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Store activation (keep in computation graph!)
            layer_activations[layer_idx] = hidden_states
            # Register backward hook to capture gradients
            if hidden_states.requires_grad:
                def grad_hook(grad):
                    layer_gradients[layer_idx] = grad.detach().clone()
                hidden_states.register_hook(grad_hook)
        return hook

    # Register hooks on all decoder layers
    hooks = []
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        for i in range(num_layers):
            hook = model.model.layers[i].register_forward_hook(
                get_forward_hook(i))
            hooks.append(hook)
    else:
        # Fallback for different model structures
        num_layers = model.config.num_hidden_layers
        for i in range(num_layers):
            hook = model.transformer.h[i].register_forward_hook(
                get_forward_hook(i))
            hooks.append(hook)

    # Forward pass with gradient tracking
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits  # [batch, seq, vocab]

    # Get logits for the target position (after prompt)
    if target_token_position >= logits.shape[1] - input_len:
        # Adjust if target is beyond current sequence
        target_token_position = logits.shape[1] - input_len - 1

    target_logits = logits[0, input_len + target_token_position, :]

    # Define target: if toxic, maximize prob of toxic tokens; if safe, minimize
    toxic_ids = get_toxic_token_ids(tokenizer)
    toxic_mask = torch.zeros_like(target_logits, dtype=torch.bool)
    for tid in toxic_ids:
        if tid < len(toxic_mask):
            toxic_mask[tid] = True

    if toxic_mask.sum() == 0:
        # No toxic tokens in vocab - use different signal
        # Use entropy of distribution (lower entropy = more confident, potentially toxic)
        probs = F.softmax(target_logits, dim=-1)
        target_score = -(probs * torch.log(probs + 1e-10)).sum()
    elif target_is_toxic:
        # Gradient in direction of increasing toxic token probability
        target_score = target_logits[toxic_mask].mean()
    else:
        # Gradient in direction of decreasing toxic token probability
        target_score = -target_logits[toxic_mask].mean()

    # Backward pass
    target_score.backward()

    # Compute attribution: gradient × activation for each layer
    layer_attributions = {}
    for layer_idx in range(num_layers):
        if layer_idx in layer_activations and layer_idx in layer_gradients:
            activation = layer_activations[layer_idx].detach()
            gradient = layer_gradients[layer_idx]

            # Sum over sequence and hidden dims: grad × activation
            # Focus on the prompt tokens only for attribution
            attr = (gradient[:, :input_len, :] *
                    activation[:, :input_len, :]).sum().item()
            layer_attributions[layer_idx] = abs(
                attr)  # Absolute value for magnitude
        else:
            layer_attributions[layer_idx] = 0.0

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return layer_attributions


def analyze_layer_importance(
    model,
    tokenizer,
    prompts: List[str],
    num_samples: int = 50,
    max_new_tokens: int = 20
) -> Dict[int, float]:
    """
    Analyze which layers are most important for toxicity across multiple prompts.
    """
    # Get number of layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        num_layers = model.config.num_hidden_layers

    layer_scores = {i: [] for i in range(num_layers)}

    toxic_ids = get_toxic_token_ids(tokenizer)

    print(f"\nAnalyzing layer importance across {num_samples} prompts...")

    for prompt_idx, prompt in enumerate(tqdm(prompts[:num_samples])):
        try:
            # Generate a continuation (in eval mode)
            model.eval()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_ids = outputs[0]
            continuation_ids = generated_ids[input_len:]

            # Check if any toxic tokens were generated
            has_toxic = any(
                tid.item() in toxic_ids for tid in continuation_ids)

            # Now compute attribution (need gradients, so switch to train mode)
            model.train()  # Enable gradients

            # Compute attribution for first few tokens
            for token_pos in range(min(3, len(continuation_ids))):
                try:
                    attr = compute_layer_attribution(
                        model,
                        tokenizer,
                        prompt,
                        target_token_position=token_pos,
                        target_is_toxic=has_toxic
                    )

                    # Accumulate scores
                    for layer_idx, score in attr.items():
                        layer_scores[layer_idx].append(score)

                except Exception as e:
                    print(
                        f"\nWarning: Failed to compute attribution for prompt {prompt_idx}, token {token_pos}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"\nWarning: Failed to process prompt {prompt_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Clear cache periodically
        if (prompt_idx + 1) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Set back to eval mode
    model.eval()

    # Aggregate scores per layer
    layer_importance = {}
    for layer_idx, scores in layer_scores.items():
        if scores:
            layer_importance[layer_idx] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'total': np.sum(scores)
            }
        else:
            layer_importance[layer_idx] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'total': 0.0
            }

    return layer_importance


def plot_layer_importance(
    layer_importance: Dict[int, Dict[str, float]],
    output_path: str = "layer_toxicity_attribution.png"
):
    """Plot layer importance scores."""
    layer_indices = sorted(layer_importance.keys())
    mean_scores = [layer_importance[i]['mean'] for i in layer_indices]
    std_scores = [layer_importance[i]['std'] for i in layer_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(layer_indices, mean_scores, yerr=std_scores, alpha=0.7, capsize=5)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Attribution Score (|grad × activation|)', fontsize=12)
    plt.title('Layer-wise Toxicity Attribution',
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Highlight top layers
    top_3_indices = sorted(
        layer_indices, key=lambda i: mean_scores[i], reverse=True)[:3]
    for idx in top_3_indices:
        plt.axvline(x=idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
        plt.text(idx, max(mean_scores) * 0.95, f'L{idx}',
                 ha='center', fontsize=10, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def load_model_like_eval_toxicity(base_model: str, adapter_path: str = None, model_it_name: str = None):
    """Load model exactly like eval_toxicity does in evals.py"""
    import logging

    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.mps.is_available() else 'cpu'

    if adapter_path is None:
        # Load base model only
        print('Loading base model...')
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="cpu"
        )

        # Copy chat template from -it model if needed
        if not getattr(tokenizer, "chat_template", None) and model_it_name:
            logging.info("No chat_template found – copying from -it model")
            try:
                toks_it = AutoTokenizer.from_pretrained(
                    model_it_name, use_fast=False)
                if getattr(toks_it, "chat_template", None):
                    tokenizer.chat_template = toks_it.chat_template
                    logging.info("chat_template copied successfully")

                # Merge additional special tokens if needed
                extra = toks_it.special_tokens_map.get(
                    "additional_special_tokens", [])
                if extra:
                    new_tokens = [
                        t for t in extra if t not in tokenizer.get_vocab()]
                    if new_tokens:
                        tokenizer.add_special_tokens(
                            {"additional_special_tokens": new_tokens})
                        model.resize_token_embeddings(len(tokenizer))
                        logging.info(
                            "Added %d extra special tokens", len(new_tokens))
            except Exception as exc:
                logging.warning("Failed to copy -it tokenizer: %s", exc)

        model.to(device)
        model.eval()
    else:
        # Load adapter model (using the fixed loader from evals.py)
        print('Loading adapter model...')
        from peft import PeftModel
        from src.models import TopKLoRALinearSTE

        tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="cpu"
        )

        # Load PEFT adapter
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map="cpu",
            use_safetensors=True
        )

        # Wrap with TopK for inference (if needed)
        wrapped_modules = {}
        replaced = 0
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and not isinstance(module, TopKLoRALinearSTE):
                parent = model.get_submodule(".".join(name.split(".")[:-1]))
                attr = name.split(".")[-1]

                # Read k from adapter config if available
                try:
                    import json
                    with open(f"{adapter_path}/../hparams.json", 'r') as f:
                        hparams = json.load(f)
                        k = hparams['lora_topk']['k_final']
                except:
                    k = 8  # Default fallback

                wrapped = TopKLoRALinearSTE(
                    base=module,
                    layer_name=name,
                    k=k,
                    temperature=0.0,
                    hard_eval=True,
                    relu_latents=True,
                    alpha_over_r=True,
                    k_final=k,
                    temperature_final=0.0
                )
                setattr(parent, attr, wrapped)
                wrapped_modules[name] = wrapped
                replaced += 1

        if replaced > 0:
            print(f"Wrapped {replaced} LoRA modules with TopK for inference")

        model.to(device)
        model.eval()

    # Setup EOS/EOT tokens (same as eval_toxicity)
    eot_token = (
        tokenizer.special_tokens_map.get(
            "additional_special_tokens", [tokenizer.eos_token])[1]
        if len(tokenizer.special_tokens_map.get("additional_special_tokens", [])) > 1
        else tokenizer.eos_token
    )

    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
    base_eos_token_id = tokenizer.eos_token_id

    if hasattr(model.generation_config, 'eos_token_id'):
        eos_token_ids = []
        if isinstance(base_eos_token_id, list):
            eos_token_ids.extend(base_eos_token_id)
        else:
            eos_token_ids.append(base_eos_token_id)

        if eot_token_id not in eos_token_ids:
            eos_token_ids.append(eot_token_id)

        model.generation_config.eos_token_id = eos_token_ids
    else:
        model.generation_config.eos_token_id = [
            base_eos_token_id, eot_token_id]

    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS token ID(s): {model.generation_config.eos_token_id}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(
        description="Find critical layers for toxicity using gradient attribution"
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='google/gemma-2-2b',
        help='Base model name or path'
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        default=None,
        help='Path to adapter checkpoint (optional, for adapter evaluation)'
    )
    parser.add_argument(
        '--model_it_name',
        type=str,
        default='google/gemma-2-2b-it',
        help='Instruction-tuned model for chat template'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of prompts to analyze'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=20,
        help='Max tokens to generate per prompt'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='layer_attribution_results',
        help='Output directory for results'
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("TOXICITY LAYER ATTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Base model: {args.base_model}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(f"Num samples: {args.num_samples}")

    # Load model using the same method as eval_toxicity
    print("\nLoading model...")
    model, tokenizer, device = load_model_like_eval_toxicity(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        model_it_name=args.model_it_name
    )
    model.eval()

    # Get number of layers from model architecture
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        # Fallback for different model structures
        num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Load prompts
    prompts = load_toxicity_prompts(args.num_samples)

    # Analyze layer importance
    layer_importance = analyze_layer_importance(
        model,
        tokenizer,
        prompts,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens
    )

    # Print results
    print("\n" + "="*80)
    print("LAYER IMPORTANCE RANKING")
    print("="*80)

    # Sort by mean attribution score
    sorted_layers = sorted(
        layer_importance.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )

    print(f"\n{'Layer':<10} {'Mean Score':<15} {'Median Score':<15} {'Std Dev':<15}")
    print("-" * 60)

    for layer_idx, scores in sorted_layers[:10]:  # Top 10
        print(
            f"{layer_idx:<10} {scores['mean']:<15.6f} {scores['median']:<15.6f} {scores['std']:<15.6f}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    top_layer = sorted_layers[0][0]
    top_score = sorted_layers[0][1]['mean']

    print(f"\n✓ PRIMARY RECOMMENDATION: Target Layer {top_layer}")
    print(f"  - Highest attribution score: {top_score:.6f}")
    print(f"  - This layer shows the strongest gradient signal for toxicity")

    # Check if there are multiple strong layers
    top_3_layers = [idx for idx, _ in sorted_layers[:3]]
    top_3_scores = [scores['mean'] for _, scores in sorted_layers[:3]]

    if top_3_scores[1] > 0.8 * top_3_scores[0]:  # If second is close to first
        print(f"\n✓ ALTERNATIVE: Multi-layer targeting")
        print(
            f"  - Layers {top_3_layers[0]}, {top_3_layers[1]}, {top_3_layers[2]} show similar importance")
        print(f"  - Consider targeting multiple layers for better coverage")

    # Provide config snippet
    print("\n" + "="*80)
    print("CONFIG SNIPPET")
    print("="*80)
    print("\nAdd to your experiment config YAML:")
    print(f"""
# Single layer targeting (recommended)
layers_to_transform: [{top_layer}]
target_modules:
  - "q_proj"
  - "k_proj" 
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
""")

    if top_3_scores[1] > 0.8 * top_3_scores[0]:
        print("\n# Or multi-layer targeting (if more capacity needed):")
        print(f"layers_to_transform: {top_3_layers}")

    # Save results
    results = {
        'base_model': args.base_model,
        'adapter_path': args.adapter_path,
        'num_samples': args.num_samples,
        'layer_importance': {
            str(k): v for k, v in layer_importance.items()
        },
        'top_layer': int(top_layer),
        'top_3_layers': [int(x) for x in top_3_layers],
        'recommendations': {
            'primary_layer': int(top_layer),
            'primary_score': float(top_score),
            'multi_layer_option': [int(x) for x in top_3_layers] if top_3_scores[1] > 0.8 * top_3_scores[0] else None
        }
    }

    results_path = output_dir / 'layer_attribution_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot
    plot_path = output_dir / 'layer_toxicity_attribution.png'
    plot_layer_importance(layer_importance, str(plot_path))

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

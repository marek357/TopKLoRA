import argparse
import json
import sys
import os
import torch
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import tqdm

# Add root to path so we can import src
sys.path.append(os.getcwd())

from src.evals import init_model_tokenizer_fixed
from src.models import TopKLoRALinearSTE

def _hard_topk_mask(z, k):
    # returns 0/1 mask with exactly k ones along last dim
    idx = z.topk(k, dim=-1).indices
    hard = torch.zeros_like(z)
    return hard.scatter_(-1, idx, 1.0)

def make_hook(module_name, features_for_module, activation_store):
    """
    module_name: string key
    features_for_module: list of feature dicts with 'id' and 'latent_index'
    activation_store: mutable dict where we’ll accumulate results
    """
    feature_ids = [f["id"] for f in features_for_module]
    latent_indices = torch.tensor(
        [f["latent_index"] for f in features_for_module],
        dtype=torch.long
    )

    def hook(module, inputs, output):
        # _last_z was set earlier in forward
        z = module._last_z    # shape [batch, seq, r]
        if z is None:
            return

        # Move to CPU, drop batch dimension (assuming batch size 1)
        z = z.detach().cpu()[0]   # [seq, r]

        # Select interesting latent dims in one go
        # z_selected: [seq, num_features]
        z_selected = z[:, latent_indices]

        # Convert to plain Python lists for JSON
        z_np = z_selected.numpy()
        # Fill activation_store for this module
        for i, feat_id in enumerate(feature_ids):
            # Each feature gets its own list of activations per token
            activation_store.setdefault(feat_id, z_np[:, i].tolist())

        # Optional: record which positions were in Top-K for those latents
        # Recompute hard mask using the same k:
        k_now = module._current_k()
        
        # We need to re-create the hard mask logic locally or use the one from models if importable
        # Since we have z on CPU now, let's just do it here.
        # Note: z is [seq, r] here.
        
        # _hard_topk_mask expects [..., r]
        hard_mask = _hard_topk_mask(z, k_now)    # [seq, r]
        hard_selected = hard_mask[:, latent_indices].numpy()
        
        for i, feat_id in enumerate(feature_ids):
            activation_store.setdefault(feat_id + "_in_topk", hard_selected[:, i].astype(int).tolist())

    return hook

def main():
    parser = argparse.ArgumentParser(description="Collect latent activations from TopKLoRA model")
    parser.add_argument("--config", type=str, required=True, help="Path to model config (YAML/JSON)")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts file (JSONL)")
    parser.add_argument("--features", type=str, required=True, help="Path to feature spec file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Path to output file (JSONL)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.config}...")
    cfg = OmegaConf.load(args.config)
    
    # Handle structure: if cfg has 'model' key, use it, else assume cfg is the model config
    if "model" in cfg:
        model_cfg = cfg.model
    else:
        model_cfg = cfg

    # 2. Load Model
    print("Loading model and tokenizer...")
    # init_model_tokenizer_fixed handles moving to device internally if hardcoded, 
    # but let's check if we need to enforce args.device.
    # The function in evals.py uses a global 'device' variable or hardcodes 'cpu' then moves to 'device'.
    # We might need to patch 'device' in src.evals or just let it be.
    # Looking at src/evals.py: 
    # device = 'cuda' if torch.cuda.is_available() else ...
    # model.to(device)
    # So it should be fine.
    
    model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(model_cfg)
    
    # Ensure model is on the requested device
    model.to(args.device)
    model.eval()

    # 3. Load Feature Spec
    print(f"Loading feature spec from {args.features}...")
    with open(args.features, 'r') as f:
        feature_specs = json.load(f)

    # Validate and build map
    module_to_features = defaultdict(list)
    all_feature_ids = []
    
    print("Validating features against wrapped modules...")
    available_modules = set(wrapped_modules.keys())
    
    for feat in feature_specs:
        m_name = feat["module_name"]
        if m_name not in available_modules:
            print(f"WARNING: Module '{m_name}' for feature '{feat['id']}' not found in wrapped modules.")
            # You might want to list available modules if this happens
            # print("Available modules:", list(available_modules))
            continue
        
        module_to_features[m_name].append(feat)
        all_feature_ids.append(feat["id"])

    if not all_feature_ids:
        print("Error: No valid features found to track. Exiting.")
        return

    print(f"Tracking {len(all_feature_ids)} features across {len(module_to_features)} modules.")

    # 4. Register Hooks
    shared_activation_store = {}
    hook_handles = []
    
    for module_name, features in module_to_features.items():
        module = wrapped_modules[module_name]
        # We pass the shared dictionary. The hook will write to it.
        # We must clear it before each forward pass.
        handle = module.register_forward_hook(
            make_hook(module_name, features, shared_activation_store)
        )
        hook_handles.append(handle)

    # 5. Process Prompts
    print(f"Processing prompts from {args.prompts}...")
    
    # Prepare output file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.prompts, 'r') as f_in, open(args.output, 'w') as f_out:
        # Count lines for tqdm
        lines = f_in.readlines()
        
        for i, line in enumerate(tqdm(lines, desc="Running prompts")):
            line = line.strip()
            if not line:
                continue
                
            try:
                prompt_data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {i+1}")
                continue
                
            text = prompt_data.get("text", "")
            ex_id = prompt_data.get("id", f"ex_{i:04d}")
            
            if not text:
                continue

            # Tokenize
            enc = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True
            )
            enc = {k: v.to(args.device) for k, v in enc.items()}
            
            # Decode tokens for logging
            input_ids = enc["input_ids"][0]
            token_strs = [tokenizer.decode([tid]) for tid in input_ids]
            
            # Reset store
            shared_activation_store.clear()
            
            # Run model
            with torch.no_grad():
                _ = model(**enc)
            
            # Construct record
            # shared_activation_store now populated by hooks
            
            record_features = {}
            for f_id in all_feature_ids:
                # Get activations, default to 0s if missing (e.g. module didn't run?)
                acts = shared_activation_store.get(f_id, [0.0] * len(token_strs))
                record_features[f_id] = acts
                
                # Also get topk info if available
                if f_id + "_in_topk" in shared_activation_store:
                    record_features[f_id + "_in_topk"] = shared_activation_store[f_id + "_in_topk"]

            record = {
                "id": ex_id,
                "text": text,
                "tokens": token_strs,
                "features": record_features
            }
            
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

    # 6. Cleanup
    for h in hook_handles:
        h.remove()
        
    print(f"Done! Results written to {args.output}")

if __name__ == "__main__":
    main()

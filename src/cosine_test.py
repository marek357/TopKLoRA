import os
import json
import re
import math
import glob
from pathlib import Path
import torch
import numpy as np
from safetensors.torch import load_file as load_safetensors


def _parse_range(fname: str):
    # supports ".../<start>_<end>.safetensors"
    m = re.search(r'(\d+)_(\d+)\.safetensors$', fname)
    if not m:
        raise ValueError(f"Cannot parse shard range from {fname}")
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise ValueError(f"Invalid shard range in {fname}")
    return a, b


def _read_width(module_dir: Path) -> int:
    cfg_path = module_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return int(cfg["width"])
    # fallback from filenames
    ends = [_parse_range(p.name)[1] for p in module_dir.glob("*.safetensors")]
    if not ends:
        raise FileNotFoundError(f"No shard files in {module_dir}")
    return max(ends) + 1


def _token_grid_shape(tokens: torch.Tensor):
    # Delphi caches "tokens" for building windows later; we just want n_tokens
    # Accept 1D [N] or 2D [B, L]; return (N, B, L)
    if tokens.ndim == 1:
        return tokens.numel(), None, None
    if tokens.ndim == 2:
        B, L = tokens.shape
        return B * L, B, L
    # Some setups store tokens only in one shard; others repeat. Either works.
    raise ValueError(f"Unexpected tokens shape: {tuple(tokens.shape)}")


def _split_locations(locs: torch.Tensor, *, B: int | None, L: int | None):
    """
    Returns (token_idx [nnz], latent_local [nnz], maybe_seq_pos tuple or None)
    Accepts:
      - [nnz, 2] or [2, nnz] -> (token_idx, latent_local)
      - [nnz, 3] or [3, nnz] -> (seq_idx, pos, latent_local)  → flatten to token_idx = seq_idx*L + pos
    """
    if locs.ndim != 2:
        raise ValueError("locations must be rank-2")
    # make it [nnz, D] for simplicity
    if locs.shape[0] in (2, 3) and locs.shape[0] < locs.shape[1]:
        locs = locs.T.contiguous()

    D = locs.shape[1]
    if D == 2:
        tok_idx = locs[:, 0].long()
        lat_loc = locs[:, 1].long()
        return tok_idx, lat_loc, None
    elif D == 3:
        if L is None:
            raise ValueError(
                "locations=[seq,pos,latent] but tokens aren’t 2D (need L).")
        seq_idx = locs[:, 0].long()
        pos = locs[:, 1].long()
        lat_loc = locs[:, 2].long()
        tok_idx = (seq_idx * L + pos).long()
        return tok_idx, lat_loc, (seq_idx, pos)
    else:
        raise ValueError(f"Unsupported locations shape: {tuple(locs.shape)}")


@torch.inference_mode()
def cosine_top20_for_s_in_module(
    module_dir: str | Path,
    # shape [width], same width as the module cache
    s: torch.Tensor,
    topk: int = 20,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    """
    Returns a list of dicts:
      { 'sim': float, 'token_idx': int, 'seq_idx': Optional[int], 'pos': Optional[int] }
    for the top-20 most similar cached token-activation vectors to s, within this module.
    """
    module_dir = Path(module_dir)
    width = _read_width(module_dir)

    s = s.detach().to(dtype=dtype, device="cpu").view(-1)
    if s.numel() != width:
        raise ValueError(f"s has dim {s.numel()} but module width is {width}")

    # find a shard to read tokens (any will do)
    shard_paths = sorted(glob.glob(str(module_dir / "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors shards in {module_dir}")

    # Try to read tokens from the first shard that has them
    tokens = None
    for p in shard_paths:
        d = load_safetensors(p, device="cpu")
        if "tokens" in d:
            tokens = d["tokens"]
            break
    if tokens is None:
        raise FileNotFoundError(
            "No 'tokens' key found in any shard; cannot deduce n_tokens.")

    n_tokens, B, L = _token_grid_shape(tokens)
    # accumulators (CPU)
    dot = torch.zeros(n_tokens, dtype=dtype)
    x2sum = torch.zeros(n_tokens, dtype=dtype)

    s_norm = torch.linalg.vector_norm(s).clamp_min(1e-12)

    # stream shards; each contributes to dot and x2 per token
    for p in shard_paths:
        start, end = _parse_range(p)
        s_chunk = s[start: end + 1]  # [chunk_width]

        d = load_safetensors(p, device="cpu")
        vals = d["activations"].to(dtype=dtype)    # [nnz]
        # [nnz, 2] or [nnz, 3] (or transposed)
        locs = d["locations"]

        tok_idx, lat_loc, _maybe_seqpos = _split_locations(locs, B=B, L=L)
        # map latent indices into this chunk
        # (lat_loc is already 0..(end-start))
        contrib = s_chunk[lat_loc] * vals          # [nnz]

        # scatter-add into accumulators
        dot.index_add_(0, tok_idx, contrib)
        x2sum.index_add_(0, tok_idx, vals * vals)

    # cosine = dot / (||s|| * ||x||)
    cos = dot / (s_norm * torch.sqrt(x2sum).clamp_min(1e-12))

    # top-K
    # (Avoid NaNs/inf just in case)
    cos = torch.nan_to_num(cos, nan=-1e9, neginf=-1e9, posinf=1e9)
    sims, idxs = torch.topk(cos, k=min(
        topk, cos.numel()), largest=True, sorted=True)

    out = []
    # if tokens were 2D, we can return (seq_idx, pos) too
    for sim, flat_idx in zip(sims.tolist(), idxs.tolist()):
        if B is not None and L is not None:
            seq_idx = flat_idx // L
            pos = flat_idx % L
            out.append({"sim": float(sim), "token_idx": int(
                flat_idx), "seq_idx": int(seq_idx), "pos": int(pos)})
        else:
            out.append({"sim": float(sim), "token_idx": int(
                flat_idx), "seq_idx": None, "pos": None})
    return out


@torch.inference_mode()
def cosine_top20_for_lora_B_rows(
    model,
    cache_root: str | Path,
    topk: int = 20,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    # Limit rows to search (since output_dim can be large)
    max_rows_per_module: int = None,
):
    """
    For each module with lora_B weights, search using each ROW of lora_B.
    Each row represents which r-dimensional pattern activates that output neuron.

    Returns: dict[module_name][row_idx] -> list[results]
    """
    cache_root = Path(cache_root)
    results = {}

    for name, module in model.named_modules():
        if not hasattr(module, 'lora_B'):
            continue

        # Get the lora_B weight matrix [output_dim, r]
        lora_B_weight = module.lora_B['default'].weight.data  # [output_dim, r]
        output_dim, r = lora_B_weight.shape

        # Convert module name to cache directory name
        # if "lora_module" in name:
        #     cache_name = name.replace(
        #         "base_model.model.model.", "").replace(".lora_module", "")
        # else:
        #     cache_name = name.replace("base_model.model.model.", "").replace(
        #         "base_model.model.", "")
        cache_name = name.replace(".lora_module", "")

        cache_name = cache_name + ".topk"
        cache_dir = cache_root / cache_name

        if not cache_dir.exists():
            print(
                f"Warning: Cache directory not found for {name}: {cache_dir}")
            continue

        try:
            width = _read_width(cache_dir)
        except Exception as e:
            print(f"Error reading width for {cache_dir}: {e}")
            continue

        # Check that r matches cache width
        if r != width:
            print(
                f"Dimension mismatch for {name}: lora_B r={r}, cache width={width}")
            continue

        print(
            f"Processing {name}: output_dim={output_dim}, r={r}, cache_dir={cache_dir}")

        # Limit number of rows to search if specified
        num_rows = min(max_rows_per_module,
                       output_dim) if max_rows_per_module else output_dim

        # Sample rows evenly if limiting
        if num_rows < output_dim:
            row_indices = torch.linspace(
                0, output_dim-1, num_rows, dtype=torch.long)
        else:
            row_indices = torch.arange(output_dim)

        module_results = []

        for row_idx in row_indices:
            # Extract row vector [r] - this is what we search with!
            s = lora_B_weight[row_idx, :].detach().to(
                dtype=dtype, device="cpu")  # [r]

            try:
                row_results = cosine_top20_for_s_in_module(
                    cache_dir, s, topk=topk, dtype=dtype, device=device)
                module_results.append({
                    'row_idx': int(row_idx),
                    'output_neuron': int(row_idx),
                    'matches': row_results
                })
            except Exception as e:
                print(f"Error searching row {row_idx} for {name}: {e}")

        results[cache_name] = module_results
        print(f"  Searched {len(module_results)} rows")

    return results


# # Usage:
# results = cosine_top20_for_lora_B_rows(
#     model,
#     './cache/delphi_causal_cache_512_4',
#     topk=20,
#     max_rows_per_module=10  # Just search first 10 output neurons per layer
# )


def summarize_lora_B_search_results(results, model, top_n=5):
    """
    Pretty-print summary of search results, showing top matches for each lora_B row
    """
    for module_name, row_results in results.items():
        print(f"\n{'='*80}")
        print(f"Module: {module_name}")
        print(f"{'='*80}")

        # Limit the number of rows to display
        for row_data in row_results[:10]:  # Show first 10 rows max
            row_idx = row_data['row_idx']
            output_neuron = row_data['output_neuron']
            matches = row_data['matches']

            if not matches:
                continue

            print(f"\n  Output neuron {output_neuron} (row {row_idx}):")
            print(f"  Top {min(top_n, len(matches))} matches:")

            for i, match in enumerate(matches[:top_n]):
                sim = match['sim']
                token_idx = match['token_idx']
                seq_idx = match.get('seq_idx')
                pos = match.get('pos')

                if seq_idx is not None and pos is not None:
                    print(
                        f"    {i+1}. Sim={sim:.4f} | Token {token_idx} (seq={seq_idx}, pos={pos})")
                else:
                    print(f"    {i+1}. Sim={sim:.4f} | Token {token_idx}")


def analyze_lora_B_patterns(results, topk_threshold=0.5):
    """
    Analyze which output neurons have the strongest matches to cached activations
    """
    for module_name, row_results in results.items():
        print(f"\n{'='*80}")
        print(f"Module: {module_name}")
        print(f"{'='*80}")

        # Find rows with highest similarity scores
        high_sim_rows = []

        for row_data in row_results:
            row_idx = row_data['row_idx']
            matches = row_data['matches']

            if matches and matches[0]['sim'] > topk_threshold:
                high_sim_rows.append({
                    'row_idx': row_idx,
                    'max_sim': matches[0]['sim'],
                    'top_match': matches[0]
                })

        # Sort by similarity
        high_sim_rows.sort(key=lambda x: x['max_sim'], reverse=True)

        print(
            f"  Found {len(high_sim_rows)} output neurons with similarity > {topk_threshold}")
        print(f"  Top 10 strongly-matching output neurons:")

        for i, row_info in enumerate(high_sim_rows[:10]):
            row_idx = row_info['row_idx']
            max_sim = row_info['max_sim']
            top_match = row_info['top_match']

            print(f"    Output {row_idx}: max_sim={max_sim:.4f}, "
                  f"best match at token {top_match['token_idx']}")


def aggregate_search_statistics(results):
    """
    Compute aggregate statistics across all searches
    """
    all_sims = []
    module_stats = {}

    for module_name, row_results in results.items():
        module_sims = []

        for row_data in row_results:
            matches = row_data['matches']
            if matches:
                # Just take top similarity for each row
                module_sims.append(matches[0]['sim'])

        if module_sims:
            module_stats[module_name] = {
                'mean_top_sim': np.mean(module_sims),
                'max_top_sim': np.max(module_sims),
                'min_top_sim': np.min(module_sims),
                'std_top_sim': np.std(module_sims),
                'num_rows': len(module_sims)
            }
            all_sims.extend(module_sims)

    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)

    for module_name, stats in module_stats.items():
        print(f"\n{module_name}:")
        print(f"  Mean top similarity: {stats['mean_top_sim']:.4f}")
        print(f"  Max top similarity:  {stats['max_top_sim']:.4f}")
        print(f"  Min top similarity:  {stats['min_top_sim']:.4f}")
        print(f"  Std top similarity:  {stats['std_top_sim']:.4f}")

    if all_sims:
        print(f"\nOverall statistics across all {len(all_sims)} rows:")
        print(f"  Mean: {np.mean(all_sims):.4f}")
        print(f"  Median: {np.median(all_sims):.4f}")
        print(f"  95th percentile: {np.percentile(all_sims, 95):.4f}")


# # Usage example:
# if __name__ == '__main__':
#     # Or just test with empty model for now
#     print("Ready to run with model and cache root")
#     # Assuming you have your model loaded with the fixed weights

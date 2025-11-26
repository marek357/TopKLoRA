import json
import numpy as np
import matplotlib.pyplot as plt


def load_example(jsonl_path, ex_id=None, index=0):
    """Load one record from the JSONL file."""
    with open(jsonl_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if ex_id is not None:
        for rec in records:
            if rec["id"] == ex_id:
                return rec
        raise ValueError(f"Example id {ex_id} not found")
    else:
        return records[index]


# --- configure these ---
jsonl_path = "outputs/activations_18layer_4096_32_selected.jsonl"
example_id = "ex_pii_address"   # or None + use index
# -----------------------

rec = load_example(jsonl_path, ex_id=example_id)

tokens = rec["tokens"]
features = rec["features"]

# pick which feature ids to show (skip the *_in_topk masks)
feature_ids = [
    "pii_struct_digits_610",
    "address_formatting_664",
    "phone_digits_1933",
    "url_structure_1309",
]

# build matrix: rows = features, cols = tokens
mat = np.vstack([np.array(features[f_id], dtype=float)
                for f_id in feature_ids])

# optional: normalize per-feature so colors are comparable
# (prettier for posters)
mat_norm = mat / (mat.max(axis=1, keepdims=True) + 1e-8)

plt.figure(figsize=(len(tokens) * 0.25, len(feature_ids) * 0.6))
plt.imshow(mat_norm, aspect="auto")
plt.colorbar(label="Normalized activation")

plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
plt.yticks(ticks=range(len(feature_ids)), labels=feature_ids)

plt.tight_layout()
plt.show()

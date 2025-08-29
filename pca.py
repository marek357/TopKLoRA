import pickle
from sklearn.decomposition import PCA
from dataclasses import dataclass
from src.evals import init_model_tokenizer
import torch

THRESHOLD_FOR_DEAD_NEURON = 5
ADAPTER_PATH = "/scratch/network/ssd/marek/lora_interp/experiments/gemma-2-2b_topk_dpo_r1024_k8/final_adapter"


def load_neuron_checkpoint(module_name):
    with open(f'cache/neuron_records/c4/{module_name}_neuron_record_checkpoint.pkl', 'rb') as f:
        module_record = pickle.load(f)
    return module_record


def find_dead_neurons(module_name):
    module_record = load_neuron_checkpoint(module_name)
    dead_neurons = []
    active_neurons = []
    for neuron_id, neuron_record in module_record.items():
        if len(neuron_record.most_positive_activation_records) < THRESHOLD_FOR_DEAD_NEURON:
            dead_neurons.append(neuron_id)
        else:
            active_neurons.append(neuron_id)
    return dead_neurons, active_neurons


@dataclass
class ModelCFG:
    base_model: str
    adapter_checkpoint_dir: str
    k: int


if __name__ == "__main__":
    model_cfg = ModelCFG(
        base_model="/scratch/network/ssd/marek/lora_interp/cache/tempartefacts/google/gemma-2-2b_sft",
        adapter_checkpoint_dir=ADAPTER_PATH,
        k=8,
    )
    model, tokenizer, adapters_map = init_model_tokenizer(model_cfg, True)
    for module_name, adapter in adapters_map.items():
        print(f"Processing module: {module_name}")
        dead_neurons, active_neurons = find_dead_neurons(
            module_name.removeprefix("base_model.model.")
        )
        print(f"Total dead neurons: {len(dead_neurons)}")
        print(f"Total active neurons: {len(active_neurons)}")
        b_matrix = adapter.lora_module.lora_B['default'].weight
        b_matrix = b_matrix.detach().cpu()
        mask = torch.ones(
            b_matrix.size(1), dtype=torch.bool, device=b_matrix.device
        )
        mask[dead_neurons] = False
        b_matrix_new = b_matrix[:, mask]
        # Xc = b_matrix_new - b_matrix_new.mean(dim=0, keepdim=True)
        # n = Xc.size(0)

        # # 2) PCA for top k components
        # k = 30
        # U, S, Vt = torch.pca_lowrank(Xc, q=k)

        # # 3) explained variance for each PC: λ_i = S_i² / (n − 1)
        # explained_variance = S**2 / (n - 1)               # shape (k,)

        # # 4) fraction of total variance:
        # #    total variance = sum of variances of each original feature
        # total_var = Xc.var(dim=0, unbiased=True).sum()    # scalar
        # explained_variance_ratio = explained_variance / total_var
X_np = b_matrix_new.cpu().numpy()
pca = PCA(n_components=10)
pca.fit(X_np)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print('-' * 80)
print('\n')

from peft.tuners.lora import LoraLayer
import torch.nn.functional as F
import torch.nn as nn
import torch


# ────────────────────────────────────────────────────────────────────────────
# Top-k LoRA module  ── self-contained for convenience
# ────────────────────────────────────────────────────────────────────────────
def _first_weight(md: nn.ModuleDict):
    return next(iter(md.values())).weight


class TopKLoRALinear(nn.Module):
    def __init__(
            self,
            base: LoraLayer, *,
            layer_name: str, r,
            alpha, k: int
    ):
        super().__init__()
        # store for unwrapping
        self.lora_module = base
        # frozen quant/FP layer
        self.base_layer = base.base_layer
        # LoRA params
        self.A = _first_weight(base.lora_A)
        self.B = _first_weight(base.lora_B)
        # support dict or int
        r_val = r["default"] if isinstance(r, dict) else r
        alpha_val = alpha["default"] if isinstance(alpha, dict) else alpha
        self.r = int(r_val)
        self.k = int(k)
        self.scale = alpha_val / r_val
        self.layer_name = layer_name
        print(f"Using a TopK LoRA Adapter with r: {self.r}, k: {self.k}")

    def forward(self, x: torch.Tensor):
        # match dtype for mixed precision
        A = self.A.to(dtype=x.dtype, device=x.device)
        B = self.B.to(dtype=x.dtype, device=x.device)
        z = F.linear(x, A)
        if self.k < self.r:
            thresh = z.abs().topk(self.k, dim=-1)[0][..., -1:]
            z = torch.where(z.abs() >= thresh, z, z.new_zeros(()))
        out = self.base_layer(x)
        out += F.linear(z, B) * self.scale
        return out

import torch


# ────────────────────────────────────────────────────────────────────────────
# Top-k LoRA module  ── self-contained for convenience
# ────────────────────────────────────────────────────────────────────────────


class TopKLoRALinear(torch.nn.Module):
    """
    Wraps a frozen Linear layer with trainable LoRA A/B matrices and Top-k
    magnitude sparsification on z = A·x.
    """

    def __init__(self, base: torch.nn.Linear, r=8, alpha=16, k=4):
        super().__init__()
        self.weight, self.bias = base.weight, base.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.r, self.k, self.scale = r, k, alpha / r
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features))
        self.B = torch.nn.Parameter(torch.empty(base.out_features, r))
        torch.nn.init.kaiming_uniform_(self.A, a=torch.sqrt(torch.tensor(5.0)))
        torch.nn.init.zeros_(self.B)

    def forward(self, x):
        z = torch.nn.functional.linear(x, self.A)            # (…, r)
        if self.k < self.r:
            thr = torch.topk(z.abs(), self.k, dim=-1)[0][..., -1:]
            z = torch.where(z.abs() >= thr, z, torch.zeros_like(z))
        return (torch.nn.functional.linear(x, self.weight, self.bias)
                + torch.nn.functional.linear(z, self.B) * self.scale)

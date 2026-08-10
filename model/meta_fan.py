import torch
import torch.nn as nn


class FAN(nn.Module):
    """Feature-wise Adaptive Normalization.

    Per-feature affine transform:  X̃_i = γ_i · X_i + β_i
    Default initialisation is identity (γ=1, β=0).
    Features in skip_indices are kept as identity (not transformed).
    """

    def __init__(self, num_features: int, skip_indices: list = None):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.skip_indices = skip_indices or []

    def forward(self, x):
        # x: (B, T, F)  — γ, β broadcast over B and T
        if self.skip_indices:
            gamma = self.gamma.clone()
            beta = self.beta.clone()
            for idx in self.skip_indices:
                gamma[idx] = 1.0
                beta[idx] = 0.0
            return gamma * x + beta
        return self.gamma * x + self.beta

    def set_params(self, gamma: torch.Tensor, beta: torch.Tensor):
        """Overwrite γ/β with externally generated values (e.g. from MetaGenerator)."""
        self.gamma.data.copy_(gamma.detach())
        self.beta.data.copy_(beta.detach())


class MetaGenerator(nn.Module):
    """Conditioned Meta-Generator (hypernetwork).

    Per-feature:  (s_i, d_i) → (γ_i^(0), β_i^(0))
    γ uses residual connection around 1 (identity as default).
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_gamma = nn.Linear(hidden_dim, 1)
        self.head_beta = nn.Linear(hidden_dim, 1)

    def forward(self, descriptors: torch.Tensor):
        """
        Args:
            descriptors: (F, 2) — row i is (s_i, d_i)
        Returns:
            gamma_init: (F,)
            beta_init:  (F,)
        """
        h = self.shared(descriptors)                       # (F, hidden_dim)
        gamma_init = 1.0 + self.head_gamma(h).squeeze(-1)  # residual around 1
        beta_init = self.head_beta(h).squeeze(-1)
        return gamma_init, beta_init


class MetaFANModel(nn.Module):
    """FAN layer + arbitrary backbone, composed as one forward pass.

    Data flow:  X → FAN → backbone → prediction
    The MetaGenerator lives outside this module (used in MAML loop).
    """

    def __init__(self, backbone: nn.Module, num_features: int, skip_indices: list = None):
        super().__init__()
        self.fan = FAN(num_features, skip_indices=skip_indices)
        self.backbone = backbone

    def forward(self, x):
        # x: (B, T, F)
        return self.backbone(self.fan(x))

    # ---- parameter groups (for MAML inner/outer loop separation) ----

    def fan_parameters(self):
        """Only FAN γ and β — adapted in the inner loop."""
        return [self.fan.gamma, self.fan.beta]

    def backbone_parameters(self):
        """Backbone parameters — updated in the outer loop."""
        return list(self.backbone.parameters())

    # ---- convenience freeze / unfreeze ----

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_fan(self):
        self.fan.gamma.requires_grad = False
        self.fan.beta.requires_grad = False

    def unfreeze_fan(self):
        self.fan.gamma.requires_grad = True
        self.fan.beta.requires_grad = True

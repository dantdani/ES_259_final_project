"""
model_expert.py

Model C: One expert per region.

Each expert is a smaller residual MLP trained only on data from its region.
At inference, the input is routed to the appropriate expert.
"""

import torch
import torch.nn as nn
from model_global import ResidualBlock


class ExpertIKModel(nn.Module):
    """A single expert model (smaller than the global model).

    Parameters
    ----------
    input_dim   : int – pose dimensionality
    output_dim  : int – 12 (sin/cos)
    hidden_dim  : int – width
    num_blocks  : int – number of residual blocks
    dropout     : float
    norm        : str – 'layer' or 'batch'
    """

    def __init__(self, input_dim: int = 7, output_dim: int = 12,
                 hidden_dim: int = 256, num_blocks: int = 4,
                 dropout: float = 0.0, norm: str = "layer"):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout, norm)
              for _ in range(num_blocks)]
        )
        NormClass = nn.LayerNorm if norm == "layer" else nn.BatchNorm1d
        self.output_head = nn.Sequential(
            NormClass(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_head(x)


class ExpertEnsemble(nn.Module):
    """Manages multiple expert models, one per region.

    At inference:
      1. Determine region from input position
      2. Route to the corresponding expert

    Parameters
    ----------
    num_experts : int
    expert_kwargs : dict passed to ExpertIKModel
    """

    def __init__(self, num_experts: int = 8, **expert_kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertIKModel(**expert_kwargs) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor,
                region_ids: torch.Tensor) -> torch.Tensor:
        """Route each sample to its region expert.

        Parameters
        ----------
        x          : (N, D) inputs
        region_ids : (N,) int region labels

        Returns
        -------
        out : (N, 12)
        """
        N = x.shape[0]
        out = torch.zeros(N, self.experts[0].output_head[-1].out_features,
                          device=x.device, dtype=x.dtype)

        for r in range(self.num_experts):
            mask = region_ids == r
            if mask.any():
                out[mask] = self.experts[r](x[mask])
        return out

    def forward_expert(self, x: torch.Tensor, region_id: int) -> torch.Tensor:
        """Forward through a single expert (convenience for training)."""
        return self.experts[region_id](x)


def build_expert_ensemble(num_experts=8, input_dim=7, output_dim=12,
                          hidden_dim=256, num_blocks=4,
                          dropout=0.0, norm="layer"):
    model = ExpertEnsemble(
        num_experts=num_experts,
        input_dim=input_dim, output_dim=output_dim,
        hidden_dim=hidden_dim, num_blocks=num_blocks,
        dropout=dropout, norm=norm,
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_per = sum(p.numel() for p in model.experts[0].parameters() if p.requires_grad)
    print(f"ExpertEnsemble: {n:,} total params ({n_per:,} per expert, "
          f"{num_experts} experts)")
    return model

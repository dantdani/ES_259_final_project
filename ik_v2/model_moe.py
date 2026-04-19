"""
model_moe.py

Model D: Mixture-of-Experts with a gating network.

  - N expert subnetworks
  - A gating network that produces expert weights from pose input
  - Final prediction = weighted sum of expert outputs
  - Supports both soft and top-k hard routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_global import ResidualBlock


class MoEExpert(nn.Module):
    """A single MoE expert (lightweight)."""

    def __init__(self, input_dim=7, output_dim=12,
                 hidden_dim=256, num_blocks=3,
                 dropout=0.0, norm="layer"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            *[ResidualBlock(hidden_dim, dropout, norm)
              for _ in range(num_blocks)],
            nn.LayerNorm(hidden_dim) if norm == "layer" else nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    """Produces expert weights from pose input."""

    def __init__(self, input_dim=7, num_experts=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x):
        return self.net(x)  # raw logits


class MixtureOfExperts(nn.Module):
    """Mixture-of-Experts for IK.

    Parameters
    ----------
    num_experts    : int
    input_dim      : int
    output_dim     : int
    expert_hidden  : int – hidden dim per expert
    expert_blocks  : int – residual blocks per expert
    gate_hidden    : int – hidden dim for gating network
    top_k          : int – if > 0, use top-k hard routing; 0 = soft routing
    dropout        : float
    norm           : str
    """

    def __init__(self, num_experts: int = 8, input_dim: int = 7,
                 output_dim: int = 12, expert_hidden: int = 256,
                 expert_blocks: int = 3, gate_hidden: int = 128,
                 top_k: int = 0, dropout: float = 0.0,
                 norm: str = "layer"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            MoEExpert(input_dim, output_dim, expert_hidden,
                      expert_blocks, dropout, norm)
            for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(input_dim, num_experts, gate_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, input_dim)

        Returns
        -------
        output : (N, output_dim) – weighted combination of expert outputs
        """
        gate_logits = self.gate(x)  # (N, num_experts)

        if self.top_k > 0 and self.top_k < self.num_experts:
            # Top-k hard routing
            topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
            gate_weights = torch.zeros_like(gate_logits).scatter_(
                1, topk_idx, F.softmax(topk_vals, dim=-1))
        else:
            gate_weights = F.softmax(gate_logits, dim=-1)  # (N, E)

        # Compute all expert outputs
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (N, E, output_dim)

        # Weighted combination
        weights = gate_weights.unsqueeze(-1)  # (N, E, 1)
        output = (expert_outputs * weights).sum(dim=1)  # (N, output_dim)

        return output

    def get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmaxed gate weights for analysis."""
        return F.softmax(self.gate(x), dim=-1)


def build_moe_model(num_experts=8, input_dim=7, output_dim=12,
                    expert_hidden=256, expert_blocks=3,
                    gate_hidden=128, top_k=0,
                    dropout=0.0, norm="layer"):
    model = MixtureOfExperts(
        num_experts, input_dim, output_dim,
        expert_hidden, expert_blocks, gate_hidden,
        top_k, dropout, norm,
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_gate = sum(p.numel() for p in model.gate.parameters())
    n_experts = sum(p.numel() for p in model.experts.parameters())
    print(f"MoE: {n:,} total params  (gate={n_gate:,}, "
          f"experts={n_experts:,}, {num_experts} experts)")
    return model

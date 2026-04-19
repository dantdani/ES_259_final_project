"""
model.py

Residual MLP for UR5e inverse kinematics.

Input:  12D full pose  [x, y, z, r11..r33]
Output: 12D sin/cos    [sin(q1), cos(q1), ..., sin(q6), cos(q6)]
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block: LayerNorm -> GELU -> Linear -> LayerNorm -> GELU -> Linear + skip."""

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class IKResidualMLP(nn.Module):
    """Residual MLP for inverse kinematics.

    Architecture:
        Linear(input_dim -> hidden_dim) -> GELU
        -> N x ResidualBlock(hidden_dim)
        -> LayerNorm -> Linear(hidden_dim -> hidden_dim//2) -> GELU
        -> Linear(hidden_dim//2 -> output_dim)

    Parameters
    ----------
    input_dim  : int – number of input features (12 for full pose)
    output_dim : int – number of output features (12 for sin/cos encoding)
    hidden_dim : int – width of the residual blocks
    num_blocks : int – number of residual blocks
    dropout    : float – dropout probability inside residual blocks
    """

    def __init__(self, input_dim=12, output_dim=12, hidden_dim=256,
                 num_blocks=4, dropout=0.0):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        # Residual trunk
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_head(x)
        return x


def build_model(input_dim=12, output_dim=12, hidden_dim=256,
                num_blocks=4, dropout=0.0):
    """Build the IKResidualMLP and print parameter count."""
    model = IKResidualMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel architecture:\n{model}")
    print(f"Trainable parameters: {num_params:,}\n")
    return model

"""
model_global.py

Model A and Model B: Global residual MLP (stronger than v1).

Model A: Global baseline — pose features only
Model B: Global + region ID — pose features + learned region embedding
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block with configurable norm."""

    def __init__(self, dim: int, dropout: float = 0.0,
                 norm: str = "layer"):
        super().__init__()
        NormClass = nn.LayerNorm if norm == "layer" else nn.BatchNorm1d
        self.block = nn.Sequential(
            NormClass(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            NormClass(dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class GlobalIKModel(nn.Module):
    """Stronger global residual MLP for IK.

    Supports optional region embedding (Model B) when num_regions > 0.

    Parameters
    ----------
    input_dim    : int   – pose dimensionality (7 for quat, 12 for rotmat, 6 for axisangle)
    output_dim   : int   – 12 (sin/cos pairs)
    hidden_dim   : int   – width
    num_blocks   : int   – number of residual blocks
    dropout      : float – dropout rate
    norm         : str   – 'layer' or 'batch'
    num_regions  : int   – if > 0, add a learned region embedding (Model B)
    region_embed_dim : int – size of the region embedding
    """

    def __init__(self, input_dim: int = 7, output_dim: int = 12,
                 hidden_dim: int = 512, num_blocks: int = 6,
                 dropout: float = 0.0, norm: str = "layer",
                 num_regions: int = 0, region_embed_dim: int = 16):
        super().__init__()

        self.num_regions = num_regions

        # Region embedding
        if num_regions > 0:
            self.region_embed = nn.Embedding(num_regions, region_embed_dim)
            proj_in = input_dim + region_embed_dim
        else:
            self.region_embed = None
            proj_in = input_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in, hidden_dim),
            nn.SiLU(),
        )

        # Residual trunk
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout, norm)
              for _ in range(num_blocks)]
        )

        # Output head
        NormClass = nn.LayerNorm if norm == "layer" else nn.BatchNorm1d
        self.output_head = nn.Sequential(
            NormClass(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor,
                region_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, input_dim) pose features
        region_ids : (N,) int tensor – region labels (only if num_regions > 0)
        """
        if self.region_embed is not None:
            if region_ids is None:
                raise ValueError("region_ids required when num_regions > 0")
            emb = self.region_embed(region_ids)  # (N, embed_dim)
            x = torch.cat([x, emb], dim=-1)

        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_head(x)
        return x


def build_global_model(input_dim=7, output_dim=12, hidden_dim=512,
                       num_blocks=6, dropout=0.0, norm="layer",
                       num_regions=0, region_embed_dim=16):
    model = GlobalIKModel(input_dim, output_dim, hidden_dim, num_blocks,
                          dropout, norm, num_regions, region_embed_dim)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GlobalIKModel: {n:,} trainable parameters  "
          f"(hidden={hidden_dim}, blocks={num_blocks}, regions={num_regions})")
    return model

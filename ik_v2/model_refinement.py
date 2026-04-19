"""
model_refinement.py

Iterative refinement architecture for IK.

Stage 1: Initial predictor estimates joints from pose
Stage 2+: Refinement networks predict corrections:
    q1 = q0 + DeltaNet(pose, q0)
    q2 = q1 + DeltaNet(pose, q1)  [optional shared weights]

Operates in sin/cos space throughout.
"""

import torch
import torch.nn as nn
from model_global import ResidualBlock


class InitialPredictor(nn.Module):
    """Stage-1 network: pose -> initial sin/cos estimate."""

    def __init__(self, input_dim=7, output_dim=12,
                 hidden_dim=512, num_blocks=4,
                 dropout=0.0, norm="layer"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            *[ResidualBlock(hidden_dim, dropout, norm)
              for _ in range(num_blocks)],
            nn.LayerNorm(hidden_dim) if norm == "layer" else nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class RefinementBlock(nn.Module):
    """Stage-2 correction network: (pose, current_sincos) -> delta_sincos."""

    def __init__(self, pose_dim=7, sincos_dim=12,
                 hidden_dim=256, num_blocks=3,
                 dropout=0.0, norm="layer"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim + sincos_dim, hidden_dim),
            nn.SiLU(),
            *[ResidualBlock(hidden_dim, dropout, norm)
              for _ in range(num_blocks)],
            nn.LayerNorm(hidden_dim) if norm == "layer" else nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, sincos_dim),
        )

    def forward(self, pose: torch.Tensor,
                current_sincos: torch.Tensor) -> torch.Tensor:
        """Predict a correction delta."""
        x = torch.cat([pose, current_sincos], dim=-1)
        return self.net(x)


class IterativeIKModel(nn.Module):
    """Full iterative refinement model.

    Parameters
    ----------
    pose_dim       : int – input pose dimension
    output_dim     : int – 12 (sin/cos)
    init_hidden    : int – hidden dim for initial predictor
    init_blocks    : int – blocks for initial predictor
    refine_hidden  : int – hidden dim for refinement blocks
    refine_blocks  : int – blocks for refinement
    num_refine     : int – number of refinement steps (1-3)
    share_refine   : bool – share weights across refinement steps
    dropout        : float
    norm           : str
    """

    def __init__(self, pose_dim: int = 7, output_dim: int = 12,
                 init_hidden: int = 512, init_blocks: int = 4,
                 refine_hidden: int = 256, refine_blocks: int = 3,
                 num_refine: int = 2, share_refine: bool = True,
                 dropout: float = 0.0, norm: str = "layer"):
        super().__init__()
        self.num_refine = num_refine

        self.initial = InitialPredictor(
            pose_dim, output_dim, init_hidden, init_blocks, dropout, norm)

        if share_refine:
            refine = RefinementBlock(
                pose_dim, output_dim, refine_hidden, refine_blocks,
                dropout, norm)
            self.refine_steps = nn.ModuleList([refine] * num_refine)
        else:
            self.refine_steps = nn.ModuleList([
                RefinementBlock(pose_dim, output_dim, refine_hidden,
                                refine_blocks, dropout, norm)
                for _ in range(num_refine)
            ])

    def forward(self, pose: torch.Tensor,
                return_intermediates: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        pose : (N, pose_dim)
        return_intermediates : if True, return list of all stage outputs

        Returns
        -------
        output : (N, 12) final sin/cos prediction
        intermediates : list of (N, 12) if return_intermediates
        """
        q = self.initial(pose)  # (N, 12) initial estimate
        intermediates = [q] if return_intermediates else None

        for refine in self.refine_steps:
            delta = refine(pose, q)
            q = q + delta
            if return_intermediates:
                intermediates.append(q)

        if return_intermediates:
            return q, intermediates
        return q


def build_refinement_model(pose_dim=7, output_dim=12,
                           init_hidden=512, init_blocks=4,
                           refine_hidden=256, refine_blocks=3,
                           num_refine=2, share_refine=True,
                           dropout=0.0, norm="layer"):
    model = IterativeIKModel(
        pose_dim, output_dim, init_hidden, init_blocks,
        refine_hidden, refine_blocks, num_refine, share_refine,
        dropout, norm)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_init = sum(p.numel() for p in model.initial.parameters())
    n_ref = sum(p.numel() for p in model.refine_steps.parameters())
    print(f"IterativeIK: {n:,} params  (init={n_init:,}, "
          f"refine={n_ref:,}, {num_refine} steps, "
          f"shared={share_refine})")
    return model

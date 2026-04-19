"""
fk_loss.py

Differentiable FK consistency loss for the IK pipeline.

Given predicted joint angles, runs forward kinematics in PyTorch
and compares the resulting end-effector pose to the target pose.

Loss = lambda_joint * L_joint
     + lambda_pos   * L_pos
     + lambda_rot   * L_rot
"""

import torch
import torch.nn as nn
import numpy as np

# Import UR5e constants directly so we can build the PyTorch FK once
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import build_ur5e_model


# ============================================================================
# Build static FK tensors (screws + M) at module load time
# ============================================================================

_UR5E = build_ur5e_model()
_W_NP = _UR5E["w"].astype(np.float32)   # (6, 3)
_V_NP = _UR5E["v"].astype(np.float32)   # (6, 3)
_M_NP = _UR5E["M"].astype(np.float32)   # (4, 4)


# ============================================================================
# PyTorch FK  (batched)
# ============================================================================

def _skew_batch(v: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrix. v: (N, 3) -> (N, 3, 3)"""
    N = v.shape[0]
    zero = torch.zeros(N, device=v.device, dtype=v.dtype)
    return torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=-1).reshape(N, 3, 3)


def _screw_exp_batch(w: torch.Tensor, v: torch.Tensor,
                     theta: torch.Tensor) -> torch.Tensor:
    """Batched screw exponential.

    w     : (3,) unit axis  (broadcast)
    v     : (3,) linear velocity  (broadcast)
    theta : (N,) joint angle

    Returns (N, 4, 4) transforms
    """
    N = theta.shape[0]
    # Expand w, v to (N, 3)
    w_exp = w.unsqueeze(0).expand(N, -1)
    v_exp = v.unsqueeze(0).expand(N, -1)
    th = theta.unsqueeze(-1)  # (N, 1)

    w_hat = _skew_batch(w_exp)  # (N, 3, 3)
    w_hat_sq = torch.bmm(w_hat, w_hat)  # (N, 3, 3)

    eye = torch.eye(3, device=theta.device, dtype=theta.dtype).unsqueeze(0)

    sin_th = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)  # (N,1,1)
    cos_th = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)

    R = eye + sin_th * w_hat + (1.0 - cos_th) * w_hat_sq  # (N,3,3)

    G = eye * th.unsqueeze(-1) + (1.0 - cos_th) * w_hat + (th.unsqueeze(-1) - sin_th) * w_hat_sq
    p = torch.bmm(G, v_exp.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    T = torch.zeros(N, 4, 4, device=theta.device, dtype=theta.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    T[:, 3, 3] = 1.0
    return T


def forward_kinematics_batch(joints: torch.Tensor) -> torch.Tensor:
    """Batch FK using Product of Exponentials (space frame).

    Parameters
    ----------
    joints : (N, 6) joint angles in radians

    Returns
    -------
    T : (N, 4, 4) end-effector transforms
    """
    device = joints.device
    dtype = joints.dtype
    N = joints.shape[0]

    # Move static screw data to same device (lazily cached via module globals)
    w_all = torch.tensor(_W_NP, device=device, dtype=dtype)   # (6, 3)
    v_all = torch.tensor(_V_NP, device=device, dtype=dtype)   # (6, 3)
    M = torch.tensor(_M_NP, device=device, dtype=dtype)       # (4, 4)

    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(N, -1, -1).clone()

    for i in range(6):
        Ti = _screw_exp_batch(w_all[i], v_all[i], joints[:, i])
        T = torch.bmm(T, Ti)

    T = torch.bmm(T, M.unsqueeze(0).expand(N, -1, -1))
    return T


# ============================================================================
# Decode sin/cos to joint angles (differentiable)
# ============================================================================

def decode_sincos_torch(sincos: torch.Tensor) -> torch.Tensor:
    """Decode (N, 12) sin/cos pairs to (N, 6) joint angles via atan2."""
    joints = torch.stack([
        torch.atan2(sincos[:, 2*i], sincos[:, 2*i+1])
        for i in range(6)
    ], dim=-1)
    return joints


# ============================================================================
# FK Consistency Loss
# ============================================================================

class FKConsistencyLoss(nn.Module):
    """Combined loss: sin/cos regression + FK position + FK orientation.

    Parameters
    ----------
    lambda_joint : float – weight on sin/cos MSE
    lambda_pos   : float – weight on position error (mm)
    lambda_rot   : float – weight on rotation error
    pos_normalize: float – divide position error by this to scale to ~1.0
    """

    def __init__(self, lambda_joint: float = 1.0,
                 lambda_pos: float = 0.01,
                 lambda_rot: float = 0.1,
                 pos_normalize: float = 100.0):
        super().__init__()
        self.lambda_joint = lambda_joint
        self.lambda_pos = lambda_pos
        self.lambda_rot = lambda_rot
        self.pos_normalize = pos_normalize
        self.mse = nn.MSELoss()

    def forward(self, pred_sincos: torch.Tensor,
                target_sincos: torch.Tensor,
                target_pose: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        pred_sincos   : (N, 12) predicted sin/cos
        target_sincos : (N, 12) ground truth sin/cos
        target_pose   : (N, D) target pose (first 3 = position, rest = orientation)

        Returns
        -------
        dict with 'total', 'joint', 'pos', 'rot' losses
        """
        # 1) Sin/cos regression loss
        L_joint = self.mse(pred_sincos, target_sincos)

        # 2) FK reconstruction
        pred_joints = decode_sincos_torch(pred_sincos)
        pred_T = forward_kinematics_batch(pred_joints)  # (N, 4, 4)

        pred_pos = pred_T[:, :3, 3]         # (N, 3)
        target_pos = target_pose[:, :3]      # (N, 3)

        L_pos = self.mse(pred_pos / self.pos_normalize,
                         target_pos / self.pos_normalize)

        # 3) Rotation loss: Frobenius norm of (R_pred - R_target)
        pred_R = pred_T[:, :3, :3]  # (N, 3, 3)

        # Reconstruct target rotation from the pose representation
        D = target_pose.shape[1]
        if D == 7:
            # quaternion: reconstruct R from (qw, qx, qy, qz)
            target_R = self._quat_to_rotmat(target_pose[:, 3:7])
        elif D == 12:
            # rotation matrix: reshape
            target_R = target_pose[:, 3:].reshape(-1, 3, 3)
        elif D == 6:
            # axis-angle: not implemented in differentiable form yet
            # Use position-only FK loss
            target_R = pred_R.detach()  # no rot gradient
        else:
            target_R = pred_R.detach()

        L_rot = self.mse(pred_R, target_R)

        total = (self.lambda_joint * L_joint
                 + self.lambda_pos * L_pos
                 + self.lambda_rot * L_rot)

        return {
            "total": total,
            "joint": L_joint,
            "pos": L_pos,
            "rot": L_rot,
        }

    @staticmethod
    def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """Batch quaternion [w,x,y,z] -> (N,3,3)."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)
        return R

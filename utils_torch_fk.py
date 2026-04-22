"""
utils.py

Shared utilities for the UR5e IK learning pipeline:
- FK helpers (skew, Rodrigues, screw exponential)
- UR5e model builder
- Forward kinematics (Product of Exponentials)
- Sin/cos encoding and decoding for joint angles
- Feature extraction from homogeneous transforms

Units: millimeters (consistent with MATLAB code).
"""

import numpy as np

# ============================================================================
# Joint limits (radians)
# ============================================================================
DEFAULT_JOINT_LIMITS = [
    (-np.pi, np.pi),  # joint 1
    (-np.pi, np.pi),  # joint 2
    (-np.pi, np.pi),  # joint 3
    (-np.pi, np.pi),  # joint 4
    (-np.pi, np.pi),  # joint 5
    (-np.pi, np.pi),  # joint 6
]

JOINT_COLS = [f"q{i+1}" for i in range(6)]
POSITION_COLS = ["x", "y", "z"]
ROTATION_COLS = [
    "r11", "r12", "r13",
    "r21", "r22", "r23",
    "r31", "r32", "r33",
]
POSE_COLS = POSITION_COLS + ROTATION_COLS

SINCOS_COLS = []
for i in range(1, 7):
    SINCOS_COLS += [f"sin_q{i}", f"cos_q{i}"]


# ============================================================================
# FK helper functions (match MATLAB implementation exactly)
# ============================================================================

def skew(v):
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def exp_rot(w_hat, theta):
    """Rodrigues rotation: exp([w]*theta)."""
    return (np.eye(3)
            + np.sin(theta) * w_hat
            + (1.0 - np.cos(theta)) * (w_hat @ w_hat))


def g_func(w_hat, theta):
    """Translation helper for the screw exponential."""
    return (np.eye(3) * theta
            + (1.0 - np.cos(theta)) * w_hat
            + (theta - np.sin(theta)) * (w_hat @ w_hat))


def screw_exp(w, v, theta):
    """Compute the 4x4 matrix exponential of a screw motion S*theta."""
    w_hat = skew(w)
    R = exp_rot(w_hat, theta)
    p = g_func(w_hat, theta) @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def adjoint(T):
    """6x6 adjoint representation of a 4x4 homogeneous transform."""
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


# ============================================================================
# UR5e model builder
# ============================================================================

def build_ur5e_model():
    """Build and return the UR5e kinematic model dictionary.

    All values exactly match the verified MATLAB IK code. Units: mm.
    """
    W2 = 259.6
    W1 = 133.3
    H2 = 99.7
    H1 = 162.5
    L1 = 425.0
    L2 = 392.2

    w = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 1, 0],
    ], dtype=float)

    q = np.array([
        [0, 0, 0],
        [0, 0, H1],
        [L1, 0, H1],
        [L1 + L2, 0, H1],
        [L1 + L2, W1, 0],
        [L1 + L2, 0, H1 - H2],
    ], dtype=float)

    v = np.zeros((6, 3))
    for i in range(6):
        v[i] = np.cross(-w[i], q[i])

    S = np.hstack([w, v])

    M = np.array([
        [-1, 0, 0, L1 + L2],
        [0, 0, 1, W1 + W2],
        [0, 1, 0, H1 - H2],
        [0, 0, 0, 1],
    ], dtype=float)

    Ad_M_inv = np.linalg.inv(adjoint(M))
    B = np.zeros((6, 6))
    for i in range(6):
        B[i] = Ad_M_inv @ S[i]

    return {
        "W1": W1, "W2": W2, "H1": H1, "H2": H2, "L1": L1, "L2": L2,
        "w": w, "v": v, "S": S, "B": B, "M": M, "num_joints": 6,
    }


# ============================================================================
# Forward kinematics (Product of Exponentials – space frame)
# ============================================================================

def forward_kinematics(theta, model):
    """Compute the 4x4 end-effector transform using the space-frame PoE.

    T = exp([S1]*th1) * ... * exp([S6]*th6) * M
    """
    w = model["w"]
    v = model["v"]
    M = model["M"]

    T = np.eye(4)
    for i in range(6):
        T = T @ screw_exp(w[i], v[i], theta[i])
    T = T @ M
    return T


# ============================================================================
# Sin/cos encoding and decoding
# ============================================================================

def encode_sincos(joints):
    """Encode joint angles to sin/cos pairs.

    Parameters
    ----------
    joints : ndarray, shape (..., 6) – joint angles in radians

    Returns
    -------
    sincos : ndarray, shape (..., 12) – [sin(q1), cos(q1), ..., sin(q6), cos(q6)]
    """
    joints = np.asarray(joints)
    out = np.empty(joints.shape[:-1] + (12,), dtype=joints.dtype)
    for i in range(6):
        out[..., 2 * i] = np.sin(joints[..., i])
        out[..., 2 * i + 1] = np.cos(joints[..., i])
    return out


def decode_sincos(sincos):
    """Decode sin/cos pairs back to joint angles using atan2.

    Parameters
    ----------
    sincos : ndarray, shape (..., 12) – [sin(q1), cos(q1), ..., sin(q6), cos(q6)]

    Returns
    -------
    joints : ndarray, shape (..., 6) – joint angles in radians
    """
    sincos = np.asarray(sincos)
    joints = np.empty(sincos.shape[:-1] + (6,), dtype=sincos.dtype)
    for i in range(6):
        joints[..., i] = np.arctan2(sincos[..., 2 * i], sincos[..., 2 * i + 1])
    return joints


# ============================================================================
# Feature extraction from homogeneous transform
# ============================================================================

def extract_pose_features(T):
    """Extract [x, y, z, r11, r12, ..., r33] from a 4x4 transform.

    Returns
    -------
    features : list of 12 floats
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    return [x, y, z,
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2]]


def angular_error(pred_rad, true_rad):
    """Compute angular error handling wraparound.

    Returns the shortest angular distance for each element.
    """
    diff = pred_rad - true_rad
    return np.arctan2(np.sin(diff), np.cos(diff))

import torch

def build_ur5e_model_torch(device='cpu'):
    # Exactly same params as numpy version but in torch
    W1 = 133.3; W2 = 259.6; H1 = 162.5; H2 = 99.7; L1 = 425.0; L2 = 392.2
    w = [[0.,0.,1.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,0.,-1.], [0.,1.,0.]]
    q = [[0.,0.,0.], [0.,0.,H1], [L1,0.,H1], [L1+L2,0.,H1], [L1+L2,W1,0.], [L1+L2,0.,H1-H2]]
    w_t = torch.tensor(w, device=device)
    q_t = torch.tensor(q, device=device)
    v_t = torch.cross(-w_t, q_t, dim=1)
    
    M = [[-1.,0.,0.,L1+L2], [0.,0.,1.,W1+W2], [0.,1.,0.,H1-H2], [0.,0.,0.,1.]]
    M_t = torch.tensor(M, device=device).unsqueeze(0)
    
    return {'w': w_t, 'v': v_t, 'M': M_t}

def batched_screw_exp(w, v, theta):
    B = theta.shape[0]
    device = theta.device
    I3 = torch.eye(3, device=device).unsqueeze(0)
    
    w_hat = torch.tensor([[0., -w[2], w[1]],
                          [w[2], 0., -w[0]],
                          [-w[1], w[0], 0.]], device=device).unsqueeze(0)
    w_hat_sq = torch.bmm(w_hat, w_hat)
    
    sin_t = torch.sin(theta).view(B, 1, 1)
    cos_t = torch.cos(theta).view(B, 1, 1)
    
    R = I3 + sin_t * w_hat + (1.0 - cos_t) * w_hat_sq
    
    theta_exp = theta.view(B, 1, 1)
    V_mat = (I3 * theta_exp) + (1.0 - cos_t) * w_hat + (theta_exp - sin_t) * w_hat_sq
    p = torch.bmm(V_mat, v.unsqueeze(0).unsqueeze(2).repeat(B, 1, 1)).view(B, 3)
    
    T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    return T

def batched_forward_kinematics(theta_batch, model):
    B = theta_batch.shape[0]
    device = theta_batch.device
    T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    for i in range(6):
        T_i = batched_screw_exp(model['w'][i], model['v'][i], theta_batch[:, i])
        T = torch.bmm(T, T_i)
    M = model['M'].repeat(B, 1, 1)
    T = torch.bmm(T, M)
    return T

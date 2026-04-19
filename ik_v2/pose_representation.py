"""
pose_representation.py

Configurable pose representations for IK inputs.

Supported modes:
  - 'rotmat'   : position + flattened rotation matrix  -> 12D
  - 'quat'     : position + quaternion (w,x,y,z)       ->  7D
  - 'axisangle': position + axis-angle                  ->  6D

Also provides the inverse (homogeneous matrix -> representation).
"""

import numpy as np


# ============================================================================
# Rotation matrix -> Quaternion
# ============================================================================

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].

    Uses Shepperd's method for numerical stability.
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    # Ensure w >= 0 for a canonical hemisphere
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


# ============================================================================
# Rotation matrix -> Axis-angle
# ============================================================================

def rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to axis-angle vector (3D).

    Returns omega * theta where |theta| <= pi.
    """
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-10:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta


# ============================================================================
# Quaternion -> Rotation matrix (for FK loss)
# ============================================================================

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


# ============================================================================
# Extract pose from 4x4 homogeneous transform
# ============================================================================

def extract_pose(T: np.ndarray, mode: str = "quat") -> np.ndarray:
    """Extract a pose vector from a 4x4 homogeneous transform.

    Parameters
    ----------
    T    : (4, 4) transform
    mode : 'rotmat', 'quat', 'axisangle'

    Returns
    -------
    pose : 1-D array – length depends on mode
    """
    pos = T[:3, 3]
    R = T[:3, :3]
    if mode == "rotmat":
        return np.concatenate([pos, R.flatten()])  # 12D
    elif mode == "quat":
        return np.concatenate([pos, rotmat_to_quat(R)])  # 7D
    elif mode == "axisangle":
        return np.concatenate([pos, rotmat_to_axisangle(R)])  # 6D
    else:
        raise ValueError(f"Unknown pose mode: {mode}")


def pose_dim(mode: str) -> int:
    """Return the dimensionality of a pose representation."""
    return {"rotmat": 12, "quat": 7, "axisangle": 6}[mode]


def pose_columns(mode: str) -> list[str]:
    """Column names for the pose representation."""
    pos_cols = ["x", "y", "z"]
    if mode == "rotmat":
        return pos_cols + [f"r{i}{j}" for i in range(1, 4) for j in range(1, 4)]
    elif mode == "quat":
        return pos_cols + ["qw", "qx", "qy", "qz"]
    elif mode == "axisangle":
        return pos_cols + ["ax1", "ax2", "ax3"]
    raise ValueError(f"Unknown pose mode: {mode}")


# ============================================================================
# Batch conversion  (for building datasets)
# ============================================================================

def extract_pose_batch(transforms: list[np.ndarray],
                       mode: str = "quat") -> np.ndarray:
    """Vectorised extraction for a list of (4,4) transforms."""
    return np.array([extract_pose(T, mode) for T in transforms])


# ============================================================================
# PyTorch-friendly quaternion operations (for FK loss)
# ============================================================================

def quat_to_rotmat_torch(q):
    """Batch quaternion [w,x,y,z] -> (N,3,3) rotation matrix.  PyTorch tensors."""
    import torch
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R

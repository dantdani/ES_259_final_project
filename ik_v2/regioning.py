"""
regioning.py

Flexible workspace / joint-space region assignment for the IK pipeline.

Supported strategies:
  - octant    : 8 regions based on sign of (x, y, z)
  - quadrant  : 4 regions based on sign of (x, y)  — ignores z
  - joint_bin : regions from discretised joint-angle ranges
  - custom    : user-supplied callable
"""

import numpy as np
from typing import Callable, Optional


# ============================================================================
# Octant labelling  (x,y,z sign -> 0..7)
# ============================================================================

def octant_label(x: float, y: float, z: float) -> int:
    """Map (x,y,z) to an octant index 0-7 via sign bits."""
    return (int(x >= 0) << 2) | (int(y >= 0) << 1) | int(z >= 0)


def octant_labels_batch(positions: np.ndarray) -> np.ndarray:
    """Vectorised octant labelling.

    Parameters
    ----------
    positions : (N, 3) – [x, y, z]

    Returns
    -------
    labels : (N,) int array in [0..7]
    """
    signs = (positions >= 0).astype(np.int32)
    return (signs[:, 0] << 2) | (signs[:, 1] << 1) | signs[:, 2]


OCTANT_NAMES = {
    0: "(-,-,-)", 1: "(-,-,+)", 2: "(-,+,-)", 3: "(-,+,+)",
    4: "(+,-,-)", 5: "(+,-,+)", 6: "(+,+,-)", 7: "(+,+,+)",
}


# ============================================================================
# Quadrant labelling  (x,y sign -> 0..3)
# ============================================================================

def quadrant_labels_batch(positions: np.ndarray) -> np.ndarray:
    signs = (positions[:, :2] >= 0).astype(np.int32)
    return (signs[:, 0] << 1) | signs[:, 1]


# ============================================================================
# Joint-bin labelling
# ============================================================================

def joint_bin_labels(joints: np.ndarray,
                     joint_idx: int = 1,
                     num_bins: int = 4) -> np.ndarray:
    """Bin a single joint into equal-width bins over [-pi, pi].

    Parameters
    ----------
    joints   : (N, 6)
    joint_idx: which joint to bin on (0-indexed)
    num_bins : number of bins

    Returns
    -------
    labels : (N,) int in [0..num_bins-1]
    """
    q = joints[:, joint_idx]
    edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    return np.clip(np.digitize(q, edges) - 1, 0, num_bins - 1)


# ============================================================================
# Flexible joint-range restriction
# ============================================================================

def restricted_joint_limits(mode: str = "full"):
    """Return joint limits for various branch restrictions.

    Parameters
    ----------
    mode : str
        'full'         – default [-pi, pi]
        'elbow_up'     – q3 in [-pi, 0]
        'elbow_down'   – q3 in [0, pi]
        'shoulder_left'  – q1 in [0, pi]
        'shoulder_right' – q1 in [-pi, 0]

    Returns
    -------
    limits : list of (lo, hi) for 6 joints
    """
    base = [(-np.pi, np.pi)] * 6
    if mode == "full":
        return base
    elif mode == "elbow_up":
        base[2] = (-np.pi, 0.0)
    elif mode == "elbow_down":
        base[2] = (0.0, np.pi)
    elif mode == "shoulder_left":
        base[0] = (0.0, np.pi)
    elif mode == "shoulder_right":
        base[0] = (-np.pi, 0.0)
    else:
        raise ValueError(f"Unknown restriction mode: {mode}")
    return base


# ============================================================================
# Generic regioning class
# ============================================================================

class RegionAssigner:
    """Unified region assignment interface.

    Parameters
    ----------
    strategy     : str – 'octant', 'quadrant', 'joint_bin', 'custom'
    num_regions  : int – auto-set for octant/quadrant; required for joint_bin
    joint_idx    : int – which joint for 'joint_bin'
    custom_fn    : callable(positions, joints) -> labels
    """

    def __init__(self, strategy: str = "octant", *,
                 num_regions: Optional[int] = None,
                 joint_idx: int = 1,
                 custom_fn: Optional[Callable] = None):
        self.strategy = strategy
        self.joint_idx = joint_idx
        self.custom_fn = custom_fn

        if strategy == "octant":
            self.num_regions = 8
        elif strategy == "quadrant":
            self.num_regions = 4
        elif strategy == "joint_bin":
            self.num_regions = num_regions or 4
        elif strategy == "custom":
            if custom_fn is None:
                raise ValueError("custom_fn required for strategy='custom'")
            if num_regions is None:
                raise ValueError("num_regions required for strategy='custom'")
            self.num_regions = num_regions
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def assign(self, positions: np.ndarray,
               joints: Optional[np.ndarray] = None) -> np.ndarray:
        """Return integer region labels for each sample."""
        if self.strategy == "octant":
            return octant_labels_batch(positions)
        elif self.strategy == "quadrant":
            return quadrant_labels_batch(positions)
        elif self.strategy == "joint_bin":
            if joints is None:
                raise ValueError("joints required for joint_bin strategy")
            return joint_bin_labels(joints, self.joint_idx, self.num_regions)
        else:
            return self.custom_fn(positions, joints)

    def region_name(self, idx: int) -> str:
        if self.strategy == "octant":
            return OCTANT_NAMES.get(idx, f"region_{idx}")
        return f"region_{idx}"

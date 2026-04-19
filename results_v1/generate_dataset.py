#!/usr/bin/env python3
"""
generate_dataset.py

Generate a seed-conditioned IK training dataset for the UR5e.

For each sample:
  1. Sample ground-truth joint angles q
  2. Compute FK(q) → 4×4 transform T
  3. Extract 9D pose from T: [x, y, z, r6d_0..r6d_5]
  4. Create seed configuration: q_seed = q + noise
  5. Store: [pose_9d, seed_6d] → [q1..q6]

The 15D input is: 3D position + 6D rotation + 6D seed joints.
The 12D output is: sin/cos encoded ground-truth joints.

This trains the network to find the IK solution closest to the seed.

Usage:
    python -m ik_v3.generate_dataset --samples 1000000
    python -m ik_v3.generate_dataset --samples 500000 --seed_noise 0.3
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(1, os.path.dirname(_this_dir))
from utils import build_ur5e_model

from representations import (
    extract_pose_9d,
    encode_sincos,
    POSE9D_COLS,
    JOINT_COLS,
    SEED_COLS,
)


# ============================================================================
# Vectorised FK  (loops over 6 joints, not N samples)
# ============================================================================

def _skew_np(v):
    """Skew-symmetric matrix for a single 3-vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)


def _fk_batch(joints: np.ndarray, model: dict) -> np.ndarray:
    """Vectorised FK: (N, 6) joints → (N, 4, 4) transforms.

    Loops over 6 joints instead of N samples — ~100× faster than per-sample.
    """
    w = model["w"].astype(np.float64)
    v = model["v"].astype(np.float64)
    M = model["M"].astype(np.float64)
    N = joints.shape[0]

    T = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))

    for j in range(6):
        theta = joints[:, j].astype(np.float64)
        wj, vj = w[j], v[j]
        wh = _skew_np(wj)
        wh2 = wh @ wh

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        R = (np.eye(3)[None, :, :]
             + sin_t[:, None, None] * wh[None, :, :]
             + (1.0 - cos_t[:, None, None]) * wh2[None, :, :])

        G = (np.eye(3)[None, :, :] * theta[:, None, None]
             + (1.0 - cos_t[:, None, None]) * wh[None, :, :]
             + (theta[:, None, None] - sin_t[:, None, None]) * wh2[None, :, :])

        p = (G @ vj[None, :, None]).squeeze(-1)

        Tj = np.zeros((N, 4, 4), dtype=np.float64)
        Tj[:, :3, :3] = R
        Tj[:, :3, 3] = p
        Tj[:, 3, 3] = 1.0

        T = np.einsum("nij,njk->nik", T, Tj)

    T = np.einsum("nij,jk->nik", T, M)
    return T


# ============================================================================
# Seed generation strategies
# ============================================================================

def generate_seeds(joints: np.ndarray, noise_std: float,
                   limits: list[tuple]) -> np.ndarray:
    """Generate seed configurations by adding Gaussian noise to ground truth.

    The seeds are clipped to stay within joint limits.

    Parameters
    ----------
    joints : (N, 6) ground-truth joint angles
    noise_std : standard deviation of Gaussian noise (radians)
    limits : list of (lo, hi) for each joint

    Returns
    -------
    seeds : (N, 6) perturbed joint angles
    """
    noise = np.random.randn(*joints.shape).astype(np.float32) * noise_std
    seeds = joints + noise
    for i, (lo, hi) in enumerate(limits):
        seeds[:, i] = np.clip(seeds[:, i], lo, hi)
    return seeds


# ============================================================================
# Main generation
# ============================================================================

def generate_dataset(num_samples: int,
                     seed_noise: float = 0.5,
                     batch_size: int = 100_000,
                     joint_limits: list[tuple] | None = None,
                     rng_seed: int = 42) -> pd.DataFrame:
    """Generate the full seed-conditioned dataset.

    Parameters
    ----------
    num_samples : total number of samples to generate
    seed_noise : std dev of Gaussian noise for seed generation (rad)
    batch_size : FK batch size
    joint_limits : per-joint (lo, hi) limits
    rng_seed : random seed for reproducibility

    Returns
    -------
    DataFrame with columns:
        [x, y, z, r6d_0..5, seed_q1..6, q1..6]
    """
    np.random.seed(rng_seed)
    model = build_ur5e_model()

    if joint_limits is None:
        joint_limits = [(-np.pi, np.pi)] * 6

    all_rows = []
    generated = 0
    t0 = time.time()

    while generated < num_samples:
        n = min(batch_size, num_samples - generated)

        # 1. Sample ground-truth joints
        joints = np.empty((n, 6), dtype=np.float32)
        for i, (lo, hi) in enumerate(joint_limits):
            joints[:, i] = np.random.uniform(lo, hi, size=n)

        # 2. Batch FK
        T_all = _fk_batch(joints, model)  # (n, 4, 4)

        # 3. Extract 9D pose
        pose_9d = extract_pose_9d(T_all).astype(np.float32)  # (n, 9)

        # 4. Generate seed joints
        seeds = generate_seeds(joints, seed_noise, joint_limits)

        # 5. Stack: [pose_9d(9), seeds(6), joints(6)] = 21 columns
        row = np.hstack([pose_9d, seeds, joints])
        all_rows.append(row)

        generated += n
        elapsed = time.time() - t0
        print(f"  Generated {generated:>10,} / {num_samples:,}  ({elapsed:.1f}s)")

    data = np.vstack(all_rows)

    columns = POSE9D_COLS + SEED_COLS + JOINT_COLS
    df = pd.DataFrame(data, columns=columns)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate seed-conditioned UR5e IK dataset")
    parser.add_argument("--samples", type=int, default=1_000_000,
                        help="Total number of samples")
    parser.add_argument("--seed_noise", type=float, default=0.5,
                        help="Std dev of seed noise (radians, default 0.5)")
    parser.add_argument("--batch_size", type=int, default=100_000,
                        help="FK batch size")
    parser.add_argument("--rng_seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    print(f"Generating {args.samples:,} samples "
          f"(seed_noise={args.seed_noise} rad)")
    print("=" * 60)

    df = generate_dataset(
        num_samples=args.samples,
        seed_noise=args.seed_noise,
        batch_size=args.batch_size,
        rng_seed=args.rng_seed,
    )

    if args.output is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ur5e_seed_dataset.csv")
    else:
        out_path = args.output

    df.to_csv(out_path, index=False)
    print(f"\nDataset: {len(df):,} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_region_dataset.py

Generate a region-labelled FK/IK dataset for the UR5e.

For each sample:
  1. Sample joint angles (within optional branch restrictions)
  2. Run FK
  3. Extract pose (quaternion / rotmat / axis-angle)
  4. Assign workspace region (octant / quadrant / joint_bin)

Targets ~100k samples per region (oversamples sparse regions).

Usage:
    python -m ik_v2.generate_region_dataset --total 800000 --pose_mode quat
    python -m ik_v2.generate_region_dataset --total 800000 --strategy octant --per_region 100000
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Add parent dir to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import build_ur5e_model, forward_kinematics, encode_sincos

from regioning import RegionAssigner, restricted_joint_limits
from pose_representation import extract_pose, pose_columns, rotmat_to_quat


def sample_joints(n: int, limits: list[tuple]) -> np.ndarray:
    """Sample n random joint angle vectors within limits."""
    joints = np.empty((n, 6), dtype=np.float32)
    for i, (lo, hi) in enumerate(limits):
        joints[:, i] = np.random.uniform(lo, hi, size=n)
    return joints


# ============================================================================
# Vectorised FK  (pure numpy, ~100x faster than per-sample loop)
# ============================================================================

def _skew_np(v):
    """Skew-symmetric for a single 3-vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)


def _fk_batch(joints, model):
    """Vectorised FK for (N, 6) joints -> (N, 4, 4) transforms.

    Uses the same Product-of-Exponentials formulation as utils.forward_kinematics,
    but loops over joints (6) rather than samples (N).
    """
    w = model["w"].astype(np.float64)
    v = model["v"].astype(np.float64)
    M = model["M"].astype(np.float64)
    N = joints.shape[0]

    T = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))  # (N,4,4)

    for j in range(6):
        theta = joints[:, j].astype(np.float64)  # (N,)
        wj = w[j]                # (3,)
        vj = v[j]                # (3,)
        wh = _skew_np(wj)        # (3,3)
        wh2 = wh @ wh            # (3,3)

        sin_t = np.sin(theta)    # (N,)
        cos_t = np.cos(theta)

        # Rodrigues:  R = I + sin*wh + (1-cos)*wh^2   -> broadcast
        R = (np.eye(3)[None, :, :]
             + sin_t[:, None, None] * wh[None, :, :]
             + (1 - cos_t[:, None, None]) * wh2[None, :, :])   # (N,3,3)

        # Translation:  G*v  where G = I*th + (1-cos)*wh + (th-sin)*wh^2
        G = (theta[:, None, None] * np.eye(3)[None, :, :]
             + (1 - cos_t[:, None, None]) * wh[None, :, :]
             + (theta - sin_t)[:, None, None] * wh2[None, :, :])  # (N,3,3)
        p = (G @ vj[None, :, None]).squeeze(-1)  # (N,3)

        Tj = np.zeros((N, 4, 4), dtype=np.float64)
        Tj[:, :3, :3] = R
        Tj[:, :3, 3] = p
        Tj[:, 3, 3] = 1.0

        T = T @ Tj   # numpy broadcasts (N,4,4) @ (N,4,4)

    T = T @ M[None, :, :]  # multiply by home config
    return T


def _extract_pose_batch(T_batch, mode):
    """Extract poses from (N,4,4) transforms without a Python loop."""
    pos = T_batch[:, :3, 3].astype(np.float32)  # (N,3)
    R = T_batch[:, :3, :3]                       # (N,3,3)

    if mode == "rotmat":
        rot = R.reshape(-1, 9).astype(np.float32)
        return np.hstack([pos, rot])
    elif mode == "quat":
        quats = np.array([rotmat_to_quat(R[i]) for i in range(len(R))],
                          dtype=np.float32)
        return np.hstack([pos, quats])
    elif mode == "axisangle":
        from pose_representation import rotmat_to_axisangle
        aa = np.array([rotmat_to_axisangle(R[i]) for i in range(len(R))],
                       dtype=np.float32)
        return np.hstack([pos, aa])
    raise ValueError(f"Unknown mode: {mode}")


def generate_samples(n: int, model: dict, pose_mode: str,
                     limits: list[tuple]) -> tuple:
    """Generate n FK samples (vectorised).

    Returns
    -------
    positions : (n, 3)
    poses     : (n, D)   D depends on pose_mode
    joints    : (n, 6)
    """
    joints = sample_joints(n, limits)
    T_batch = _fk_batch(joints, model)             # (N,4,4)
    positions = T_batch[:, :3, 3].astype(np.float32)
    poses = _extract_pose_batch(T_batch, pose_mode)
    return positions, poses, joints


def generate_dataset(total: int, per_region: int, pose_mode: str,
                     strategy: str, joint_mode: str,
                     max_attempts_factor: int = 20) -> pd.DataFrame:
    """Generate a region-balanced dataset.

    Samples in batches, assigns regions, and keeps sampling until
    each region reaches per_region samples or the attempt budget is exhausted.
    """
    model = build_ur5e_model()
    limits = restricted_joint_limits(joint_mode)
    assigner = RegionAssigner(strategy)
    num_regions = assigner.num_regions

    pose_cols = pose_columns(pose_mode)
    joint_cols = [f"q{i+1}" for i in range(6)]

    # Collect samples per region
    region_data = {r: [] for r in range(num_regions)}
    region_counts = np.zeros(num_regions, dtype=int)
    target_per = per_region if per_region > 0 else (total // num_regions)

    total_generated = 0
    max_total = total * max_attempts_factor
    batch_size = min(100_000, total)

    print(f"Generating dataset: strategy={strategy}, pose={pose_mode}, "
          f"joint_mode={joint_mode}")
    print(f"Target: {target_per:,} samples per region, {num_regions} regions")
    print(f"Batch size: {batch_size:,}, max attempts: {max_total:,}")
    print("-" * 60)

    t0 = time.time()

    while True:
        # Check if all regions are full
        if np.all(region_counts >= target_per):
            print("All regions reached target!")
            break
        if total_generated >= max_total:
            print(f"Reached max attempt budget ({max_total:,})")
            break

        # Generate a batch
        n = min(batch_size, max_total - total_generated)
        positions, poses, joints = generate_samples(n, model, pose_mode, limits)
        total_generated += n

        # Assign regions
        labels = assigner.assign(positions, joints)

        # Distribute to region buckets
        for r in range(num_regions):
            if region_counts[r] >= target_per:
                continue
            mask = labels == r
            needed = target_per - region_counts[r]
            idx = np.where(mask)[0][:needed]
            if len(idx) > 0:
                for j in idx:
                    row = np.concatenate([
                        poses[j], [r], joints[j]
                    ])
                    region_data[r].append(row)
                region_counts[r] += len(idx)

        elapsed = time.time() - t0
        print(f"  Generated {total_generated:,} | "
              f"Region counts: {region_counts.tolist()} | "
              f"{elapsed:.1f}s")

    # Build DataFrame
    all_rows = []
    for r in range(num_regions):
        all_rows.extend(region_data[r])

    columns = pose_cols + ["region_id"] + joint_cols
    df = pd.DataFrame(all_rows, columns=columns)
    df["region_id"] = df["region_id"].astype(int)

    elapsed = time.time() - t0
    print(f"\nDataset: {len(df):,} total samples in {elapsed:.1f}s")
    print(f"Per-region counts:")
    for r in range(num_regions):
        name = assigner.region_name(r)
        print(f"  {name}: {region_counts[r]:,}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate region-labelled IK dataset for UR5e")
    parser.add_argument("--total", type=int, default=800_000,
                        help="Total target samples (used for attempt budget)")
    parser.add_argument("--per_region", type=int, default=100_000,
                        help="Target samples per region (0 = total/num_regions)")
    parser.add_argument("--pose_mode", type=str, default="quat",
                        choices=["rotmat", "quat", "axisangle"])
    parser.add_argument("--strategy", type=str, default="octant",
                        choices=["octant", "quadrant", "joint_bin"])
    parser.add_argument("--joint_mode", type=str, default="full",
                        choices=["full", "elbow_up", "elbow_down",
                                 "shoulder_left", "shoulder_right"])
    parser.add_argument("--output", type=str, default="ur5e_region_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    df = generate_dataset(
        total=args.total,
        per_region=args.per_region,
        pose_mode=args.pose_mode,
        strategy=args.strategy,
        joint_mode=args.joint_mode,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

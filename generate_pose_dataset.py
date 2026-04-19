#!/usr/bin/env python3
"""
generate_pose_dataset.py

Generate a full-pose IK dataset for the UR5e robot.

For each sample:
  1. Randomly sample joint angles within limits
  2. Run forward kinematics
  3. Store: [x, y, z, r11..r33, q1..q6]  (12 + 6 = 18 columns)

Output is a CSV file suitable for training.

Usage:
    python generate_pose_dataset.py --num_samples 500000
    python generate_pose_dataset.py --num_samples 1000000 --output ur5e_pose_1M.csv
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from utils import (
    DEFAULT_JOINT_LIMITS,
    POSE_COLS,
    JOINT_COLS,
    build_ur5e_model,
    forward_kinematics,
    extract_pose_features,
)

CHUNK_SIZE = 50_000  # write to CSV in chunks to keep memory bounded


def generate_dataset(num_samples, output_csv, seed=42, joint_limits=None):
    """Generate a full-pose IK dataset and write it to CSV in chunks.

    Parameters
    ----------
    num_samples  : int
    output_csv   : str
    seed         : int
    joint_limits : list of (lo, hi) tuples

    Returns
    -------
    stats : dict
    """
    if joint_limits is None:
        joint_limits = DEFAULT_JOINT_LIMITS

    rng = np.random.default_rng(seed)
    model = build_ur5e_model()

    all_cols = POSE_COLS + JOINT_COLS
    header_written = False
    success = 0
    fail = 0
    chunk_rows = []
    log_interval = max(1, num_samples // 20)

    t_start = time.time()

    # Pre-compute limits arrays for vectorised sampling
    lo = np.array([lim[0] for lim in joint_limits])
    hi = np.array([lim[1] for lim in joint_limits])

    for i in range(num_samples):
        try:
            theta = rng.uniform(lo, hi)
            T = forward_kinematics(theta, model)

            if np.any(np.isnan(T)) or np.any(np.isinf(T)):
                raise ValueError("FK returned NaN/Inf")

            features = extract_pose_features(T)
            chunk_rows.append(features + theta.tolist())
            success += 1

        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  [warn] Sample {i}: {e}")

        # Flush chunk to CSV
        if len(chunk_rows) >= CHUNK_SIZE:
            df_chunk = pd.DataFrame(chunk_rows, columns=all_cols)
            df_chunk.to_csv(
                output_csv,
                mode="a",
                header=not header_written,
                index=False,
            )
            header_written = True
            chunk_rows = []

        if (i + 1) % log_interval == 0 or (i + 1) == num_samples:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  progress: {i+1:>8d} / {num_samples}  "
                f"({100*(i+1)/num_samples:5.1f}%)  "
                f"success={success}  fail={fail}  "
                f"elapsed={elapsed:.1f}s  ({rate:.0f} samples/s)"
            )

    # Flush remaining rows
    if chunk_rows:
        df_chunk = pd.DataFrame(chunk_rows, columns=all_cols)
        df_chunk.to_csv(
            output_csv,
            mode="a",
            header=not header_written,
            index=False,
        )

    stats = {
        "requested": num_samples,
        "success": success,
        "fail": fail,
        "output_file": output_csv,
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    return stats


def verify_fk(model):
    """Quick sanity checks on FK."""
    print("=" * 60)
    print("FK VERIFICATION")
    print("=" * 60)

    M = model["M"]
    T0 = forward_kinematics(np.zeros(6), model)
    err = np.linalg.norm(T0 - M)
    print(f"  FK(zeros) vs M  ->  error = {err:.2e}  ", end="")
    assert err < 1e-10, f"FK mismatch: {err}"
    print("PASS")

    rng = np.random.default_rng(99)
    for trial in range(20):
        theta = rng.uniform(-np.pi, np.pi, size=6)
        T = forward_kinematics(theta, model)
        R = T[:3, :3]
        ortho = np.linalg.norm(R @ R.T - np.eye(3))
        det = abs(np.linalg.det(R) - 1.0)
        assert ortho < 1e-8 and det < 1e-8, f"Rotation check failed (trial {trial})"
    print("  Rotation orthonormality (20 configs)  PASS")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate UR5e full-pose IK dataset.")
    parser.add_argument("--num_samples", "-n", type=int, default=500_000)
    parser.add_argument("--output", "-o", type=str, default="ur5e_pose_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_verify", action="store_true")
    args = parser.parse_args()

    model = build_ur5e_model()

    if not args.skip_verify:
        verify_fk(model)

    # Remove old file if it exists (since we append in chunks)
    if os.path.exists(args.output):
        os.remove(args.output)

    print(f"Generating {args.num_samples:,} full-pose samples  (seed={args.seed})")
    print(f"Output -> {args.output}\n")

    stats = generate_dataset(
        num_samples=args.num_samples,
        output_csv=args.output,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Requested : {stats['requested']:,}")
    print(f"  Success   : {stats['success']:,}")
    print(f"  Failed    : {stats['fail']:,}")
    print(f"  Output    : {stats['output_file']}")
    print(f"  Elapsed   : {stats['elapsed_sec']} s")
    print("=" * 60)


if __name__ == "__main__":
    main()

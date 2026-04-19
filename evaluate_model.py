#!/usr/bin/env python3
"""
evaluate_model.py

Load a trained IK model and evaluate on a whole dataset or test split.
Reports per-joint MAE, RMSE, FK reconstruction error, and saves scatter plots.

Usage:
    python evaluate_model.py
    python evaluate_model.py --model pose_results/ik_pose_best.pt --csv ur5e_pose_dataset.csv
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from utils import (
    POSE_COLS,
    JOINT_COLS,
    encode_sincos,
    decode_sincos,
    angular_error,
    build_ur5e_model,
    forward_kinematics,
)
from model import build_model


def fk_error(q_pred, q_true):
    """Compute FK reconstruction errors (position mm + orientation deg).

    Returns
    -------
    pos_errors : ndarray (N,)  – Euclidean distance in mm
    rot_errors : ndarray (N,)  – rotation error in degrees (Frobenius-based)
    """
    S, M = build_ur5e_model()
    n = q_pred.shape[0]
    pos_errors = np.empty(n)
    rot_errors = np.empty(n)

    for i in range(n):
        T_pred = forward_kinematics(S, M, q_pred[i])
        T_true = forward_kinematics(S, M, q_true[i])

        # Position
        dp = T_pred[:3, 3] - T_true[:3, 3]
        pos_errors[i] = np.linalg.norm(dp)

        # Orientation (rotation matrix difference as angle)
        dR = T_pred[:3, :3] @ T_true[:3, :3].T
        cos_angle = (np.trace(dR) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rot_errors[i] = np.degrees(np.arccos(cos_angle))

    return pos_errors, rot_errors


def main():
    parser = argparse.ArgumentParser(description="Evaluate IK model")
    parser.add_argument("--model", default="pose_results/ik_pose_best.pt",
                        help="Path to saved model weights")
    parser.add_argument("--scaler", default="pose_results/input_scaler.pkl",
                        help="Path to input scaler")
    parser.add_argument("--csv", default="ur5e_pose_dataset.csv",
                        help="Dataset CSV")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fk_samples", type=int, default=5000,
                        help="Number of samples for FK reconstruction check")
    parser.add_argument("--out_dir", default="pose_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    net = build_model(input_dim=12, output_dim=12,
                      hidden_dim=args.hidden_dim,
                      num_blocks=args.num_blocks,
                      dropout=args.dropout)
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    print(f"Loaded model from {args.model}")

    # --- Load scaler ---
    with open(args.scaler, "rb") as f:
        input_scaler = pickle.load(f)
    print(f"Loaded scaler from {args.scaler}")

    # --- Load data & get test split ---
    df = pd.read_csv(args.csv)
    X = df[POSE_COLS].values.astype(np.float32)
    Y_raw = df[JOINT_COLS].values.astype(np.float32)

    _, X_tmp, _, Y_tmp = train_test_split(X, Y_raw, test_size=0.2,
                                          random_state=args.seed)
    _, X_te, _, Y_te_raw = train_test_split(X_tmp, Y_tmp, test_size=0.5,
                                            random_state=args.seed)

    print(f"Test set: {len(X_te):,} samples")

    # --- Predict ---
    X_te_s = input_scaler.transform(X_te).astype(np.float32)
    X_tensor = torch.from_numpy(X_te_s).to(device)

    with torch.no_grad():
        preds_sincos = net(X_tensor).cpu().numpy()

    preds_rad = decode_sincos(preds_sincos)
    targets_rad = Y_te_raw

    # --- Angular metrics ---
    ang_err = angular_error(preds_rad, targets_rad)
    abs_err = np.abs(ang_err)

    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(ang_err ** 2))
    per_joint_mae = np.mean(abs_err, axis=0)
    per_joint_rmse = np.sqrt(np.mean(ang_err ** 2, axis=0))

    print("\n" + "=" * 70)
    print("TEST SET — ANGULAR METRICS")
    print("=" * 70)
    print(f"  Overall MAE  : {mae:.6f} rad  ({np.degrees(mae):.2f} deg)")
    print(f"  Overall RMSE : {rmse:.6f} rad  ({np.degrees(rmse):.2f} deg)")
    print(f"  Per-joint breakdown:")
    for i, name in enumerate(JOINT_COLS):
        print(f"    {name}: MAE={per_joint_mae[i]:.6f} ({np.degrees(per_joint_mae[i]):.2f}°)  "
              f"RMSE={per_joint_rmse[i]:.6f} ({np.degrees(per_joint_rmse[i]):.2f}°)")

    # --- FK reconstruction error (subset) ---
    n_fk = min(args.fk_samples, len(preds_rad))
    idx = np.random.RandomState(args.seed).choice(len(preds_rad), n_fk, replace=False)

    print(f"\nComputing FK reconstruction error on {n_fk} samples...")
    pos_err, rot_err = fk_error(preds_rad[idx], targets_rad[idx])

    print(f"\n{'=' * 70}")
    print("TEST SET — FK RECONSTRUCTION ERROR")
    print(f"{'=' * 70}")
    print(f"  Position error (mm):")
    print(f"    Mean  : {np.mean(pos_err):.2f}")
    print(f"    Median: {np.median(pos_err):.2f}")
    print(f"    Std   : {np.std(pos_err):.2f}")
    print(f"    Max   : {np.max(pos_err):.2f}")
    print(f"  Orientation error (deg):")
    print(f"    Mean  : {np.mean(rot_err):.2f}")
    print(f"    Median: {np.median(rot_err):.2f}")
    print(f"    Std   : {np.std(rot_err):.2f}")
    print(f"    Max   : {np.max(rot_err):.2f}")
    print("=" * 70)

    # --- Save summary ---
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Angular Metrics (test set) ===\n")
        f.write(f"Overall MAE  : {mae:.6f} rad ({np.degrees(mae):.2f} deg)\n")
        f.write(f"Overall RMSE : {rmse:.6f} rad ({np.degrees(rmse):.2f} deg)\n")
        f.write(f"\nPer-joint MAE:\n")
        for i, name in enumerate(JOINT_COLS):
            f.write(f"  {name}: {per_joint_mae[i]:.6f} rad ({np.degrees(per_joint_mae[i]):.2f} deg)\n")
        f.write(f"\n=== FK Reconstruction Error ({n_fk} samples) ===\n")
        f.write(f"Position (mm)  : mean={np.mean(pos_err):.2f}, median={np.median(pos_err):.2f}\n")
        f.write(f"Orientation (°): mean={np.mean(rot_err):.2f}, median={np.median(rot_err):.2f}\n")
    print(f"\nSaved summary -> {summary_path}")

    # --- Scatter plots ---
    for i in range(6):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(targets_rad[:, i], preds_rad[:, i],
                   s=2, alpha=0.3, edgecolors="none")
        lims = [-np.pi - 0.2, np.pi + 0.2]
        ax.plot(lims, lims, "r--", linewidth=1, label="perfect")
        ax.set_xlabel(f"True q{i+1} (rad)")
        ax.set_ylabel(f"Predicted q{i+1} (rad)")
        ax.set_title(f"q{i+1}: MAE={np.degrees(per_joint_mae[i]):.2f}°")
        ax.legend()
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"eval_q{i+1}.png"), dpi=150)
        plt.close(fig)
    print(f"Saved scatter plots -> {args.out_dir}/eval_q*.png")

    # --- Position error histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos_err, bins=80, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(pos_err), color="r", linestyle="--",
               label=f"Mean = {np.mean(pos_err):.1f} mm")
    ax.set_xlabel("Position Error (mm)")
    ax.set_ylabel("Count")
    ax.set_title("FK Reconstruction — Position Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fk_position_error_hist.png"), dpi=150)
    plt.close(fig)
    print(f"Saved FK error histogram")


if __name__ == "__main__":
    main()

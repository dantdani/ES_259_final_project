#!/usr/bin/env python3
"""
evaluate.py

Comprehensive evaluation of trained IK models.

Metrics:
  - Per-joint MAE (rad / deg)
  - Overall MAE, median error
  - % predictions within thresholds (1, 5, 10, 20 degrees)
  - FK position error (mm)
  - FK orientation error (deg)
  - All metrics per region
  - Failure analysis
"""

import argparse
import os
import sys
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (build_ur5e_model, forward_kinematics,
                   encode_sincos, decode_sincos, angular_error)

from pose_representation import pose_columns, pose_dim, rotmat_to_quat
from regioning import RegionAssigner, OCTANT_NAMES
from fk_loss import decode_sincos_torch


def load_model(model_type, model_path, args, device):
    """Load the appropriate model architecture and weights."""
    input_d = pose_dim(args.pose_mode)

    if model_type == "global":
        from model_global import build_global_model
        model = build_global_model(
            input_dim=input_d, hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks, norm=args.norm)
    elif model_type == "global_region":
        from model_global import build_global_model
        model = build_global_model(
            input_dim=input_d, hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks, norm=args.norm,
            num_regions=args.num_regions,
            region_embed_dim=args.region_embed_dim)
    elif model_type == "expert":
        from model_expert import build_expert_ensemble
        model = build_expert_ensemble(
            num_experts=args.num_regions, input_dim=input_d,
            hidden_dim=args.expert_hidden, num_blocks=args.expert_blocks,
            norm=args.norm)
    elif model_type == "moe":
        from model_moe import build_moe_model
        model = build_moe_model(
            num_experts=args.num_regions, input_dim=input_d,
            expert_hidden=args.expert_hidden, expert_blocks=args.expert_blocks,
            gate_hidden=args.gate_hidden, top_k=args.top_k,
            norm=args.norm)
    elif model_type == "refinement":
        from model_refinement import build_refinement_model
        model = build_refinement_model(
            pose_dim=input_d, init_hidden=args.hidden_dim,
            init_blocks=args.num_blocks,
            refine_hidden=args.refine_hidden,
            refine_blocks=args.refine_blocks,
            num_refine=args.num_refine,
            share_refine=args.share_refine, norm=args.norm)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device,
                                      weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict(model, X_scaled, regions, model_type, device, batch_size=512):
    """Run inference and return sin/cos predictions."""
    model.eval()
    preds = []
    N = len(X_scaled)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(X_scaled[i:i+batch_size]).to(device)
            rb = torch.from_numpy(regions[i:i+batch_size]).to(device)

            if model_type == "global_region":
                pred = model(xb, rb)
            elif model_type == "expert":
                pred = model(xb, rb)
            elif model_type in ("refinement",):
                pred = model(xb)
            else:
                pred = model(xb)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def compute_fk_errors(pred_joints, true_joints, ur5e_model):
    """Compute FK position (mm) and orientation (deg) errors."""
    N = len(pred_joints)
    pos_errors = np.zeros(N)
    rot_errors = np.zeros(N)

    for i in range(N):
        T_pred = forward_kinematics(pred_joints[i], ur5e_model)
        T_true = forward_kinematics(true_joints[i], ur5e_model)

        # Position error (Euclidean, mm)
        pos_errors[i] = np.linalg.norm(T_pred[:3, 3] - T_true[:3, 3])

        # Orientation error (angle of R_err = R_pred @ R_true^T)
        R_err = T_pred[:3, :3] @ T_true[:3, :3].T
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_errors[i] = np.degrees(np.arccos(cos_angle))

    return pos_errors, rot_errors


def compute_metrics(pred_rad, true_rad, regions, num_regions,
                    ur5e_model, max_fk_samples=10000):
    """Compute all evaluation metrics."""
    ang_err = angular_error(pred_rad, true_rad)
    abs_err = np.abs(ang_err)
    abs_err_deg = np.degrees(abs_err)

    results = {}

    # -- Overall --
    results["overall"] = {
        "mae_rad": float(np.mean(abs_err)),
        "mae_deg": float(np.mean(abs_err_deg)),
        "median_rad": float(np.median(abs_err)),
        "median_deg": float(np.median(abs_err_deg)),
        "rmse_rad": float(np.sqrt(np.mean(ang_err**2))),
        "pct_below_1deg": float(np.mean(abs_err_deg < 1) * 100),
        "pct_below_5deg": float(np.mean(abs_err_deg < 5) * 100),
        "pct_below_10deg": float(np.mean(abs_err_deg < 10) * 100),
        "pct_below_20deg": float(np.mean(abs_err_deg < 20) * 100),
    }

    # Per-joint MAE
    per_joint = {}
    for j in range(6):
        per_joint[f"q{j+1}_mae_rad"] = float(np.mean(abs_err[:, j]))
        per_joint[f"q{j+1}_mae_deg"] = float(np.mean(abs_err_deg[:, j]))
    results["per_joint"] = per_joint

    # FK errors (subsample for speed)
    fk_idx = np.random.choice(len(pred_rad),
                               min(max_fk_samples, len(pred_rad)),
                               replace=False)
    pos_err, rot_err = compute_fk_errors(
        pred_rad[fk_idx], true_rad[fk_idx], ur5e_model)
    results["fk"] = {
        "pos_mean_mm": float(np.mean(pos_err)),
        "pos_median_mm": float(np.median(pos_err)),
        "pos_p95_mm": float(np.percentile(pos_err, 95)),
        "rot_mean_deg": float(np.mean(rot_err)),
        "rot_median_deg": float(np.median(rot_err)),
        "rot_p95_deg": float(np.percentile(rot_err, 95)),
    }

    # -- Per-region --
    results["per_region"] = {}
    for r in range(num_regions):
        mask = regions == r
        if mask.sum() == 0:
            continue
        r_abs = abs_err[mask]
        r_deg = abs_err_deg[mask]
        results["per_region"][r] = {
            "count": int(mask.sum()),
            "mae_rad": float(np.mean(r_abs)),
            "mae_deg": float(np.mean(r_deg)),
            "median_deg": float(np.median(r_deg)),
            "pct_below_5deg": float(np.mean(r_deg < 5) * 100),
            "pct_below_10deg": float(np.mean(r_deg < 10) * 100),
        }

    return results


def print_results(results, num_regions):
    """Pretty-print evaluation results."""
    o = results["overall"]
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print(f"  MAE:    {o['mae_rad']:.4f} rad  ({o['mae_deg']:.2f} deg)")
    print(f"  Median: {o['median_rad']:.4f} rad  ({o['median_deg']:.2f} deg)")
    print(f"  RMSE:   {o['rmse_rad']:.4f} rad")
    print(f"  < 1°:   {o['pct_below_1deg']:.1f}%")
    print(f"  < 5°:   {o['pct_below_5deg']:.1f}%")
    print(f"  < 10°:  {o['pct_below_10deg']:.1f}%")
    print(f"  < 20°:  {o['pct_below_20deg']:.1f}%")

    print("\nPER-JOINT MAE:")
    pj = results["per_joint"]
    for j in range(6):
        print(f"  q{j+1}: {pj[f'q{j+1}_mae_rad']:.4f} rad  "
              f"({pj[f'q{j+1}_mae_deg']:.2f} deg)")

    print("\nFK RECONSTRUCTION:")
    fk = results["fk"]
    print(f"  Position: mean={fk['pos_mean_mm']:.1f} mm, "
          f"median={fk['pos_median_mm']:.1f} mm, "
          f"p95={fk['pos_p95_mm']:.1f} mm")
    print(f"  Rotation: mean={fk['rot_mean_deg']:.2f} deg, "
          f"median={fk['rot_median_deg']:.2f} deg, "
          f"p95={fk['rot_p95_deg']:.2f} deg")

    print("\nPER-REGION:")
    for r in sorted(results["per_region"]):
        rr = results["per_region"][r]
        name = OCTANT_NAMES.get(r, f"region_{r}")
        print(f"  {name} (n={rr['count']:,}): "
              f"MAE={rr['mae_deg']:.2f}°, "
              f"median={rr['median_deg']:.2f}°, "
              f"<5°={rr['pct_below_5deg']:.1f}%, "
              f"<10°={rr['pct_below_10deg']:.1f}%")
    print("=" * 70)


def save_results(results, pred_rad, true_rad, regions, out_dir, num_regions):
    """Save metrics and plots."""
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics as text
    with open(os.path.join(out_dir, "eval_results.txt"), "w") as f:
        import json
        json.dump(results, f, indent=2)

    # Pred vs true scatter for each joint
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for j in range(6):
        ax = axes[j // 3][j % 3]
        ax.scatter(true_rad[:, j], pred_rad[:, j], s=1, alpha=0.1)
        lims = [-np.pi - 0.2, np.pi + 0.2]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel(f"True q{j+1}")
        ax.set_ylabel(f"Pred q{j+1}")
        ax.set_title(f"q{j+1} — MAE={results['per_joint'][f'q{j+1}_mae_deg']:.1f}°")
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pred_vs_true.png"), dpi=150)
    plt.close(fig)

    # Per-region bar chart
    regions_sorted = sorted(results["per_region"].keys())
    if len(regions_sorted) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [OCTANT_NAMES.get(r, f"R{r}") for r in regions_sorted]
        maes = [results["per_region"][r]["mae_deg"] for r in regions_sorted]
        ax.bar(names, maes, color="steelblue")
        ax.set_ylabel("MAE (degrees)")
        ax.set_title("Per-Region MAE")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "per_region_mae.png"), dpi=150)
        plt.close(fig)

    # Error distribution histogram
    abs_err_deg = np.degrees(np.abs(angular_error(pred_rad, true_rad)))
    mean_err = np.mean(abs_err_deg, axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mean_err, bins=100, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(mean_err), color="red", ls="--",
               label=f"Mean={np.mean(mean_err):.1f}°")
    ax.axvline(np.median(mean_err), color="green", ls="--",
               label=f"Median={np.median(mean_err):.1f}°")
    ax.set_xlabel("Mean Per-Sample Error (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"Saved evaluation outputs to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate IK model (v2)")

    parser.add_argument("--csv", type=str, default="ur5e_region_dataset.csv")
    parser.add_argument("--pose_mode", type=str, default="quat")
    parser.add_argument("--model_type", type=str, default="global",
                        choices=["global", "global_region", "expert",
                                 "moe", "refinement"])
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing model_best.pt and input_scaler.pkl")

    # Architecture params (must match training)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--norm", type=str, default="layer")
    parser.add_argument("--num_regions", type=int, default=8)
    parser.add_argument("--region_embed_dim", type=int, default=16)
    parser.add_argument("--expert_hidden", type=int, default=256)
    parser.add_argument("--expert_blocks", type=int, default=4)
    parser.add_argument("--gate_hidden", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--refine_hidden", type=int, default=256)
    parser.add_argument("--refine_blocks", type=int, default=3)
    parser.add_argument("--num_refine", type=int, default=2)
    parser.add_argument("--share_refine", action="store_true", default=True)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            args.csv)
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.csv)

    p_cols = pose_columns(args.pose_mode)
    j_cols = [f"q{i+1}" for i in range(6)]
    df = pd.read_csv(csv_path)
    X = df[p_cols].values.astype(np.float32)
    regions = df["region_id"].values.astype(np.int64)
    Y_raw = df[j_cols].values.astype(np.float32)

    # Use same split as training
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.2,
                                        random_state=args.seed,
                                        stratify=regions)
    _, idx_te = train_test_split(idx_tmp, test_size=0.5,
                                 random_state=args.seed,
                                 stratify=regions[idx_tmp])

    # Load scaler
    with open(os.path.join(args.model_dir, "input_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X_s = scaler.transform(X).astype(np.float32)

    # Load model
    model_path = os.path.join(args.model_dir, "model_best.pt")
    model = load_model(args.model_type, model_path, args, device)

    # Predict on test set
    print(f"\nEvaluating on {len(idx_te):,} test samples...")
    pred_sc = predict(model, X_s[idx_te], regions[idx_te],
                      args.model_type, device)
    pred_rad = decode_sincos(pred_sc)
    true_rad = Y_raw[idx_te]

    # Metrics
    ur5e = build_ur5e_model()
    results = compute_metrics(pred_rad, true_rad, regions[idx_te],
                              args.num_regions, ur5e)
    print_results(results, args.num_regions)

    # Save
    eval_dir = os.path.join(args.model_dir, "eval")
    save_results(results, pred_rad, true_rad, regions[idx_te],
                 eval_dir, args.num_regions)

    print("\nDone.")


if __name__ == "__main__":
    main()

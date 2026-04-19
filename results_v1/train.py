#!/usr/bin/env python3
"""
train.py

Training script for the seed-conditioned IK model (v3).

Input:  15D = 3D position + 6D rotation (continuous) + 6D seed joints
Output: 12D = sin/cos pairs for 6 joints

Key features:
  - L2 normalization of predicted sin/cos pairs before loss
  - Standard MSE on normalized 12D output vs ground truth
  - ReduceLROnPlateau scheduling
  - Early stopping with patience
  - Gradient clipping

Usage:
    python -m ik_v3.train --csv ur5e_seed_dataset.csv
    python -m ik_v3.train --csv ur5e_seed_dataset.csv --epochs 200 --lr 1e-3
"""

import argparse
import os
import sys
import time
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(1, os.path.dirname(_this_dir))

from model import build_model
from representations import (
    encode_sincos,
    decode_sincos,
    normalize_sincos,
    POSE9D_COLS,
    JOINT_COLS,
    SEED_COLS,
)


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(csv_path: str):
    """Load the seed-conditioned dataset.

    Returns
    -------
    X_pose : (N, 9) — position + 6D rotation
    X_seed : (N, 6) — seed joint angles
    Y_raw  : (N, 6) — ground-truth joint angles (radians)
    """
    df = pd.read_csv(csv_path)

    required = POSE9D_COLS + SEED_COLS + JOINT_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X_pose = df[POSE9D_COLS].values.astype(np.float32)  # (N, 9)
    X_seed = df[SEED_COLS].values.astype(np.float32)    # (N, 6)
    Y_raw = df[JOINT_COLS].values.astype(np.float32)    # (N, 6)

    print(f"Loaded {len(df):,} samples from {csv_path}")
    print(f"  Pose:  {X_pose.shape} (3D pos + 6D rot)")
    print(f"  Seed:  {X_seed.shape} (6 seed joints)")
    print(f"  Joints: {Y_raw.shape} (6 ground-truth joints)")
    return X_pose, X_seed, Y_raw


# ============================================================================
# Loss with L2 normalization
# ============================================================================

class NormalizedSinCosLoss(nn.Module):
    """MSE loss with L2 normalization of predicted sin/cos pairs.

    Before computing loss, each (sin, cos) pair is L2-normalized
    so that sin²+cos²=1. This prevents the network from predicting
    mathematically impossible trigonometric values.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (N, 12) raw model output
        target : (N, 12) ground-truth sin/cos

        Returns
        -------
        loss : scalar MSE between L2-normalized pred and target
        """
        pred_norm = normalize_sincos(pred)
        return self.mse(pred_norm, target)


# ============================================================================
# Angular error metric (handles wraparound)
# ============================================================================

def angular_error(pred_rad: np.ndarray, true_rad: np.ndarray) -> np.ndarray:
    """Shortest angular distance, handling wraparound."""
    diff = pred_rad - true_rad
    return np.arctan2(np.sin(diff), np.cos(diff))


# ============================================================================
# Training
# ============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            args.csv)
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.csv)
    X_pose, X_seed, Y_raw = load_dataset(csv_path)

    # 80/10/10 split
    idx = np.arange(len(X_pose))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.2,
                                        random_state=args.seed)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.5,
                                       random_state=args.seed)
    print(f"Split: train={len(idx_tr):,}, val={len(idx_val):,}, "
          f"test={len(idx_te):,}")

    # Encode target joints as sin/cos
    Y_sc = encode_sincos(Y_raw).astype(np.float32)

    # Scale inputs: standardize pose, standardize seed separately
    pose_scaler = StandardScaler().fit(X_pose[idx_tr])
    seed_scaler = StandardScaler().fit(X_seed[idx_tr])

    X_pose_s = pose_scaler.transform(X_pose).astype(np.float32)
    X_seed_s = seed_scaler.transform(X_seed).astype(np.float32)

    # Concatenate to form 15D input: [pose_9d_scaled, seed_6d_scaled]
    X_all = np.hstack([X_pose_s, X_seed_s])  # (N, 15)
    print(f"Combined input: {X_all.shape[1]}D")

    # DataLoaders
    def make_loader(idx, shuffle=True):
        ds = TensorDataset(
            torch.from_numpy(X_all[idx]),
            torch.from_numpy(Y_sc[idx]),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0,
                          pin_memory=torch.cuda.is_available())

    train_loader = make_loader(idx_tr, shuffle=True)
    val_loader = make_loader(idx_val, shuffle=False)

    # Build model
    model = build_model()
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # Loss
    criterion = NormalizedSinCosLoss()

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_mae_rad": [], "val_mae_deg": [], "lr": []}

    t0 = time.time()
    print(f"\nTraining for up to {args.epochs} epochs")
    print(f"  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Patience: {args.patience}  |  Grad clip: {args.grad_clip}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_count += xb.size(0)
        train_loss = train_loss_sum / train_count

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)

                # Normalize before decoding
                pred_norm = normalize_sincos(pred)
                val_preds.append(pred_norm.cpu().numpy())
        val_loss = val_loss_sum / val_count

        # Decode and compute angular MAE
        preds_sc = np.vstack(val_preds)
        preds_rad = decode_sincos(preds_sc)
        val_joints = Y_raw[idx_val]
        mae_rad = float(np.mean(np.abs(angular_error(preds_rad, val_joints))))
        mae_deg = np.degrees(mae_rad)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_rad"].append(mae_rad)
        history["val_mae_deg"].append(mae_deg)
        history["lr"].append(lr)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>4d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"mae={mae_rad:.4f} rad ({mae_deg:.2f}°)  "
              f"lr={lr:.1e}  ({elapsed:.1f}s){improved}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # --- Save ---
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
    with open(os.path.join(out_dir, "pose_scaler.pkl"), "wb") as f:
        pickle.dump(pose_scaler, f)
    with open(os.path.join(out_dir, "seed_scaler.pkl"), "wb") as f:
        pickle.dump(seed_scaler, f)
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, "training_history.csv"), index=False)

    # Save test indices for evaluation
    np.save(os.path.join(out_dir, "test_indices.npy"), idx_te)

    # Plots
    plot_history(history, out_dir)

    print(f"Saved to {out_dir}/")

    # --- Quick test-set evaluation ---
    print("\n" + "=" * 60)
    print("Quick test-set evaluation:")
    print("=" * 60)
    test_loader = make_loader(idx_te, shuffle=False)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            pred_norm = normalize_sincos(pred)
            test_preds.append(pred_norm.cpu().numpy())
    test_preds_sc = np.vstack(test_preds)
    test_preds_rad = decode_sincos(test_preds_sc)
    test_joints = Y_raw[idx_te]

    errs = np.abs(angular_error(test_preds_rad, test_joints))
    per_joint_mae = np.mean(errs, axis=0)
    overall_mae = np.mean(errs)
    overall_mae_deg = np.degrees(overall_mae)

    print(f"  Overall MAE: {overall_mae:.4f} rad ({overall_mae_deg:.2f}°)")
    for j in range(6):
        print(f"  Joint {j+1}: {per_joint_mae[j]:.4f} rad "
              f"({np.degrees(per_joint_mae[j]):.2f}°)")

    pct_below_5 = np.mean(np.degrees(errs) < 5) * 100
    pct_below_10 = np.mean(np.degrees(errs) < 10) * 100
    pct_below_1 = np.mean(np.degrees(errs) < 1) * 100
    print(f"  Errors < 1°:  {pct_below_1:.1f}%")
    print(f"  Errors < 5°:  {pct_below_5:.1f}%")
    print(f"  Errors < 10°: {pct_below_10:.1f}%")


def plot_history(history, out_dir):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_mae_deg"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (degrees)")
    axes[1].set_title("Validation MAE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["lr"], color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train seed-conditioned IK model (v3)")
    parser.add_argument("--csv", type=str, default="ur5e_seed_dataset.csv")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()

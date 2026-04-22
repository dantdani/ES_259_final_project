#!/usr/bin/env python3
"""
train_ik_model.py

Train a Residual MLP for UR5e inverse kinematics using:
  - Full pose inputs   (12D: position + rotation matrix)
  - Sin/cos outputs    (12D: sin/cos pairs for 6 joints)
  - ReduceLROnPlateau scheduler
  - Early stopping
  - Decoded angle MAE tracking

Usage:
    python train_ik_model.py
    python train_ik_model.py --csv ur5e_pose_dataset.csv --epochs 150 --hidden_dim 512
"""

import argparse
import math
import os
import pickle
import time

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

from utils_torch_fk import build_ur5e_model_torch, batched_forward_kinematics
from utils_torch_fk import (
    POSE_COLS,
    JOINT_COLS,
    SINCOS_COLS,
    encode_sincos,
    decode_sincos,
    angular_error,
    build_ur5e_model,
    forward_kinematics,
)
from model_fk_loss import build_model

# ============================================================================
# Defaults
# ============================================================================
DEFAULT_CSV = "ur5e_pose_dataset.csv"
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 20
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_BLOCKS = 4
DEFAULT_DROPOUT = 0.0


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Data loading
# ============================================================================

def load_data(csv_path, subset_size=None):
    """Load CSV, validate columns, return X (pose features) and Y (raw joints)."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")

    if subset_size is not None:
        df = df.sample(n=min(subset_size, len(df)), random_state=42)

    missing_pose = [c for c in POSE_COLS if c not in df.columns]
    missing_joint = [c for c in JOINT_COLS if c not in df.columns]
    if missing_pose:
        raise ValueError(f"Missing pose columns: {missing_pose}")
    if missing_joint:
        raise ValueError(f"Missing joint columns: {missing_joint}")

    nan_count = df[POSE_COLS + JOINT_COLS].isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: dropping {nan_count} NaN entries")
        df = df.dropna(subset=POSE_COLS + JOINT_COLS)

    X = df[POSE_COLS].values.astype(np.float32)
    Y_raw = df[JOINT_COLS].values.astype(np.float32)

    print(f"Dataset: {X.shape[0]:,} samples, "
          f"input_dim={X.shape[1]}, output_dim(raw)={Y_raw.shape[1]}")
    return X, Y_raw


# ============================================================================
# Splitting, scaling, encoding
# ============================================================================

def split_data(X, Y, seed=42):
    """80/10/10 split."""
    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X, Y, test_size=0.2, random_state=seed)
    X_val, X_te, Y_val, Y_te = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=seed)
    print(f"Split: train={len(X_tr):,}, val={len(X_val):,}, test={len(X_te):,}")
    return X_tr, X_val, X_te, Y_tr, Y_val, Y_te


def prepare_targets(Y_raw):
    """Encode raw joint angles into sin/cos pairs.

    Returns
    -------
    Y_sincos : ndarray (N, 12)
    """
    return encode_sincos(Y_raw).astype(np.float32)


def fit_input_scaler(X_train):
    """Fit StandardScaler on training inputs only."""
    scaler = StandardScaler().fit(X_train)
    return scaler


def scale_inputs(X, scaler):
    return scaler.transform(X).astype(np.float32)


# ============================================================================
# DataLoader helper
# ============================================================================

def make_loader(X, Y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, val_joints_raw,
                epochs, lr, patience, device, lambda_pos=0.0, scaler=None):
    """Train with Adam + MSE + ReduceLROnPlateau + early stopping.

    Parameters
    ----------
    val_joints_raw : ndarray (N_val, 6) – raw joint angles for decoded MAE tracking

    Returns
    -------
    history : dict
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )


    best_val_loss = float("inf")
    
    ur5e_torch = build_ur5e_model_torch(device)
    if scaler is not None:
        t_mean = torch.tensor(scaler.mean_[:3], device=device, dtype=torch.float32)
        t_scale = torch.tensor(scaler.scale_[:3], device=device, dtype=torch.float32)

    best_state = None
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae_rad": [],
        "lr": [],
    }

    print(f"\nTraining on {device} for up to {epochs} epochs  "
          f"(patience={patience}, lr={lr})")
    print("-" * 80)

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss_joint = criterion(pred, yb)
            loss = loss_joint
            
            if lambda_pos > 0.0:
                pred_rad = torch.atan2(pred[:, 0::2], pred[:, 1::2])
                T_pred = batched_forward_kinematics(pred_rad, ur5e_torch)
                pred_xyz = T_pred[:, :3, 3] # in mm
                true_xyz = xb[:, :3] * t_scale + t_mean
                # Compute MSE on meters to keep scale reasonable
                loss_pos = torch.nn.functional.mse_loss(pred_xyz / 1000.0, true_xyz / 1000.0)
                loss = loss_joint + (lambda_pos * loss_pos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_count += xb.size(0)
        train_loss = train_loss_sum / train_count

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)
                val_preds_all.append(pred.cpu().numpy())
        val_loss = val_loss_sum / val_count

        # Decoded angle MAE
        val_preds_sincos = np.vstack(val_preds_all)
        val_preds_rad = decode_sincos(val_preds_sincos)
        ang_err = np.abs(angular_error(val_preds_rad, val_joints_raw))
        val_mae_rad = float(np.mean(ang_err))

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_rad"].append(val_mae_rad)
        history["lr"].append(current_lr)

        # Early stopping
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>4d}/{epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"mae={val_mae_rad:.4f} rad  "
            f"lr={current_lr:.1e}  ({elapsed:.1f}s){improved}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    return history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, test_loader, test_joints_raw, device, lambda_pos=0.0, scaler=None):
    """Evaluate on test set. Report sin/cos MSE and decoded angle metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(yb.numpy())

    preds_sincos = np.vstack(all_preds)
    targets_sincos = np.vstack(all_targets)

    # Sin/cos space metrics
    sincos_mse = float(np.mean((preds_sincos - targets_sincos) ** 2))

    # Decode to radians
    preds_rad = decode_sincos(preds_sincos)
    targets_rad = test_joints_raw  # already raw radians

    ur5e = build_ur5e_model()
    pos_errors = []
    rot_errors = []
    for i in range(len(preds_rad)):
        T_pred = forward_kinematics(preds_rad[i], ur5e)
        T_true = forward_kinematics(targets_rad[i], ur5e)
        pos_errors.append(np.linalg.norm(T_pred[:3, 3] - T_true[:3, 3]))
        
        R_pred = T_pred[:3, :3]
        R_true = T_true[:3, :3]
        val = (np.trace(R_pred @ R_true.T) - 1) / 2
        val = max(-1.0, min(1.0, val))
        rot_errors.append(math.acos(val))
    
    mean_pos_error_mm = float(np.mean(pos_errors))
    mean_rot_error_rad = float(np.mean(rot_errors))

    # Angular error with wraparound handling
    ang_err = angular_error(preds_rad, targets_rad)
    abs_err = np.abs(ang_err)

    mae = float(np.mean(abs_err))
    per_joint_mae = np.mean(abs_err, axis=0)
    rmse = float(np.sqrt(np.mean(ang_err ** 2)))
    joint_norm = float(np.mean(np.linalg.norm(ang_err, axis=1)))

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    print(f"  Sin/cos MSE          : {sincos_mse:.6f}")
    print(f"  Decoded RMSE (rad)   : {rmse:.6f}  ({np.degrees(rmse):.2f} deg)")
    print(f"  Decoded MAE  (rad)   : {mae:.6f}  ({np.degrees(mae):.2f} deg)")
    print(f"  Mean joint-norm (rad): {joint_norm:.6f}")
    print(f"  Mean FK pos error (mm): {mean_pos_error_mm:.6f}")
    print(f"  Mean FK rot error (rad): {mean_rot_error_rad:.6f}")
    print(f"  Per-joint MAE (rad / deg):")
    for i, name in enumerate(JOINT_COLS):
        print(f"    {name}: {per_joint_mae[i]:.6f} rad  ({np.degrees(per_joint_mae[i]):.2f} deg)")
    print("=" * 70)

    metrics = {
        "sincos_mse": sincos_mse,
        "mae_rad": mae,
        "rmse_rad": rmse,
        "joint_norm_rad": joint_norm,
        "mean_pos_error_mm": mean_pos_error_mm,
        "mean_rot_error_rad": mean_rot_error_rad,
        "per_joint_mae_rad": dict(zip(JOINT_COLS, per_joint_mae.tolist())),
        "per_joint_mae_deg": dict(zip(JOINT_COLS, np.degrees(per_joint_mae).tolist())),
    }
    return metrics, preds_rad, targets_rad


# ============================================================================
# Saving
# ============================================================================

def save_artifacts(model, input_scaler, history, metrics, out_dir, prefix="ik_pose"):
    os.makedirs(out_dir, exist_ok=True)

    # Model weights
    model_path = os.path.join(out_dir, f"{prefix}_best.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model   -> {model_path}")

    # Input scaler
    sc_path = os.path.join(out_dir, "input_scaler.pkl")
    with open(sc_path, "wb") as f:
        pickle.dump(input_scaler, f)
    print(f"Saved scaler  -> {sc_path}")

    # History
    hist_path = os.path.join(out_dir, "training_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"Saved history -> {hist_path}")

    # Metrics
    met_path = os.path.join(out_dir, "test_metrics.txt")
    with open(met_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved metrics -> {met_path}")

    # Loss curve
    plot_loss_and_lr(history, out_dir)

    # Pred vs true for each joint
    print("Saved plots   -> loss_curve.png, lr_schedule.png")


def plot_loss_and_lr(history, out_dir):
    """Plot training/val loss and LR schedule."""
    epochs = history["epoch"]

    # Loss curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (sin/cos)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # MAE curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["val_mae_rad"], label="Val MAE (rad)", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (rad)")
    ax.set_title("Validation Decoded Angle MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_mae_curve.png"), dpi=150)
    plt.close(fig)

    # LR schedule
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(epochs, history["lr"], color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lr_schedule.png"), dpi=150)
    plt.close(fig)


def plot_pred_vs_true(preds, targets, out_dir):
    """Scatter plots of predicted vs true for all 6 joints."""
    for i in range(6):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(targets[:, i], preds[:, i], s=2, alpha=0.3, edgecolors="none")
        lims = [-np.pi - 0.2, np.pi + 0.2]
        ax.plot(lims, lims, "r--", linewidth=1, label="perfect")
        ax.set_xlabel(f"True q{i+1} (rad)")
        ax.set_ylabel(f"Predicted q{i+1} (rad)")
        ax.set_title(f"Predicted vs True — q{i+1}")
        ax.legend()
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pred_vs_true_q{i+1}.png"), dpi=150)
        plt.close(fig)
    print(f"Saved pred-vs-true plots -> {out_dir}/pred_vs_true_q*.png")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Residual MLP for UR5e full-pose IK.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num_blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--out_dir", type=str, default="pose_results")
    parser.add_argument("--arch", type=str, choices=["resmlp", "mlp"], default="resmlp")
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--lambda_pos", type=float, default=0.0, help="Lambda for task-space position loss.")
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Load data ---
    X, Y_raw = load_data(args.csv, subset_size=args.subset_size)

    # --- Split ---
    X_tr, X_val, X_te, Y_tr_raw, Y_val_raw, Y_te_raw = split_data(X, Y_raw, args.seed)

    # --- Encode targets to sin/cos ---
    Y_tr_sc = prepare_targets(Y_tr_raw)
    Y_val_sc = prepare_targets(Y_val_raw)
    Y_te_sc = prepare_targets(Y_te_raw)
    print(f"Target encoding: raw joints (6) -> sin/cos pairs (12)")

    # --- Scale inputs (sin/cos targets are already in [-1, 1]) ---
    input_scaler = fit_input_scaler(X_tr)
    X_tr_s = scale_inputs(X_tr, input_scaler)
    X_val_s = scale_inputs(X_val, input_scaler)
    X_te_s = scale_inputs(X_te, input_scaler)

    # --- DataLoaders ---
    train_loader = make_loader(X_tr_s, Y_tr_sc, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val_s, Y_val_sc, args.batch_size, shuffle=False)
    test_loader = make_loader(X_te_s, Y_te_sc, args.batch_size, shuffle=False)

    # --- Build model ---
    net = build_model(
        input_dim=12, output_dim=12,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        arch=args.arch,
        num_hidden_layers=args.num_hidden_layers
    )

    # --- Train ---
    history = train_model(
        net, train_loader, val_loader, Y_val_raw,
        epochs=args.epochs, lr=args.lr,
        patience=args.patience, device=device, lambda_pos=args.lambda_pos, scaler=input_scaler
    )

    # --- Evaluate ---
    metrics, preds_rad, targets_rad = evaluate_model(
        net, test_loader, Y_te_raw, device
    )

    # --- Save ---
    save_artifacts(net, input_scaler, history, metrics, args.out_dir)
    plot_pred_vs_true(preds_rad, targets_rad, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

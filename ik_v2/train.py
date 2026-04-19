#!/usr/bin/env python3
"""
train.py

Unified training script for all IK model variants.

Experiments:
  1. global        – Model A: stronger global residual MLP
  2. global_region – Model B: global + region ID embedding
  3. expert        – Model C: one expert per octant
  4. moe           – Model D: mixture-of-experts
  5. refinement    – Model E: iterative refinement

Each can optionally use FK consistency loss.

Usage:
    python -m ik_v2.train --model global --csv ur5e_region_dataset.csv
    python -m ik_v2.train --model expert --fk_loss --csv ur5e_region_dataset.csv
    python -m ik_v2.train --model refinement --fk_loss --csv ur5e_region_dataset.csv
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
from torch.utils.data import DataLoader, TensorDataset, Subset

# Add parent for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import encode_sincos, decode_sincos, angular_error

from pose_representation import pose_columns, pose_dim
from model_global import build_global_model
from model_expert import build_expert_ensemble
from model_moe import build_moe_model
from model_refinement import build_refinement_model
from fk_loss import FKConsistencyLoss, decode_sincos_torch


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(csv_path: str, pose_mode: str):
    """Load region-labelled dataset.

    Returns X_pose, region_ids, Y_joints_raw
    """
    df = pd.read_csv(csv_path)
    p_cols = pose_columns(pose_mode)
    j_cols = [f"q{i+1}" for i in range(6)]

    missing = [c for c in p_cols + ["region_id"] + j_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[p_cols].values.astype(np.float32)
    regions = df["region_id"].values.astype(np.int64)
    Y = df[j_cols].values.astype(np.float32)

    print(f"Loaded {len(df):,} samples from {csv_path}")
    print(f"  Pose mode: {pose_mode} ({X.shape[1]}D)")
    print(f"  Regions: {np.unique(regions).tolist()}")
    for r in np.unique(regions):
        print(f"    region {r}: {(regions == r).sum():,}")
    return X, regions, Y


def split_data(X, regions, Y, seed=42):
    """80/10/10 stratified split."""
    idx = np.arange(len(X))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.2,
                                        random_state=seed,
                                        stratify=regions)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.5,
                                       random_state=seed,
                                       stratify=regions[idx_tmp])
    print(f"Split: train={len(idx_tr):,}, val={len(idx_val):,}, "
          f"test={len(idx_te):,}")
    return idx_tr, idx_val, idx_te


# ============================================================================
# DataLoader builders
# ============================================================================

def make_loader(X, regions, Y_sincos, batch_size, shuffle=True):
    """Create a DataLoader with (pose, region_id, target_sincos)."""
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(regions),
        torch.from_numpy(Y_sincos),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0,
                      pin_memory=torch.cuda.is_available())


def make_region_loaders(X, regions, Y_sincos, batch_size,
                        num_regions, shuffle=True):
    """Create per-region DataLoaders for expert training."""
    loaders = {}
    for r in range(num_regions):
        mask = regions == r
        if mask.sum() == 0:
            continue
        ds = TensorDataset(
            torch.from_numpy(X[mask]),
            torch.from_numpy(regions[mask]),
            torch.from_numpy(Y_sincos[mask]),
        )
        loaders[r] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                num_workers=0)
    return loaders


# ============================================================================
# Training loops
# ============================================================================

def train_global(model, train_loader, val_loader, val_joints_raw,
                 args, device, use_region=False):
    """Train a global model (Model A or B)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    if args.fk_loss:
        criterion = FKConsistencyLoss(
            lambda_joint=args.lambda_joint,
            lambda_pos=args.lambda_pos,
            lambda_rot=args.lambda_rot)
    else:
        criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_mae_rad": [], "lr": []}
    if args.fk_loss:
        history.update({"fk_pos": [], "fk_rot": []})

    t0 = time.time()
    print(f"\nTraining '{args.model}' on {device} for up to {args.epochs} epochs")
    print(f"  FK loss: {args.fk_loss}  |  use_region: {use_region}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, rb, yb in train_loader:
            xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
            if use_region:
                pred = model(xb, rb)
            else:
                pred = model(xb)

            if args.fk_loss:
                losses = criterion(pred, yb, xb)
                loss = losses["total"]
            else:
                loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_count += xb.size(0)
        train_loss = train_loss_sum / train_count

        # -- Validation --
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds = []
        fk_pos_sum = 0.0
        fk_rot_sum = 0.0
        with torch.no_grad():
            for xb, rb, yb in val_loader:
                xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
                if use_region:
                    pred = model(xb, rb)
                else:
                    pred = model(xb)

                if args.fk_loss:
                    losses = criterion(pred, yb, xb)
                    loss = losses["total"]
                    fk_pos_sum += losses["pos"].item() * xb.size(0)
                    fk_rot_sum += losses["rot"].item() * xb.size(0)
                else:
                    loss = criterion(pred, yb)

                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)
                val_preds.append(pred.cpu().numpy())
        val_loss = val_loss_sum / val_count

        # Decoded MAE
        preds_sc = np.vstack(val_preds)
        preds_rad = decode_sincos(preds_sc)
        mae = float(np.mean(np.abs(angular_error(preds_rad, val_joints_raw))))

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_rad"].append(mae)
        history["lr"].append(lr)
        if args.fk_loss:
            history["fk_pos"].append(fk_pos_sum / val_count)
            history["fk_rot"].append(fk_rot_sum / val_count)

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
        extra = ""
        if args.fk_loss:
            extra = f"  fk_pos={fk_pos_sum/val_count:.4f}"
        print(f"Epoch {epoch:>4d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"mae={mae:.4f} rad  lr={lr:.1e}{extra}  "
              f"({elapsed:.1f}s){improved}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation loss: {best_val_loss:.6f}")
    return history


def train_experts(model, train_loaders, val_loaders,
                  val_joints_per_region, args, device):
    """Train each expert independently on its region data."""
    model.to(device)
    all_histories = {}

    if args.fk_loss:
        criterion = FKConsistencyLoss(
            lambda_joint=args.lambda_joint,
            lambda_pos=args.lambda_pos,
            lambda_rot=args.lambda_rot)
    else:
        criterion = nn.MSELoss()

    for r in sorted(train_loaders.keys()):
        print(f"\n{'='*60}")
        print(f"Training Expert {r}  "
              f"(train={len(train_loaders[r].dataset):,}, "
              f"val={len(val_loaders[r].dataset):,})")
        print(f"{'='*60}")

        expert = model.experts[r]
        expert.to(device)
        optimizer = torch.optim.Adam(expert.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

        best_val = float("inf")
        best_state = None
        patience_counter = 0
        history = {"epoch": [], "train_loss": [], "val_loss": [],
                   "val_mae_rad": [], "lr": []}
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            expert.train()
            tloss = 0.0
            tcount = 0
            for xb, rb, yb in train_loaders[r]:
                xb, yb = xb.to(device), yb.to(device)
                pred = expert(xb)
                if args.fk_loss:
                    losses = criterion(pred, yb, xb)
                    loss = losses["total"]
                else:
                    loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(expert.parameters(), args.grad_clip)
                optimizer.step()
                tloss += loss.item() * xb.size(0)
                tcount += xb.size(0)
            tloss /= tcount

            expert.eval()
            vloss = 0.0
            vcount = 0
            vpreds = []
            with torch.no_grad():
                for xb, rb, yb in val_loaders[r]:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = expert(xb)
                    if args.fk_loss:
                        losses = criterion(pred, yb, xb)
                        loss = losses["total"]
                    else:
                        loss = criterion(pred, yb)
                    vloss += loss.item() * xb.size(0)
                    vcount += xb.size(0)
                    vpreds.append(pred.cpu().numpy())
            vloss /= vcount

            psc = np.vstack(vpreds)
            prad = decode_sincos(psc)
            mae = float(np.mean(np.abs(
                angular_error(prad, val_joints_per_region[r]))))

            scheduler.step(vloss)
            lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch)
            history["train_loss"].append(tloss)
            history["val_loss"].append(vloss)
            history["val_mae_rad"].append(mae)
            history["lr"].append(lr)

            improved = ""
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.cpu().clone()
                              for k, v in expert.state_dict().items()}
                patience_counter = 0
                improved = " *"
            else:
                patience_counter += 1

            elapsed = time.time() - t0
            print(f"  E{r} Epoch {epoch:>3d}/{args.epochs}  "
                  f"train={tloss:.6f}  val={vloss:.6f}  "
                  f"mae={mae:.4f}  lr={lr:.1e}  ({elapsed:.1f}s){improved}")

            if patience_counter >= args.patience:
                print(f"  E{r} early stop at epoch {epoch}")
                break

        if best_state is not None:
            expert.load_state_dict(best_state)
        all_histories[r] = history
        print(f"  E{r} best val: {best_val:.6f}")

    return all_histories


def train_refinement(model, train_loader, val_loader,
                     val_joints_raw, args, device):
    """Train iterative refinement model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    if args.fk_loss:
        criterion = FKConsistencyLoss(
            lambda_joint=args.lambda_joint,
            lambda_pos=args.lambda_pos,
            lambda_rot=args.lambda_rot)
    else:
        criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_mae_rad": [], "lr": []}
    t0 = time.time()

    print(f"\nTraining refinement model on {device}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tloss = 0.0
        tcount = 0
        for xb, rb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            final, intermediates = model(xb, return_intermediates=True)

            # Loss on all stages (weighted: later stages matter more)
            total_loss = torch.tensor(0.0, device=device)
            num_stages = len(intermediates)
            for s, pred in enumerate(intermediates):
                w = (s + 1) / num_stages  # increasing weight
                if args.fk_loss:
                    losses = criterion(pred, yb, xb)
                    total_loss = total_loss + w * losses["total"]
                else:
                    total_loss = total_loss + w * criterion(pred, yb)

            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tloss += total_loss.item() * xb.size(0)
            tcount += xb.size(0)
        tloss /= tcount

        model.eval()
        vloss = 0.0
        vcount = 0
        vpreds = []
        with torch.no_grad():
            for xb, rb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                if args.fk_loss:
                    losses = criterion(pred, yb, xb)
                    loss = losses["total"]
                else:
                    loss = criterion(pred, yb)
                vloss += loss.item() * xb.size(0)
                vcount += xb.size(0)
                vpreds.append(pred.cpu().numpy())
        vloss /= vcount

        psc = np.vstack(vpreds)
        prad = decode_sincos(psc)
        mae = float(np.mean(np.abs(angular_error(prad, val_joints_raw))))

        scheduler.step(vloss)
        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(tloss)
        history["val_loss"].append(vloss)
        history["val_mae_rad"].append(mae)
        history["lr"].append(lr)

        improved = ""
        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>4d}/{args.epochs}  "
              f"train={tloss:.6f}  val={vloss:.6f}  "
              f"mae={mae:.4f} rad  lr={lr:.1e}  ({elapsed:.1f}s){improved}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation loss: {best_val_loss:.6f}")
    return history


# ============================================================================
# Save artifacts
# ============================================================================

def save_all(model, scaler, history, out_dir, model_type):
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
    with open(os.path.join(out_dir, "input_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    if isinstance(history, dict) and "epoch" in history:
        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, "training_history.csv"), index=False)
        plot_history(history, out_dir)
    elif isinstance(history, dict):
        # Expert histories — dict of {region: history_dict}
        for r, h in history.items():
            pd.DataFrame(h).to_csv(
                os.path.join(out_dir, f"history_expert_{r}.csv"), index=False)
            plot_history(h, out_dir, suffix=f"_expert_{r}")

    # Save model config
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        f.write(f"model_type: {model_type}\n")
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"trainable_params: {n}\n")

    print(f"\nSaved to {out_dir}/")


def plot_history(history, out_dir, suffix=""):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_mae_rad"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (rad)")
    axes[1].set_title("Val MAE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["lr"], color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"training_curves{suffix}.png"), dpi=150)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train IK models (v2)")

    # Data
    parser.add_argument("--csv", type=str, default="ur5e_region_dataset.csv")
    parser.add_argument("--pose_mode", type=str, default="quat",
                        choices=["quat", "rotmat", "axisangle"])

    # Model selection
    parser.add_argument("--model", type=str, default="global",
                        choices=["global", "global_region", "expert",
                                 "moe", "refinement"])

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--norm", type=str, default="layer",
                        choices=["layer", "batch"])
    parser.add_argument("--num_regions", type=int, default=8)
    parser.add_argument("--region_embed_dim", type=int, default=16)

    # MoE specific
    parser.add_argument("--expert_hidden", type=int, default=256)
    parser.add_argument("--expert_blocks", type=int, default=4)
    parser.add_argument("--gate_hidden", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=0)

    # Refinement specific
    parser.add_argument("--refine_hidden", type=int, default=256)
    parser.add_argument("--refine_blocks", type=int, default=3)
    parser.add_argument("--num_refine", type=int, default=2)
    parser.add_argument("--share_refine", action="store_true", default=True)

    # FK loss
    parser.add_argument("--fk_loss", action="store_true", default=False)
    parser.add_argument("--lambda_joint", type=float, default=1.0)
    parser.add_argument("--lambda_pos", type=float, default=0.01)
    parser.add_argument("--lambda_rot", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: results/<model>)")

    args = parser.parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output dir
    if args.out_dir is None:
        suffix = "_fk" if args.fk_loss else ""
        args.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", f"{args.model}{suffix}")

    # Load data
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            args.csv)
    if not os.path.isfile(csv_path):
        # Try parent dir
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.csv)
    X, regions, Y_raw = load_dataset(csv_path, args.pose_mode)

    # Split
    idx_tr, idx_val, idx_te = split_data(X, regions, Y_raw, args.seed)

    # Encode targets
    Y_sc = encode_sincos(Y_raw).astype(np.float32)

    # Scale inputs
    scaler = StandardScaler().fit(X[idx_tr])
    X_s = scaler.transform(X).astype(np.float32)
    print(f"Target: sin/cos (12D)")

    input_dim = pose_dim(args.pose_mode)

    # ================================================================
    # Build model and train
    # ================================================================

    if args.model == "global":
        model = build_global_model(
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks, dropout=args.dropout,
            norm=args.norm)

        train_loader = make_loader(X_s[idx_tr], regions[idx_tr],
                                   Y_sc[idx_tr], args.batch_size)
        val_loader = make_loader(X_s[idx_val], regions[idx_val],
                                 Y_sc[idx_val], args.batch_size,
                                 shuffle=False)
        history = train_global(model, train_loader, val_loader,
                               Y_raw[idx_val], args, device,
                               use_region=False)
        save_all(model, scaler, history, args.out_dir, "global")

    elif args.model == "global_region":
        model = build_global_model(
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks, dropout=args.dropout,
            norm=args.norm, num_regions=args.num_regions,
            region_embed_dim=args.region_embed_dim)

        train_loader = make_loader(X_s[idx_tr], regions[idx_tr],
                                   Y_sc[idx_tr], args.batch_size)
        val_loader = make_loader(X_s[idx_val], regions[idx_val],
                                 Y_sc[idx_val], args.batch_size,
                                 shuffle=False)
        history = train_global(model, train_loader, val_loader,
                               Y_raw[idx_val], args, device,
                               use_region=True)
        save_all(model, scaler, history, args.out_dir, "global_region")

    elif args.model == "expert":
        model = build_expert_ensemble(
            num_experts=args.num_regions, input_dim=input_dim,
            hidden_dim=args.expert_hidden, num_blocks=args.expert_blocks,
            dropout=args.dropout, norm=args.norm)

        train_loaders = make_region_loaders(
            X_s[idx_tr], regions[idx_tr], Y_sc[idx_tr],
            args.batch_size, args.num_regions)
        val_loaders = make_region_loaders(
            X_s[idx_val], regions[idx_val], Y_sc[idx_val],
            args.batch_size, args.num_regions, shuffle=False)

        val_joints_per_region = {}
        for r in range(args.num_regions):
            mask = regions[idx_val] == r
            if mask.sum() > 0:
                val_joints_per_region[r] = Y_raw[idx_val][mask]

        history = train_experts(model, train_loaders, val_loaders,
                                val_joints_per_region, args, device)
        save_all(model, scaler, history, args.out_dir, "expert")

    elif args.model == "moe":
        model = build_moe_model(
            num_experts=args.num_regions, input_dim=input_dim,
            expert_hidden=args.expert_hidden, expert_blocks=args.expert_blocks,
            gate_hidden=args.gate_hidden, top_k=args.top_k,
            dropout=args.dropout, norm=args.norm)

        train_loader = make_loader(X_s[idx_tr], regions[idx_tr],
                                   Y_sc[idx_tr], args.batch_size)
        val_loader = make_loader(X_s[idx_val], regions[idx_val],
                                 Y_sc[idx_val], args.batch_size,
                                 shuffle=False)
        history = train_global(model, train_loader, val_loader,
                               Y_raw[idx_val], args, device,
                               use_region=False)
        save_all(model, scaler, history, args.out_dir, "moe")

    elif args.model == "refinement":
        model = build_refinement_model(
            pose_dim=input_dim, init_hidden=args.hidden_dim,
            init_blocks=args.num_blocks,
            refine_hidden=args.refine_hidden,
            refine_blocks=args.refine_blocks,
            num_refine=args.num_refine,
            share_refine=args.share_refine,
            dropout=args.dropout, norm=args.norm)

        train_loader = make_loader(X_s[idx_tr], regions[idx_tr],
                                   Y_sc[idx_tr], args.batch_size)
        val_loader = make_loader(X_s[idx_val], regions[idx_val],
                                 Y_sc[idx_val], args.batch_size,
                                 shuffle=False)
        history = train_refinement(model, train_loader, val_loader,
                                   Y_raw[idx_val], args, device)
        save_all(model, scaler, history, args.out_dir, "refinement")

    print("\nDone.")


if __name__ == "__main__":
    main()

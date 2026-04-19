#!/usr/bin/env python3
"""
infer_ik.py

Clean inference interface: given an end-effector pose, predict joint angles.

Usage:
    # Interactive mode
    python infer_ik.py

    # Command-line mode (position + flattened rotation matrix)
    python infer_ik.py --pose 100 200 300  1 0 0  0 1 0  0 0 1

    # From file (one pose per line, 12 values each)
    python infer_ik.py --file poses.txt
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

from utils import (
    POSE_COLS,
    JOINT_COLS,
    decode_sincos,
    build_ur5e_model,
    forward_kinematics,
)
from model import build_model


class IKInference:
    """Wrapper for loading model + scaler and running inference."""

    def __init__(self, model_path="pose_results/ik_pose_best.pt",
                 scaler_path="pose_results/input_scaler.pkl",
                 hidden_dim=256, num_blocks=4, dropout=0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.net = build_model(input_dim=12, output_dim=12,
                               hidden_dim=hidden_dim,
                               num_blocks=num_blocks,
                               dropout=dropout)
        self.net.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.net.to(self.device)
        self.net.eval()

        # Load scaler
        with open(scaler_path, "rb") as f:
            self.input_scaler = pickle.load(f)

        # FK model for verification
        self.S, self.M = build_ur5e_model()

    def predict(self, pose):
        """Predict joint angles from a single pose.

        Parameters
        ----------
        pose : array-like, shape (12,)
            [x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33]

        Returns
        -------
        q : ndarray, shape (6,)
            Joint angles in radians.
        """
        pose = np.asarray(pose, dtype=np.float32).reshape(1, -1)
        if pose.shape[1] != 12:
            raise ValueError(f"Expected 12 pose features, got {pose.shape[1]}")

        pose_scaled = self.input_scaler.transform(pose).astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(pose_scaled).to(self.device)
            sincos = self.net(x).cpu().numpy()

        q = decode_sincos(sincos)[0]
        return q

    def predict_batch(self, poses):
        """Predict joint angles from multiple poses.

        Parameters
        ----------
        poses : array-like, shape (N, 12)

        Returns
        -------
        q : ndarray, shape (N, 6)
        """
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)

        poses_scaled = self.input_scaler.transform(poses).astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(poses_scaled).to(self.device)
            sincos = self.net(x).cpu().numpy()

        return decode_sincos(sincos)

    def verify(self, q):
        """Run FK on predicted joints and return the resulting pose.

        Parameters
        ----------
        q : array-like, shape (6,)

        Returns
        -------
        T : ndarray, shape (4, 4) – homogeneous transformation
        """
        return forward_kinematics(self.S, self.M, np.asarray(q))

    def predict_and_verify(self, pose):
        """Predict IK and show FK reconstruction.

        Parameters
        ----------
        pose : array-like, shape (12,)

        Returns
        -------
        q : ndarray (6,)
        T_reconstructed : ndarray (4, 4)
        pos_error : float – Euclidean position error (mm)
        """
        q = self.predict(pose)
        T = self.verify(q)

        pose = np.asarray(pose, dtype=np.float64)
        pos_error = np.linalg.norm(T[:3, 3] - pose[:3])
        return q, T, pos_error


def interactive_mode(engine):
    """Interactive REPL."""
    print("\n=== UR5e IK Inference (interactive) ===")
    print("Enter 12 values: x y z r11 r12 r13 r21 r22 r23 r31 r32 r33")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            line = input("pose> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.lower() in ("quit", "exit", "q"):
            break
        if not line:
            continue

        try:
            vals = [float(v) for v in line.split()]
            if len(vals) != 12:
                print(f"  Expected 12 values, got {len(vals)}. Try again.")
                continue

            q, T, pos_err = engine.predict_and_verify(vals)

            print("  Predicted joints (rad):")
            for i in range(6):
                print(f"    q{i+1} = {q[i]:+.6f}  ({np.degrees(q[i]):+.2f} deg)")
            print(f"  FK position error: {pos_err:.2f} mm")
            print()

        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="UR5e IK inference")
    parser.add_argument("--model", default="pose_results/ik_pose_best.pt")
    parser.add_argument("--scaler", default="pose_results/input_scaler.pkl")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pose", type=float, nargs=12, metavar="V",
                        help="12 pose values: x y z r11 r12 r13 r21 r22 r23 r31 r32 r33")
    parser.add_argument("--file", type=str,
                        help="File with one pose per line (12 values)")
    args = parser.parse_args()

    engine = IKInference(
        model_path=args.model,
        scaler_path=args.scaler,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )

    if args.pose is not None:
        q, T, pos_err = engine.predict_and_verify(args.pose)
        print("Predicted joints (rad):")
        for i in range(6):
            print(f"  q{i+1} = {q[i]:+.6f}  ({np.degrees(q[i]):+.2f} deg)")
        print(f"\nFK reconstruction:")
        print(f"  Position: [{T[0,3]:.2f}, {T[1,3]:.2f}, {T[2,3]:.2f}]")
        print(f"  Position error: {pos_err:.2f} mm")

    elif args.file is not None:
        poses = np.loadtxt(args.file)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        q_all = engine.predict_batch(poses)
        print(f"Predicted {len(q_all)} configurations:")
        for i, q in enumerate(q_all):
            deg_str = ", ".join(f"{np.degrees(a):+.2f}" for a in q)
            print(f"  [{i}] {deg_str} deg")

    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()

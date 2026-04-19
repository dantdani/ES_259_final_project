#!/usr/bin/env python3
"""
infer.py

Inference module for the seed-conditioned IK model (v3).

Given a target pose (4×4 transform or 9D vector) and the robot's current
joint angles (seed), predicts the closest IK solution.

Usage:
    # As a module
    from ik_v3.infer import IKSolver
    solver = IKSolver("ik_v3/results")
    joints = solver.solve(T_target, current_joints)

    # CLI demo
    python -m ik_v3.infer --model_dir ik_v3/results --num_samples 20
"""

import argparse
import os
import sys
import pickle

import numpy as np
import torch

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(1, os.path.dirname(_this_dir))
from utils import build_ur5e_model, forward_kinematics

from model import SeedConditionedIKModel
from representations import (
    extract_pose_9d,
    normalize_sincos,
    decode_sincos_torch,
    decode_sincos,
)


class IKSolver:
    """Seed-conditioned neural IK solver.

    Loads a trained model and scalers, provides a clean API for
    solving IK given a target pose and current joint state.
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.ur5e = build_ur5e_model()

        # Load model
        self.model = SeedConditionedIKModel()
        state = torch.load(
            os.path.join(model_dir, "model_best.pt"),
            map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # Load scalers
        with open(os.path.join(model_dir, "pose_scaler.pkl"), "rb") as f:
            self.pose_scaler = pickle.load(f)
        with open(os.path.join(model_dir, "seed_scaler.pkl"), "rb") as f:
            self.seed_scaler = pickle.load(f)

    def solve(self, T_target: np.ndarray,
              current_joints: np.ndarray) -> np.ndarray:
        """Solve IK for a target pose given the current joint state.

        Parameters
        ----------
        T_target : (4, 4) target homogeneous transform
        current_joints : (6,) current joint angles in radians

        Returns
        -------
        joints : (6,) predicted joint angles in radians
        """
        # Extract 9D pose
        pose_9d = extract_pose_9d(T_target).reshape(1, -1).astype(np.float32)
        seed = current_joints.reshape(1, -1).astype(np.float32)

        # Scale
        pose_s = self.pose_scaler.transform(pose_9d).astype(np.float32)
        seed_s = self.seed_scaler.transform(seed).astype(np.float32)

        # Concatenate: 15D
        x = np.hstack([pose_s, seed_s])
        x_t = torch.from_numpy(x).to(self.device)

        # Predict
        with torch.no_grad():
            pred = self.model(x_t)
            pred_norm = normalize_sincos(pred)
            joints = decode_sincos_torch(pred_norm)

        return joints.cpu().numpy().flatten()

    def solve_batch(self, T_targets: np.ndarray,
                    current_joints: np.ndarray) -> np.ndarray:
        """Batch IK solve.

        Parameters
        ----------
        T_targets : (N, 4, 4) target transforms
        current_joints : (N, 6) current joint angles

        Returns
        -------
        joints : (N, 6) predicted joint angles
        """
        pose_9d = extract_pose_9d(T_targets).astype(np.float32)
        seed = current_joints.astype(np.float32)

        pose_s = self.pose_scaler.transform(pose_9d).astype(np.float32)
        seed_s = self.seed_scaler.transform(seed).astype(np.float32)

        x = np.hstack([pose_s, seed_s])
        x_t = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            pred = self.model(x_t)
            pred_norm = normalize_sincos(pred)
            joints = decode_sincos_torch(pred_norm)

        return joints.cpu().numpy()

    def verify(self, T_target: np.ndarray,
               predicted_joints: np.ndarray) -> dict:
        """Verify a prediction by running FK on the predicted joints.

        Returns
        -------
        dict with 'pos_error_mm', 'rot_error_deg', 'T_predicted'
        """
        T_pred = forward_kinematics(predicted_joints, self.ur5e)

        pos_err = np.linalg.norm(T_target[:3, 3] - T_pred[:3, 3])

        R_err = T_target[:3, :3].T @ T_pred[:3, :3]
        cos_angle = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        rot_err_deg = np.degrees(np.arccos(cos_angle))

        return {
            "pos_error_mm": float(pos_err),
            "rot_error_deg": float(rot_err_deg),
            "T_predicted": T_pred,
        }


def demo(args):
    """Run a demo: random FK → IK → FK verification."""
    solver = IKSolver(args.model_dir, device="cpu")
    ur5e = build_ur5e_model()

    print(f"\nDemo: {args.num_samples} random FK → IK → verify")
    print("=" * 70)

    pos_errors = []
    rot_errors = []
    joint_errors = []

    for i in range(args.num_samples):
        # Random ground-truth joints
        q_true = np.random.uniform(-np.pi, np.pi, size=6).astype(np.float32)
        T_target = forward_kinematics(q_true, ur5e)

        # Random seed (nearby the true solution)
        q_seed = q_true + np.random.randn(6).astype(np.float32) * 0.5
        q_seed = np.clip(q_seed, -np.pi, np.pi)

        # Solve
        q_pred = solver.solve(T_target, q_seed)

        # Verify
        result = solver.verify(T_target, q_pred)

        # Joint-level error
        j_err = np.arctan2(np.sin(q_pred - q_true), np.cos(q_pred - q_true))
        j_mae = np.mean(np.abs(j_err))

        pos_errors.append(result["pos_error_mm"])
        rot_errors.append(result["rot_error_deg"])
        joint_errors.append(j_mae)

        if i < 10 or (i + 1) == args.num_samples:
            print(f"  [{i+1}] pos_err={result['pos_error_mm']:.2f} mm  "
                  f"rot_err={result['rot_error_deg']:.2f}°  "
                  f"joint_mae={np.degrees(j_mae):.2f}°")

    print(f"\nSummary ({args.num_samples} samples):")
    print(f"  Position error:  mean={np.mean(pos_errors):.2f} mm  "
          f"median={np.median(pos_errors):.2f} mm")
    print(f"  Rotation error:  mean={np.mean(rot_errors):.2f}°  "
          f"median={np.median(rot_errors):.2f}°")
    print(f"  Joint MAE:       mean={np.degrees(np.mean(joint_errors)):.2f}°  "
          f"median={np.degrees(np.median(joint_errors)):.2f}°")


def main():
    parser = argparse.ArgumentParser(description="IK inference & demo (v3)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to results directory")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of demo samples")
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results")

    demo(args)


if __name__ == "__main__":
    main()

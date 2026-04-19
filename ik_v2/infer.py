#!/usr/bin/env python3
"""
infer.py

Inference interface for all IK model types.

Supports:
  - Single pose inference
  - Batch inference from CSV
  - Routing logic for expert / MoE models
  - Optional refinement step
"""

import argparse
import os
import sys
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (build_ur5e_model, forward_kinematics,
                   decode_sincos, encode_sincos)

from pose_representation import (extract_pose, pose_dim, pose_columns,
                                  rotmat_to_quat)
from regioning import octant_label


class IKInference:
    """Unified inference wrapper for all model types.

    Parameters
    ----------
    model_dir    : str – directory with model_best.pt, input_scaler.pkl
    model_type   : str – 'global', 'global_region', 'expert', 'moe', 'refinement'
    pose_mode    : str – 'quat', 'rotmat', 'axisangle'
    device       : str
    model_kwargs : dict – architecture params matching training
    """

    def __init__(self, model_dir: str, model_type: str = "global",
                 pose_mode: str = "quat", device: str = "cpu",
                 **model_kwargs):
        self.model_type = model_type
        self.pose_mode = pose_mode
        self.device = torch.device(device)
        self.ur5e = build_ur5e_model()

        # Load scaler
        with open(os.path.join(model_dir, "input_scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        # Load model
        input_d = pose_dim(pose_mode)
        model = self._build_model(input_d, model_kwargs)
        state = torch.load(os.path.join(model_dir, "model_best.pt"),
                           map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def _build_model(self, input_d, kw):
        if self.model_type in ("global", "global_region"):
            from model_global import GlobalIKModel
            num_reg = kw.get("num_regions", 0) if self.model_type == "global_region" else 0
            return GlobalIKModel(
                input_dim=input_d,
                hidden_dim=kw.get("hidden_dim", 512),
                num_blocks=kw.get("num_blocks", 6),
                num_regions=num_reg,
                region_embed_dim=kw.get("region_embed_dim", 16))
        elif self.model_type == "expert":
            from model_expert import ExpertEnsemble
            return ExpertEnsemble(
                num_experts=kw.get("num_regions", 8),
                input_dim=input_d,
                hidden_dim=kw.get("expert_hidden", 256),
                num_blocks=kw.get("expert_blocks", 4))
        elif self.model_type == "moe":
            from model_moe import MixtureOfExperts
            return MixtureOfExperts(
                num_experts=kw.get("num_regions", 8),
                input_dim=input_d,
                expert_hidden=kw.get("expert_hidden", 256),
                expert_blocks=kw.get("expert_blocks", 3),
                gate_hidden=kw.get("gate_hidden", 128),
                top_k=kw.get("top_k", 0))
        elif self.model_type == "refinement":
            from model_refinement import IterativeIKModel
            return IterativeIKModel(
                pose_dim=input_d,
                init_hidden=kw.get("hidden_dim", 512),
                init_blocks=kw.get("num_blocks", 4),
                refine_hidden=kw.get("refine_hidden", 256),
                refine_blocks=kw.get("refine_blocks", 3),
                num_refine=kw.get("num_refine", 2),
                share_refine=kw.get("share_refine", True))
        raise ValueError(f"Unknown model type: {self.model_type}")

    def predict_pose(self, pose: np.ndarray,
                     region_id: int = None) -> np.ndarray:
        """Predict joint angles from a single pose vector.

        Parameters
        ----------
        pose      : (D,) pose vector
        region_id : int (required for expert / global_region)

        Returns
        -------
        joints : (6,) predicted joint angles in radians
        """
        # Determine region if not provided
        if region_id is None:
            region_id = octant_label(pose[0], pose[1], pose[2])

        x = self.scaler.transform(pose.reshape(1, -1)).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device)
        rt = torch.tensor([region_id], dtype=torch.long, device=self.device)

        with torch.no_grad():
            if self.model_type == "global_region":
                pred = self.model(xt, rt)
            elif self.model_type == "expert":
                pred = self.model(xt, rt)
            elif self.model_type == "refinement":
                pred = self.model(xt)
            else:
                pred = self.model(xt)

        sc = pred.cpu().numpy()
        joints = decode_sincos(sc)[0]
        return joints

    def predict_from_transform(self, T: np.ndarray) -> np.ndarray:
        """Predict joints from a 4x4 transform."""
        pose = extract_pose(T, self.pose_mode)
        return self.predict_pose(pose)

    def predict_batch(self, poses: np.ndarray,
                      region_ids: np.ndarray = None) -> np.ndarray:
        """Batch prediction.

        Parameters
        ----------
        poses      : (N, D) pose vectors
        region_ids : (N,) int, auto-computed if None

        Returns
        -------
        joints : (N, 6)
        """
        if region_ids is None:
            region_ids = np.array([
                octant_label(p[0], p[1], p[2]) for p in poses
            ], dtype=np.int64)

        x = self.scaler.transform(poses).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device)
        rt = torch.from_numpy(region_ids).to(self.device)

        with torch.no_grad():
            if self.model_type == "global_region":
                pred = self.model(xt, rt)
            elif self.model_type == "expert":
                pred = self.model(xt, rt)
            elif self.model_type == "refinement":
                pred = self.model(xt)
            else:
                pred = self.model(xt)

        sc = pred.cpu().numpy()
        return decode_sincos(sc)

    def verify(self, joints: np.ndarray, target_pose: np.ndarray):
        """Run FK on predicted joints and report position/orientation error."""
        T_pred = forward_kinematics(joints, self.ur5e)
        pos_err = np.linalg.norm(T_pred[:3, 3] - target_pose[:3])
        R_pred = T_pred[:3, :3]
        # Simple orientation check via quaternion distance
        q_pred = rotmat_to_quat(R_pred)
        return {
            "position_error_mm": float(pos_err),
            "predicted_joints_rad": joints.tolist(),
            "predicted_joints_deg": np.degrees(joints).tolist(),
            "predicted_position": T_pred[:3, 3].tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description="IK Inference (v2)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="global")
    parser.add_argument("--pose_mode", type=str, default="quat")

    # Architecture (must match training)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--num_regions", type=int, default=8)
    parser.add_argument("--expert_hidden", type=int, default=256)
    parser.add_argument("--expert_blocks", type=int, default=4)
    parser.add_argument("--refine_hidden", type=int, default=256)
    parser.add_argument("--refine_blocks", type=int, default=3)
    parser.add_argument("--num_refine", type=int, default=2)

    # Demo: random test
    parser.add_argument("--demo", action="store_true",
                        help="Run a quick demo with random joint angles")
    args = parser.parse_args()

    kw = {k: v for k, v in vars(args).items()
          if k not in ("model_dir", "model_type", "pose_mode", "demo")}
    ik = IKInference(args.model_dir, args.model_type, args.pose_mode, **kw)

    if args.demo:
        print("Running demo inference...")
        ur5e = build_ur5e_model()
        for i in range(5):
            q_true = np.random.uniform(-np.pi, np.pi, 6)
            T = forward_kinematics(q_true, ur5e)
            pose = extract_pose(T, args.pose_mode)
            q_pred = ik.predict_pose(pose)

            err_rad = np.abs(q_pred - q_true)
            err_deg = np.degrees(err_rad)
            info = ik.verify(q_pred, T[:3, 3])

            print(f"\nSample {i+1}:")
            print(f"  True:  {np.degrees(q_true).round(1)}")
            print(f"  Pred:  {np.degrees(q_pred).round(1)}")
            print(f"  Error: {err_deg.round(2)} deg  "
                  f"(mean={err_deg.mean():.2f}°)")
            print(f"  FK pos error: {info['position_error_mm']:.1f} mm")
    else:
        print("Use --demo for a quick test, or import IKInference in your code.")


if __name__ == "__main__":
    main()

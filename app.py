#!/usr/bin/env python3
"""One-command IK demo app.

Usage:
    python app.py

What it does:
1) Loads the v3 DNN IK model and scalers.
2) Generates a random reachable square trajectory in Cartesian space.
3) Solves each waypoint with DNN IK (seed-conditioned).
4) Refines each waypoint with Newton-Raphson IK.
5) Prints a side-by-side accuracy and convergence report.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

from ik_v3.infer import IKSolver
from utils import build_ur5e_model, forward_kinematics, skew

try:
    import rospy
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    ROS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on ROS runtime
    rospy = None
    JointTrajectory = None
    JointTrajectoryPoint = None
    ROS_IMPORT_ERROR = exc


UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@dataclass
class WaypointResult:
    idx: int
    pos_err_dnn_mm: float
    rot_err_dnn_deg: float
    converged_dnn: bool
    pos_err_ref_mm: float
    rot_err_ref_deg: float
    iters_ref: int
    converged_ref: bool


def make_robot_publisher(topic: str = "/scaled_pos_joint_traj_controller/command"):
    """Create ROS publisher for UR5e joint trajectory commands."""
    if rospy is None:
        raise RuntimeError(f"ROS is not available in this environment: {ROS_IMPORT_ERROR}")

    if not rospy.core.is_initialized():
        rospy.init_node("ik_robot_sender", anonymous=False)

    pub = rospy.Publisher(topic, JointTrajectory, queue_size=10)
    rospy.sleep(0.5)
    return pub


def send_joint_position(pub, q: np.ndarray, duration: float = 5.0) -> None:
    """Publish one UR5e joint trajectory waypoint."""
    msg = JointTrajectory()
    msg.joint_names = list(UR5E_JOINT_NAMES)

    point = JointTrajectoryPoint()
    point.positions = [float(x) for x in np.asarray(q).flatten().tolist()]
    point.velocities = [0.0] * 6
    point.time_from_start = rospy.Duration(duration)

    msg.points.append(point)
    pub.publish(msg)


def space_jacobian(theta: np.ndarray, model: dict) -> np.ndarray:
    """Space-frame Jacobian from screw axes."""
    w = model["w"]
    v = model["v"]
    theta = np.asarray(theta, dtype=np.float64).flatten()
    j = np.zeros((6, 6), dtype=np.float64)

    t_acc = np.eye(4, dtype=np.float64)
    for i in range(6):
        if i == 0:
            j[:3, 0] = v[0]
            j[3:, 0] = w[0]
        else:
            r = t_acc[:3, :3]
            p = t_acc[:3, 3]
            w_new = r @ w[i]
            v_new = r @ v[i] + np.cross(p, w_new)
            j[:3, i] = v_new
            j[3:, i] = w_new

        w_hat = skew(w[i])
        ct = np.cos(theta[i])
        st = np.sin(theta[i])
        r_i = np.eye(3) + st * w_hat + (1 - ct) * (w_hat @ w_hat)
        g_i = np.eye(3) * theta[i] + (1 - ct) * w_hat + (theta[i] - st) * (w_hat @ w_hat)
        p_i = g_i @ v[i]
        t_i = np.eye(4)
        t_i[:3, :3] = r_i
        t_i[:3, 3] = p_i
        t_acc = t_acc @ t_i

    return j


def ik_newton_refine(
    t_target: np.ndarray,
    q_init: np.ndarray,
    model: dict,
    max_iter: int = 200,
    eps_v_mm: float = 1.0,
    eps_w_rad: float = 0.01,
) -> tuple[np.ndarray, int, float, float, bool]:
    """Newton-Raphson IK with damped least-squares updates."""
    q = q_init.astype(np.float64).copy()

    for it in range(1, max_iter + 1):
        t_cur = forward_kinematics(q, model)
        dp = t_target[:3, 3] - t_cur[:3, 3]

        r_err = t_target[:3, :3] @ t_cur[:3, :3].T
        cos_a = np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_a)
        if angle < 1e-10:
            dw = np.zeros(3)
        else:
            s = np.sin(angle)
            dw = (angle / (2.0 * s)) * np.array(
                [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]]
            )

        pos_err = float(np.linalg.norm(dp))
        rot_err = float(angle)
        if pos_err < eps_v_mm and rot_err < eps_w_rad:
            return q, it, pos_err, float(np.degrees(rot_err)), True

        twist = np.concatenate([dp, dw])
        j = space_jacobian(q, model)
        lam = 0.01 * np.linalg.norm(twist)
        dq = np.linalg.solve(j.T @ j + lam * np.eye(6), j.T @ twist)

        max_step = 0.5
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_step:
            dq = dq * (max_step / dq_norm)

        q = np.clip(q + dq, -np.pi, np.pi)

    t_cur = forward_kinematics(q, model)
    dp = t_target[:3, 3] - t_cur[:3, 3]
    r_err = t_target[:3, :3] @ t_cur[:3, :3].T
    angle = np.arccos(np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0))
    return q, max_iter, float(np.linalg.norm(dp)), float(np.degrees(angle)), False


def build_square_targets(center_t: np.ndarray, side_mm: float, steps_per_edge: int) -> list[np.ndarray]:
    """Create a square path around center pose in XY, fixed orientation."""
    c = center_t[:3, 3].copy()
    r = center_t[:3, :3].copy()
    h = side_mm / 2.0

    corners = [
        np.array([c[0] - h, c[1] - h, c[2]]),
        np.array([c[0] + h, c[1] - h, c[2]]),
        np.array([c[0] + h, c[1] + h, c[2]]),
        np.array([c[0] - h, c[1] + h, c[2]]),
    ]

    targets: list[np.ndarray] = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        for k in range(steps_per_edge):
            alpha = k / float(steps_per_edge)
            p = (1.0 - alpha) * p0 + alpha * p1
            t = np.eye(4)
            t[:3, :3] = r
            t[:3, 3] = p
            targets.append(t)
    return targets


def run_demo(
    model_dir: str,
    side_mm: float,
    steps_per_edge: int,
    seed: int,
    send_to_robot: bool = False,
    move_duration: float = 5.0,
) -> int:
    np.random.seed(seed)
    ur5e = build_ur5e_model()
    solver = IKSolver(model_dir=model_dir, device="cpu")

    # Pick a random reachable center from FK of a random valid configuration.
    q_center = np.random.uniform(-np.pi, np.pi, size=6).astype(np.float32)
    t_center = forward_kinematics(q_center, ur5e)

    targets = build_square_targets(t_center, side_mm=side_mm, steps_per_edge=steps_per_edge)

    print("\n=== ES259 One-Command IK Demo ===")
    print(f"Model dir: {model_dir}")
    print(f"Random seed: {seed}")
    print(f"Square side: {side_mm:.1f} mm | Steps/edge: {steps_per_edge} | Waypoints: {len(targets)}")
    print("Comparison:")
    print("  WITHOUT IK  = DNN-only output")
    print("  WITH IK     = DNN output + Newton refinement")
    print("  Convergence tolerance: position < 1.0 mm AND rotation < 0.57 deg")

    pub = None
    if send_to_robot:
        print("WARNING: Sending IK waypoints to the real robot. Make sure the workspace is clear.")
        pub = make_robot_publisher(topic="/scaled_pos_joint_traj_controller/command")

    q_seed = q_center.copy()
    results: list[WaypointResult] = []
    start = time.time()

    for i, t_target in enumerate(targets, start=1):
        q_dnn = solver.solve(t_target, q_seed)
        dnn_check = solver.verify(t_target, q_dnn)
        dnn_ok = (dnn_check["pos_error_mm"] < 1.0) and (dnn_check["rot_error_deg"] < np.degrees(0.01))

        q_ref, it_ref, pe_ref, re_ref, ok_ref = ik_newton_refine(
            t_target,
            q_dnn,
            ur5e,
            max_iter=200,
            eps_v_mm=1.0,
            eps_w_rad=0.01,
        )

        if send_to_robot:
            if ok_ref:
                q_print = [float(v) for v in np.asarray(q_ref).flatten().tolist()]
                print(f"[robot] waypoint {i:02d}: publishing q_ref = {q_print}")
                send_joint_position(pub, q_ref, duration=move_duration)
                rospy.sleep(move_duration + 0.5)
            else:
                print(f"[robot] waypoint {i:02d}: refinement failed, skipping publish")

        results.append(
            WaypointResult(
                idx=i,
                pos_err_dnn_mm=float(dnn_check["pos_error_mm"]),
                rot_err_dnn_deg=float(dnn_check["rot_error_deg"]),
                converged_dnn=bool(dnn_ok),
                pos_err_ref_mm=float(pe_ref),
                rot_err_ref_deg=float(re_ref),
                iters_ref=int(it_ref),
                converged_ref=bool(ok_ref),
            )
        )

        q_seed = q_ref

    elapsed = time.time() - start

    pos_dnn = np.array([r.pos_err_dnn_mm for r in results], dtype=float)
    rot_dnn = np.array([r.rot_err_dnn_deg for r in results], dtype=float)
    conv_dnn = np.array([r.converged_dnn for r in results], dtype=bool)
    pos_ref = np.array([r.pos_err_ref_mm for r in results], dtype=float)
    rot_ref = np.array([r.rot_err_ref_deg for r in results], dtype=float)
    its_ref = np.array([r.iters_ref for r in results], dtype=float)
    conv_ref = np.array([r.converged_ref for r in results], dtype=bool)

    print("\nPer-waypoint (first 10):")
    for r in results[:10]:
        dnn_flag = "OK" if r.converged_dnn else "FAIL"
        ref_flag = "OK" if r.converged_ref else "FAIL"
        print(
            f"  wp {r.idx:02d} | DNN: pos={r.pos_err_dnn_mm:7.2f} mm rot={r.rot_err_dnn_deg:6.2f} deg"
            f" {dnn_flag} | IK refine: pos={r.pos_err_ref_mm:7.3f} mm rot={r.rot_err_ref_deg:6.3f} deg"
            f" iters={r.iters_ref:3d} {ref_flag}"
        )

    print("\nSummary:")
    print("  WITHOUT IK (DNN-only):")
    print(f"    Mean pos err: {np.mean(pos_dnn):.2f} mm | Mean rot err: {np.mean(rot_dnn):.2f} deg")
    print(f"    Convergence:  {np.sum(conv_dnn)}/{len(conv_dnn)} ({100.0*np.mean(conv_dnn):.1f}%)")
    print("  WITH IK (DNN + Newton refinement):")
    print(f"    Mean pos err: {np.mean(pos_ref):.3f} mm | Mean rot err: {np.mean(rot_ref):.3f} deg")
    print(f"    Convergence:  {np.sum(conv_ref)}/{len(conv_ref)} ({100.0*np.mean(conv_ref):.1f}%)")
    print(f"    Mean iterations (Newton): {np.mean(its_ref):.1f}")
    print(f"  Elapsed: {elapsed:.2f}s")

    if np.all(conv_ref):
        print("\nPASS: Square trajectory verified with both DNN IK and Newton IK refinement.")
        return 0
    print("\nWARN: Some waypoints did not meet IK refinement tolerance.")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command DNN+IK square trajectory demo")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ik_v3", "results"),
        help="Path containing model_best.pt, pose_scaler.pkl, seed_scaler.pkl",
    )
    parser.add_argument("--side_mm", type=float, default=60.0, help="Square side length in mm")
    parser.add_argument("--steps_per_edge", type=int, default=10, help="Waypoints per edge")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--send_to_robot",
        action="store_true",
        help="Publish refined IK waypoints to /scaled_pos_joint_traj_controller/command",
    )
    parser.add_argument(
        "--move_duration",
        type=float,
        default=5.0,
        help="Trajectory point time_from_start in seconds when sending to robot",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isdir(args.model_dir):
        print(f"ERROR: model directory not found: {args.model_dir}")
        return 2
    if args.send_to_robot and rospy is None:
        print(f"ERROR: --send_to_robot was requested but ROS imports are unavailable: {ROS_IMPORT_ERROR}")
        return 2
    required = ["model_best.pt", "pose_scaler.pkl", "seed_scaler.pkl"]
    missing = [name for name in required if not os.path.exists(os.path.join(args.model_dir, name))]
    if missing:
        print(f"ERROR: model directory missing files: {missing}")
        return 2
    return run_demo(
        model_dir=args.model_dir,
        side_mm=args.side_mm,
        steps_per_edge=args.steps_per_edge,
        seed=args.seed,
        send_to_robot=args.send_to_robot,
        move_duration=args.move_duration,
    )


if __name__ == "__main__":
    raise SystemExit(main())

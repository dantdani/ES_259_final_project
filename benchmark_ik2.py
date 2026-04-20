"""Benchmark: DNN-init vs Random-init with textbook Newton-Raphson IK."""
import argparse
import json
import os, numpy as np, sys, time

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from ik_v3.infer import IKSolver
from utils import build_ur5e_model, forward_kinematics, skew

ur5e = build_ur5e_model()


def mat_log3(R):
    """SO(3) matrix logarithm → 3-vector (angle-axis)."""
    cos_a = np.clip((np.trace(R) - 1) / 2, -1, 1)
    angle = np.arccos(cos_a)
    if angle < 1e-10:
        return np.zeros(3), 0.0
    if abs(angle - np.pi) < 1e-6:
        # angle ≈ π: find eigenvector of R with eigenvalue 1
        _, vecs = np.linalg.eig(R)
        # find column closest to real with eigenvalue 1
        for i in range(3):
            if abs(np.real(np.linalg.eigvals(R)[i]) - 1.0) < 0.1:
                w = np.real(vecs[:, i])
                w = w / np.linalg.norm(w)
                return w * angle, angle
        w = np.real(vecs[:, 0])
        w = w / np.linalg.norm(w)
        return w * angle, angle
    w_hat = (R - R.T) / (2 * np.sin(angle))
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    return w * angle, angle


def mat_log6(T):
    """SE(3) matrix logarithm → 6-vector twist [v; w]."""
    R = T[:3, :3]
    p = T[:3, 3]
    w_vec, angle = mat_log3(R)

    if angle < 1e-10:
        return np.concatenate([p, np.zeros(3)]), np.linalg.norm(p), 0.0

    w_hat = skew(w_vec / angle)
    G_inv = (np.eye(3) / angle
             - w_hat / 2
             + (1/angle - 1/(2*np.tan(angle/2))) * (w_hat @ w_hat))
    v = G_inv @ p
    return np.concatenate([v, w_vec / angle]) * angle, np.linalg.norm(p), angle


def space_jacobian(theta, model):
    """Space-frame Jacobian using screw axes."""
    w = model['w']; v = model['v']
    theta = np.asarray(theta, dtype=np.float64).flatten()
    J = np.zeros((6, 6), dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        if i == 0:
            J[:3, 0] = v[0]
            J[3:, 0] = w[0]
        else:
            # Adjoint of accumulated transform
            R = T[:3, :3]; p = T[:3, 3]
            S_i = np.concatenate([v[i], w[i]])  # 6-vector [v; w]
            # Ad(T) * S_i
            w_new = R @ w[i]
            v_new = R @ v[i] + np.cross(p, R @ w[i])
            J[:3, i] = v_new
            J[3:, i] = w_new

        # Update accumulated transform
        w_hat = skew(w[i])
        ct = np.cos(theta[i]); st = np.sin(theta[i])
        R_i = np.eye(3) + st * w_hat + (1 - ct) * (w_hat @ w_hat)
        G_i = np.eye(3) * theta[i] + (1 - ct) * w_hat + (theta[i] - st) * (w_hat @ w_hat)
        p_i = G_i @ v[i]
        Ti = np.eye(4); Ti[:3, :3] = R_i; Ti[:3, 3] = p_i
        T = T @ Ti

    return J


def ik_newton(T_target, q_init, model, max_iter=200, eps_v=1.0, eps_w=0.01):
    """Newton-Raphson IK using spatial Jacobian + damped least squares.
    
    eps_v: position tolerance in mm
    eps_w: rotation tolerance in radians (~0.57 deg)
    """
    q = q_init.copy().astype(np.float64)
    
    for it in range(1, max_iter + 1):
        T_cur = forward_kinematics(q, model)
        
        # Spatial error
        dp = T_target[:3, 3] - T_cur[:3, 3]
        R_err = T_target[:3, :3] @ T_cur[:3, :3].T
        cos_a = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        angle = np.arccos(cos_a)
        
        if angle < 1e-10:
            dw = np.zeros(3)
        else:
            s = np.sin(angle)
            dw = (angle / (2 * s)) * np.array([
                R_err[2,1]-R_err[1,2],
                R_err[0,2]-R_err[2,0],
                R_err[1,0]-R_err[0,1]])
        
        pos_err = np.linalg.norm(dp)
        rot_err = angle
        
        if pos_err < eps_v and rot_err < eps_w:
            return q, it, pos_err, np.degrees(rot_err), True
        
        twist = np.concatenate([dp, dw])
        
        J = space_jacobian(q, model)
        
        # Damped least squares (Levenberg-Marquardt)
        lam = 0.01 * np.linalg.norm(twist)
        JtJ = J.T @ J
        dq = np.linalg.solve(JtJ + lam * np.eye(6), J.T @ twist)
        
        # Clamp step size
        max_step = 0.5
        step_norm = np.linalg.norm(dq)
        if step_norm > max_step:
            dq = dq * (max_step / step_norm)
        
        q = q + dq
        q = np.clip(q, -np.pi, np.pi)
    
    T_cur = forward_kinematics(q, model)
    dp = T_target[:3, 3] - T_cur[:3, 3]
    R_err = T_target[:3, :3] @ T_cur[:3, :3].T
    angle = np.arccos(np.clip((np.trace(R_err)-1)/2, -1, 1))
    return q, max_iter, np.linalg.norm(dp), np.degrees(angle), False


def _stats(results, total_samples):
    conv = sum(1 for r in results if r[3])
    iters_conv = [r[0] for r in results if r[3]]
    stats = {
        "converged": int(conv),
        "failed": int(total_samples - conv),
        "convergence_percent": float(100.0 * conv / total_samples),
    }
    if iters_conv:
        pct = np.percentile(iters_conv, [25, 50, 75, 90])
        stats.update({
            "avg_iterations": float(np.mean(iters_conv)),
            "median_iterations": int(np.median(iters_conv)),
            "min_iterations": int(min(iters_conv)),
            "max_iterations": int(max(iters_conv)),
            "p25_iterations": int(pct[0]),
            "p50_iterations": int(pct[1]),
            "p75_iterations": int(pct[2]),
            "p90_iterations": int(pct[3]),
        })
    return stats


def run_benchmark(model_dir, num_samples=1000, seed=42, device="cpu"):
    np.random.seed(seed)
    solver = IKSolver(model_dir, device=device)

    eps_v = 1.0    # 1 mm position
    eps_w = 0.01   # ~0.57 deg rotation
    dnn_results = []
    rand_results = []

    t0 = time.time()
    for i in range(num_samples):
        q_true = np.random.uniform(-np.pi, np.pi, 6).astype(np.float32)
        T_target = forward_kinematics(q_true, ur5e)

        q_seed = q_true + np.random.randn(6).astype(np.float32) * 0.5
        q_seed = np.clip(q_seed, -np.pi, np.pi)
        q_dnn = solver.solve(T_target, q_seed)
        _, iters, pe, re, conv = ik_newton(
            T_target, q_dnn, ur5e, eps_v=eps_v, eps_w=eps_w)
        dnn_results.append((int(iters), float(pe), float(re), bool(conv)))

        best_rand = (200, 999.0, 999.0, False)
        for _ in range(5):
            q_rand = np.random.uniform(-np.pi, np.pi, 6)
            _, iters2, pe2, re2, conv2 = ik_newton(
                T_target, q_rand, ur5e, eps_v=eps_v, eps_w=eps_w)
            if conv2 and (not best_rand[3] or iters2 < best_rand[0]):
                best_rand = (iters2, pe2, re2, conv2)
            elif not best_rand[3] and pe2 < best_rand[1]:
                best_rand = (iters2, pe2, re2, conv2)
        rand_results.append(
            (int(best_rand[0]), float(best_rand[1]), float(best_rand[2]), bool(best_rand[3])))

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_samples}")

    elapsed = time.time() - t0
    dnn_stats = _stats(dnn_results, num_samples)
    rand_stats = _stats(rand_results, num_samples)

    summary = {
        "model_dir": model_dir,
        "num_samples": int(num_samples),
        "seed": int(seed),
        "device": device,
        "tolerance_position_mm": float(eps_v),
        "tolerance_rotation_deg": float(np.degrees(eps_w)),
        "elapsed_seconds": float(elapsed),
        "dnn": dnn_stats,
        "random": rand_stats,
    }
    if dnn_stats.get("avg_iterations") and rand_stats.get("avg_iterations"):
        summary["speedup_x"] = float(
            rand_stats["avg_iterations"] / dnn_stats["avg_iterations"])
        summary["saved_iterations"] = float(
            rand_stats["avg_iterations"] - dnn_stats["avg_iterations"])
    return summary, dnn_results, rand_results


def print_summary(summary, dnn_results, rand_results):
    num_samples = summary["num_samples"]
    dnn_stats = summary["dnn"]
    rand_stats = summary["random"]

    print()
    print("=" * 65)
    print(f"  IK SOLVER BENCHMARK ({num_samples} random targets)")
    print("  Newton-Raphson + Damped Least Squares + Step Clamping")
    print(
        f"  Model dir: {summary['model_dir']} | Device: {summary['device']}")
    print(
        f"  Tolerance: {summary['tolerance_position_mm']} mm position, {summary['tolerance_rotation_deg']:.2f} deg rotation")
    print("  Max iterations: 200 | Random uses best-of-5 restarts")
    print(f"  Time: {summary['elapsed_seconds']:.1f}s")
    print("=" * 65)
    print()
    print("  DNN-Initialized (1 DNN call + Newton polish):")
    print(
        f"    Convergence rate:    {dnn_stats['converged']}/{num_samples} ({dnn_stats['convergence_percent']:.0f}%)")
    if "avg_iterations" in dnn_stats:
        print(f"    Avg iterations:      {dnn_stats['avg_iterations']:.1f}")
        print(f"    Median iterations:   {dnn_stats['median_iterations']}")
        print(
            f"    Min / Max:           {dnn_stats['min_iterations']} / {dnn_stats['max_iterations']}")
        print(
            f"    Percentiles:         P25={dnn_stats['p25_iterations']} P50={dnn_stats['p50_iterations']} P75={dnn_stats['p75_iterations']} P90={dnn_stats['p90_iterations']}")
    print()
    print("  Random-Initialized (best of 5 restarts):")
    print(
        f"    Convergence rate:    {rand_stats['converged']}/{num_samples} ({rand_stats['convergence_percent']:.0f}%)")
    if "avg_iterations" in rand_stats:
        print(f"    Avg iterations:      {rand_stats['avg_iterations']:.1f}")
        print(f"    Median iterations:   {rand_stats['median_iterations']}")
        print(
            f"    Min / Max:           {rand_stats['min_iterations']} / {rand_stats['max_iterations']}")
        print(
            f"    Percentiles:         P25={rand_stats['p25_iterations']} P50={rand_stats['p50_iterations']} P75={rand_stats['p75_iterations']} P90={rand_stats['p90_iterations']}")
    print()
    if "speedup_x" in summary:
        print(
            f"  SPEEDUP: {summary['speedup_x']:.1f}x fewer iters (DNN vs Random)")
        print(
            f"  SAVED:   ~{summary['saved_iterations']:.0f} iterations per solve")
    elif "avg_iterations" in dnn_stats:
        print(
            f"  Random NEVER converged. DNN avg = {dnn_stats['avg_iterations']:.1f} iters")
    print(
        f"  RELIABILITY: DNN {dnn_stats['converged']} vs Random {rand_stats['converged']} converged out of {num_samples}")
    print()
    print("-" * 65)
    print("  First 30 individual results:")
    hdr = (
        f"  {'#':>3}  {'DNN it':>7} {'DNN':>4} {'pos_mm':>7}  {'Rnd it':>7} {'Rnd':>4} {'pos_mm':>7}")
    print(hdr)
    for i in range(min(30, num_samples)):
        d = dnn_results[i]
        r = rand_results[i]
        dc = "YES" if d[3] else "no"
        rc = "YES" if r[3] else "no"
        print(
            f"  {i+1:>3}  {d[0]:>7}  {dc:>4} {d[1]:>7.2f}  {r[0]:>7}  {rc:>4} {r[1]:>7.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DNN-initialized IK against random restarts")
    parser.add_argument("--model_dir", type=str, default="ik_v3/results")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary, dnn_results, rand_results = run_benchmark(
        model_dir=args.model_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
    )
    print_summary(summary, dnn_results, rand_results)

    if args.output_json:
        payload = {
            "summary": summary,
            "first_30_examples": [
                {
                    "index": i + 1,
                    "dnn": {
                        "iterations": dnn_results[i][0],
                        "position_error_mm": dnn_results[i][1],
                        "rotation_error_deg": dnn_results[i][2],
                        "converged": dnn_results[i][3],
                    },
                    "random": {
                        "iterations": rand_results[i][0],
                        "position_error_mm": rand_results[i][1],
                        "rotation_error_deg": rand_results[i][2],
                        "converged": rand_results[i][3],
                    },
                }
                for i in range(min(30, summary["num_samples"]))
            ],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

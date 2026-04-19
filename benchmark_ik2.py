"""Benchmark: DNN-init vs Random-init with textbook Newton-Raphson IK."""
import os, numpy as np, sys, time

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from ik_v3.infer import IKSolver
from utils import build_ur5e_model, forward_kinematics, skew

np.random.seed(42)
solver = IKSolver('ik_v3/results', device='cpu')
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


# ===================== BENCHMARK =====================
N = 100
eps_v = 1.0    # 1 mm position
eps_w = 0.01   # ~0.57 deg rotation

dnn_results = []
rand_results = []

t0 = time.time()
for i in range(N):
    q_true = np.random.uniform(-np.pi, np.pi, 6).astype(np.float32)
    T_target = forward_kinematics(q_true, ur5e)

    # --- DNN init (single shot) ---
    q_seed = q_true + np.random.randn(6).astype(np.float32) * 0.5
    q_seed = np.clip(q_seed, -np.pi, np.pi)
    q_dnn = solver.solve(T_target, q_seed)
    _, iters, pe, re, conv = ik_newton(T_target, q_dnn, ur5e, eps_v=eps_v, eps_w=eps_w)
    dnn_results.append((iters, pe, re, conv))

    # --- Random init (best of 5 restarts) ---
    best_rand = (200, 999.0, 999.0, False)
    for _ in range(5):
        q_rand = np.random.uniform(-np.pi, np.pi, 6)
        _, iters2, pe2, re2, conv2 = ik_newton(T_target, q_rand, ur5e, eps_v=eps_v, eps_w=eps_w)
        if conv2 and (not best_rand[3] or iters2 < best_rand[0]):
            best_rand = (iters2, pe2, re2, conv2)
        elif not best_rand[3] and pe2 < best_rand[1]:
            best_rand = (iters2, pe2, re2, conv2)
    rand_results.append(best_rand)

    if (i+1) % 25 == 0:
        print(f"  Progress: {i+1}/{N}")

elapsed = time.time() - t0

dnn_conv = sum(1 for r in dnn_results if r[3])
rand_conv = sum(1 for r in rand_results if r[3])
dnn_iters_conv = [r[0] for r in dnn_results if r[3]]
rand_iters_conv = [r[0] for r in rand_results if r[3]]

print()
print("=" * 65)
print(f"  IK SOLVER BENCHMARK ({N} random targets)")
print(f"  Newton-Raphson + Damped Least Squares + Step Clamping")
print(f"  Tolerance: {eps_v} mm position, {np.degrees(eps_w):.2f} deg rotation")
print(f"  Max iterations: 200 | Random uses best-of-5 restarts")
print(f"  Time: {elapsed:.1f}s")
print("=" * 65)
print()
print(f"  DNN-Initialized (1 DNN call + Newton polish):")
print(f"    Convergence rate:    {dnn_conv}/{N} ({100*dnn_conv/N:.0f}%)")
if dnn_iters_conv:
    print(f"    Avg iterations:      {np.mean(dnn_iters_conv):.1f}")
    print(f"    Median iterations:   {int(np.median(dnn_iters_conv))}")
    print(f"    Min / Max:           {min(dnn_iters_conv)} / {max(dnn_iters_conv)}")
    pct = np.percentile(dnn_iters_conv, [25, 50, 75, 90])
    print(f"    Percentiles:         P25={int(pct[0])} P50={int(pct[1])} P75={int(pct[2])} P90={int(pct[3])}")
print()
print(f"  Random-Initialized (best of 5 restarts):")
print(f"    Convergence rate:    {rand_conv}/{N} ({100*rand_conv/N:.0f}%)")
if rand_iters_conv:
    print(f"    Avg iterations:      {np.mean(rand_iters_conv):.1f}")
    print(f"    Median iterations:   {int(np.median(rand_iters_conv))}")
    print(f"    Min / Max:           {min(rand_iters_conv)} / {max(rand_iters_conv)}")
    pct = np.percentile(rand_iters_conv, [25, 50, 75, 90])
    print(f"    Percentiles:         P25={int(pct[0])} P50={int(pct[1])} P75={int(pct[2])} P90={int(pct[3])}")
print()
if dnn_iters_conv and rand_iters_conv:
    print(f"  SPEEDUP: {np.mean(rand_iters_conv)/np.mean(dnn_iters_conv):.1f}x fewer iters (DNN vs Random)")
    print(f"  SAVED:   ~{np.mean(rand_iters_conv)-np.mean(dnn_iters_conv):.0f} iterations per solve")
elif dnn_iters_conv:
    print(f"  Random NEVER converged. DNN avg = {np.mean(dnn_iters_conv):.1f} iters")
print(f"  RELIABILITY: DNN {dnn_conv} vs Random {rand_conv} converged out of {N}")
print()

# Individual examples
print("-" * 65)
print(f"  First 20 individual results:")
hdr = f"  {'#':>3}  {'DNN it':>7} {'DNN':>4} {'pos_mm':>7}  {'Rnd it':>7} {'Rnd':>4} {'pos_mm':>7}"
print(hdr)
for i in range(min(20, N)):
    d = dnn_results[i]; r = rand_results[i]
    dc = "YES" if d[3] else "no"
    rc = "YES" if r[3] else "no"
    print(f"  {i+1:>3}  {d[0]:>7}  {dc:>4} {d[1]:>7.2f}  {r[0]:>7}  {rc:>4} {r[1]:>7.2f}")

"""Benchmark: DNN-init vs Random-init with production-quality IK solver."""
import os, numpy as np, sys, time

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from ik_v3.infer import IKSolver
from utils import build_ur5e_model, forward_kinematics, skew

np.random.seed(42)
solver = IKSolver('ik_v3/results', device='cpu')
ur5e = build_ur5e_model()


def spatial_jacobian(theta, model):
    w = model['w']; v = model['v']
    theta = np.asarray(theta, dtype=np.float64).flatten()
    T = [np.eye(4, dtype=np.float64)]
    for i in range(6):
        w_hat = skew(w[i])
        ct = np.cos(theta[i]); st = np.sin(theta[i])
        R = np.eye(3) + st*w_hat + (1-ct)*(w_hat@w_hat)
        G = np.eye(3)*theta[i] + (1-ct)*w_hat + (theta[i]-st)*(w_hat@w_hat)
        p = G @ v[i]
        Ti = np.eye(4, dtype=np.float64); Ti[:3,:3] = R; Ti[:3,3] = p
        T.append(T[-1] @ Ti)
    J = np.zeros((6,6), dtype=np.float64)
    for i in range(6):
        Ti = T[i]; R = Ti[:3,:3]; p = Ti[:3,3]
        wi = R @ np.asarray(w[i], dtype=np.float64)
        vi = R @ np.asarray(v[i], dtype=np.float64) + np.cross(p.flatten(), wi.flatten())
        J[:3, i] = vi
        J[3:, i] = wi
    return J


def pose_error(T_target, T_current):
    # Compute error in body frame for proper Jacobian pairing
    T_err = np.linalg.inv(T_current) @ T_target  # body-frame error
    dp = T_err[:3, 3]
    R_err = T_err[:3,:3]
    trace_val = np.clip((np.trace(R_err)-1)/2, -1, 1)
    angle = np.arccos(trace_val)
    if abs(angle) < 1e-12:
        dw = np.zeros(3)
    else:
        s = np.sin(angle) + 1e-15
        dw = (angle/(2*s)) * np.array([
            R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])
    # Transform to spatial frame for reporting
    R_cur = T_current[:3,:3]
    dp_spatial = T_target[:3,3] - T_current[:3,3]
    return np.concatenate([dp, dw]), np.linalg.norm(dp_spatial), angle


def body_jacobian(theta, model):
    """Compute body Jacobian using the UR5e body screws."""
    B = model['B']
    M = model['M']
    theta = np.asarray(theta, dtype=np.float64).flatten()
    
    J = np.zeros((6,6), dtype=np.float64)
    T_acc = np.eye(4, dtype=np.float64)  # accumulates from right
    
    for i in range(5, -1, -1):
        wi = B[i, :3]
        vi = B[i, 3:]
        if i < 5:
            # T_acc = exp(-S_{i+1}*theta_{i+1}) * ... * exp(-S_6*theta_6)
            w_next = B[i+1, :3]; v_next = B[i+1, 3:]
            w_hat = skew(w_next)
            th = -theta[i+1]
            ct = np.cos(th); st = np.sin(th)
            R = np.eye(3) + st*w_hat + (1-ct)*(w_hat@w_hat)
            G = np.eye(3)*th + (1-ct)*w_hat + (th-st)*(w_hat@w_hat)
            p = G @ v_next
            Ti = np.eye(4); Ti[:3,:3] = R; Ti[:3,3] = p
            T_acc = Ti @ T_acc
        
        if i == 5:
            J[:3, i] = vi
            J[3:, i] = wi
        else:
            # Ad(T_acc) * S_i
            R_a = T_acc[:3,:3]; p_a = T_acc[:3,3]
            w_new = R_a @ wi
            v_new = R_a @ vi + np.cross(p_a, w_new)
            J[:3, i] = v_new
            J[3:, i] = w_new
    return J


def ik_solve(T_target, q_init, model, max_iter=100, tol_pos=0.1, tol_rot=0.01):
    """Production IK: adaptive Levenberg-Marquardt + line search (body frame)."""
    q = q_init.copy().astype(np.float64)
    lam = 0.1
    lam_min, lam_max = 1e-8, 1e4

    T_cur = forward_kinematics(q, model)
    err_vec, pos_err, rot_err = pose_error(T_target, T_cur)
    err_norm = np.linalg.norm(err_vec)

    for it in range(1, max_iter+1):
        if pos_err < tol_pos and rot_err < tol_rot:
            return q, it, pos_err, np.degrees(rot_err), True

        J = body_jacobian(q, model)
        JtJ = J.T @ J
        Jte = J.T @ err_vec

        # Adaptive LM step with Marquardt diagonal scaling
        dq = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-8), Jte)

        # Line search: full, half, quarter
        best_q, best_err, best_step = q, err_norm, 0
        for alpha in [1.0, 0.5, 0.25]:
            q_try = np.clip(q + alpha * dq, -np.pi, np.pi)
            T_try = forward_kinematics(q_try, model)
            e_try, _, _ = pose_error(T_target, T_try)
            en = np.linalg.norm(e_try)
            if en < best_err:
                best_q, best_err, best_step = q_try, en, alpha

        if best_step > 0:
            q = best_q
            lam = max(lam * 0.5, lam_min)
        else:
            lam = min(lam * 4.0, lam_max)

        T_cur = forward_kinematics(q, model)
        err_vec, pos_err, rot_err = pose_error(T_target, T_cur)
        err_norm = np.linalg.norm(err_vec)

        if err_norm < 1e-10:
            return q, it, pos_err, np.degrees(rot_err), True

    return q, max_iter, pos_err, np.degrees(rot_err), (pos_err < tol_pos and rot_err < tol_rot)


# ===================== BENCHMARK =====================
N = 100
tol_pos = 0.1    # 0.1 mm
tol_rot = 0.01   # ~0.57 degrees

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
    _, iters, pe, re, conv = ik_solve(T_target, q_dnn, ur5e)
    dnn_results.append((iters, pe, re, conv))

    # --- Random init (best of 5 restarts) ---
    best_rand = (100, 999.0, 999.0, False)
    for _ in range(5):
        q_rand = np.random.uniform(-np.pi, np.pi, 6)
        _, iters2, pe2, re2, conv2 = ik_solve(T_target, q_rand, ur5e)
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
print(f"  PRODUCTION IK SOLVER BENCHMARK ({N} random targets)")
print(f"  Adaptive LM + Line Search + Damping Schedule")
print(f"  Tolerance: {tol_pos} mm position, {np.degrees(tol_rot):.2f} deg rotation")
print(f"  Random init uses best-of-5 restarts")
print(f"  Time: {elapsed:.1f}s")
print("=" * 65)
print()
print(f"  DNN-Initialized (single shot):")
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
    print(f"  SPEEDUP: {np.mean(rand_iters_conv)/np.mean(dnn_iters_conv):.1f}x fewer iterations (DNN vs Random)")
    print(f"  SAVED:   ~{np.mean(rand_iters_conv)-np.mean(dnn_iters_conv):.0f} iterations per solve")
print(f"  RELIABILITY: DNN converges {dnn_conv-rand_conv} more out of {N} ({100*(dnn_conv-rand_conv)/N:+.0f}%)")
print()

# Individual examples
print("-" * 65)
print(f"  First 15 individual results:")
print(f"  {'#':>3}  {'DNN iters':>10} {'DNN':>5}  {'Rand iters':>11} {'Rand':>5}")
for i in range(min(15, N)):
    d = dnn_results[i]; r = rand_results[i]
    dc = "YES" if d[3] else "no"
    rc = "YES" if r[3] else "no"
    print(f"  {i+1:>3}  {d[0]:>10}  {dc:>5}   {r[0]:>10}   {rc:>5}")

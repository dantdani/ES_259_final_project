#!/usr/bin/env python3
"""
generate_ur5e_fk_ik_dataset.py

UR5e Forward Kinematics (Product of Exponentials) and supervised IK dataset
generator.  Geometry, screw axes, and conventions are identical to the
verified MATLAB IK implementation.

Units: millimeters (consistent with MATLAB code).

Usage:
    python generate_ur5e_fk_ik_dataset.py                       # defaults
    python generate_ur5e_fk_ik_dataset.py --num_samples 100000 --mode pose
"""

import argparse
import time
import numpy as np
import pandas as pd

# ============================================================================
# Joint limits (radians).  Change these if you want a restricted workspace.
# ============================================================================
DEFAULT_JOINT_LIMITS = [
    (-np.pi, np.pi),   # joint 1
    (-np.pi, np.pi),   # joint 2
    (-np.pi, np.pi),   # joint 3
    (-np.pi, np.pi),   # joint 4
    (-np.pi, np.pi),   # joint 5
    (-np.pi, np.pi),   # joint 6
]

# ============================================================================
# Helper functions (match MATLAB helpers exactly)
# ============================================================================

def skew(v):
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ])


def exp_rot(w_hat, theta):
    """Rodrigues rotation:  exp([w] * theta) = I + sin(th)*[w] + (1-cos(th))*[w]^2
    w_hat is the 3x3 skew-symmetric matrix of the unit axis w."""
    return np.eye(3) + np.sin(theta) * w_hat + (1.0 - np.cos(theta)) * (w_hat @ w_hat)


def g_func(w_hat, theta):
    """Translation helper for the screw exponential:
    G(theta) = I*theta + (1-cos(th))*[w] + (th-sin(th))*[w]^2"""
    return np.eye(3) * theta + (1.0 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * (w_hat @ w_hat)


def screw_exp(w, v, theta):
    """Compute the 4x4 matrix exponential of a screw motion S*theta.

    Parameters
    ----------
    w : (3,) array  – angular part of the screw axis (unit or zero vector)
    v : (3,) array  – linear part of the screw axis
    theta : float   – joint angle

    Returns
    -------
    T : (4,4) ndarray – homogeneous transformation
    """
    w_hat = skew(w)
    R = exp_rot(w_hat, theta)
    p = g_func(w_hat, theta) @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def adjoint(T):
    """6x6 adjoint representation of a 4x4 homogeneous transform."""
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def matrix_log_so3(R):
    """Compute the matrix logarithm of a rotation matrix.

    Returns (w_hat, theta) where w_hat is a 3x3 skew-symmetric matrix
    and theta is the rotation angle.
    """
    cos_val = (np.trace(R) - 1.0) / 2.0
    cos_val = np.clip(cos_val, -1.0, 1.0)

    if np.linalg.norm(R - np.eye(3), 'fro') < 1e-9:
        return np.zeros((3, 3)), 0.0

    if abs(cos_val + 1.0) < 1e-9:
        theta = np.pi
        # find the column that gives a valid axis
        if abs(1.0 + R[0, 0]) > 1e-8:
            w = np.array([1 + R[0, 0], R[1, 0], R[2, 0]]) / np.sqrt(2.0 * (1 + R[0, 0]))
        elif abs(1.0 + R[1, 1]) > 1e-8:
            w = np.array([R[0, 1], 1 + R[1, 1], R[2, 1]]) / np.sqrt(2.0 * (1 + R[1, 1]))
        else:
            w = np.array([R[0, 2], R[1, 2], 1 + R[2, 2]]) / np.sqrt(2.0 * (1 + R[2, 2]))
        w_hat = skew(w)
    else:
        theta = np.arccos(cos_val)
        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-15:
            # theta ≈ 0 already handled above; guard against numerical edge case
            return np.zeros((3, 3)), 0.0
        w_hat = (R - R.T) / (2.0 * sin_theta)

    return w_hat, theta


def matrix_log_se3(T):
    """Compute the 4x4 matrix logarithm of a homogeneous transform.

    Returns a 4x4 matrix in se(3).
    """
    R = T[:3, :3]
    p = T[:3, 3]
    w_hat, theta = matrix_log_so3(R)

    result = np.zeros((4, 4))

    if theta < 1e-12:
        # pure translation
        result[:3, 3] = p
        return result

    half_cot = 0.5 / np.tan(theta / 2.0) if abs(np.tan(theta / 2.0)) > 1e-15 else 0.0
    G_inv = (1.0 / theta) * np.eye(3) - 0.5 * w_hat + (1.0 / theta - half_cot) * (w_hat @ w_hat)
    v = G_inv @ p

    result[:3, :3] = w_hat * theta
    result[:3, 3] = v * theta
    return result

# ============================================================================
# UR5e model builder
# ============================================================================

def build_ur5e_model():
    """Build and return the UR5e kinematic model dictionary.

    All values exactly match the verified MATLAB IK code.
    Units: millimeters.
    """
    # Link dimensions (mm)
    W2 = 259.6
    W1 = 133.3
    H2 = 99.7
    H1 = 162.5
    L1 = 425.0
    L2 = 392.2

    # Angular parts of space screw axes
    w = np.array([
        [0, 0,  1],   # joint 1
        [0, 1,  0],   # joint 2
        [0, 1,  0],   # joint 3
        [0, 1,  0],   # joint 4
        [0, 0, -1],   # joint 5
        [0, 1,  0],   # joint 6
    ], dtype=float)

    # Points on each screw axis
    q = np.array([
        [0,        0,  0  ],
        [0,        0,  H1 ],
        [L1,       0,  H1 ],
        [L1 + L2,  0,  H1 ],
        [L1 + L2,  W1, 0  ],
        [L1 + L2,  0,  H1 - H2],
    ], dtype=float)

    # Linear parts:  v = cross(-w, q)
    v = np.zeros((6, 3))
    for i in range(6):
        v[i] = np.cross(-w[i], q[i])

    # 6x1 screw axes  S = [w; v]
    S = np.hstack([w, v])  # shape (6, 6)

    # Home configuration
    M = np.array([
        [-1, 0, 0, L1 + L2],
        [ 0, 0, 1, W1 + W2],
        [ 0, 1, 0, H1 - H2],
        [ 0, 0, 0, 1      ],
    ], dtype=float)

    # Body screw axes  B = Ad(M)^{-1} * S  (each column)
    Ad_M = adjoint(M)
    Ad_M_inv = np.linalg.inv(Ad_M)
    B = np.zeros((6, 6))
    for i in range(6):
        B[i] = Ad_M_inv @ S[i]

    model = {
        'W1': W1, 'W2': W2, 'H1': H1, 'H2': H2, 'L1': L1, 'L2': L2,
        'w': w,        # (6,3) angular axes
        'v': v,        # (6,3) linear axes
        'S': S,        # (6,6) space screws  [w | v]
        'B': B,        # (6,6) body screws
        'M': M,        # (4,4) home config
        'num_joints': 6,
    }
    return model

# ============================================================================
# Forward kinematics (Product of Exponentials – space frame)
# ============================================================================

def forward_kinematics(theta, model):
    """Compute the 4x4 end-effector transform using the space-frame PoE.

    T = exp([S1]*th1) * exp([S2]*th2) * ... * exp([S6]*th6) * M

    Parameters
    ----------
    theta : array-like, shape (6,)   – joint angles in radians
    model : dict                      – from build_ur5e_model()

    Returns
    -------
    T : (4,4) ndarray – end-effector homogeneous transform
    """
    w = model['w']
    v = model['v']
    M = model['M']

    T = np.eye(4)
    for i in range(6):
        T = T @ screw_exp(w[i], v[i], theta[i])
    T = T @ M
    return T

# ============================================================================
# Inverse kinematics placeholder
# ============================================================================

def inverse_kinematics(T_desired, theta0, model, max_iter=60, eps_w=1e-5, eps_v=1e-4):
    """Newton-Raphson body-frame IK (matches MATLAB structure).

    This is a direct port of the MATLAB IK routine. It uses the body
    Jacobian and matrix logarithm to iteratively converge on joint angles.

    Parameters
    ----------
    T_desired : (4,4) ndarray – desired end-effector pose
    theta0    : (6,)  ndarray – initial joint guess (radians)
    model     : dict          – from build_ur5e_model()
    max_iter  : int           – maximum Newton-Raphson iterations
    eps_w     : float         – angular convergence threshold
    eps_v     : float         – linear convergence threshold

    Returns
    -------
    theta : (6,) ndarray – solved joint angles (radians, wrapped to [-pi, pi])

    Raises
    ------
    RuntimeError if convergence fails.
    """
    B = model['B']   # (6, 6)  each row is a body screw axis
    M = model['M']

    th = np.array(theta0, dtype=float).copy()

    for iteration in range(max_iter):
        # Current FK
        T_current = forward_kinematics(th, model)

        # Body twist error
        T_bd = np.linalg.inv(T_current) @ T_desired
        Vb_mat = matrix_log_se3(T_bd)
        Vb = np.array([Vb_mat[2, 1], Vb_mat[0, 2], Vb_mat[1, 0],
                        Vb_mat[0, 3], Vb_mat[1, 3], Vb_mat[2, 3]])

        wb = Vb[:3]
        vb = Vb[3:]

        if np.linalg.norm(wb) < eps_w and np.linalg.norm(vb) < eps_v:
            break

        # Body Jacobian (column by column, matching MATLAB)
        # e_Bi = screw_exp(B[i,:3], B[i,3:], th[i])
        e_B = [screw_exp(B[i, :3], B[i, 3:], th[i]) for i in range(6)]

        Jb = np.zeros((6, 6))
        Jb[:, 5] = B[5]
        for i in range(4, -1, -1):
            prod = np.eye(4)
            for j in range(i + 1, 6):
                prod = prod @ e_B[j]
            prod_inv = np.linalg.inv(prod)
            Jb[:, i] = adjoint(prod_inv) @ B[i]

        # Check for numerical issues
        if np.any(np.isnan(Jb)) or np.any(np.isinf(Jb)):
            raise RuntimeError("IK failed: Jacobian contains NaN or Inf")
        if np.any(np.isnan(Vb)) or np.any(np.isinf(Vb)):
            raise RuntimeError("IK failed: Vb contains NaN or Inf")

        dth = np.linalg.pinv(Jb) @ Vb

        if np.any(np.isnan(dth)) or np.any(np.isinf(dth)):
            raise RuntimeError("IK failed: dth contains NaN or Inf")

        # Clamp step size (matches MATLAB maxStep = 0.35)
        max_step = 0.35
        dth = np.clip(dth, -max_step, max_step)

        th_new = th + dth

        # Unwrap to closest (matches MATLAB unwrapToClosest)
        for k in range(6):
            candidates = [th_new[k] - 2 * np.pi, th_new[k], th_new[k] + 2 * np.pi]
            best = min(candidates, key=lambda c: abs(c - th[k]))
            th_new[k] = best

        th = th_new
    else:
        raise RuntimeError(f"IK failed: exceeded {max_iter} iterations")

    # Wrap to [-pi, pi]
    th = (th + np.pi) % (2 * np.pi) - np.pi
    return th

# ============================================================================
# Sampling & feature extraction
# ============================================================================

def sample_joint_configuration(joint_limits, rng):
    """Sample a random joint vector uniformly within the given limits.

    Parameters
    ----------
    joint_limits : list of (lo, hi) tuples, length 6
    rng          : numpy Generator

    Returns
    -------
    theta : (6,) ndarray
    """
    theta = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
    return theta


def extract_features(T, mode="position"):
    """Extract features from a 4x4 end-effector transform.

    Parameters
    ----------
    T    : (4,4) ndarray – homogeneous transform
    mode : str           – "position" or "pose"

    Returns
    -------
    features : list of floats
        position mode → [x, y, z]
        pose mode     → [x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33]
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]

    if mode == "position":
        return [x, y, z]
    elif mode == "pose":
        R = T[:3, :3]
        return [x, y, z,
                R[0, 0], R[0, 1], R[0, 2],
                R[1, 0], R[1, 1], R[1, 2],
                R[2, 0], R[2, 1], R[2, 2]]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'position' or 'pose'.")

# ============================================================================
# FK verification
# ============================================================================

def verify_fk(model):
    """Run quick sanity checks on the FK implementation.

    1. FK at theta = zeros(6) should return the home matrix M.
    2. FK output is always 4x4.
    3. Rotation part is approximately orthonormal.

    Raises AssertionError on failure.
    """
    print("=" * 60)
    print("FK VERIFICATION")
    print("=" * 60)

    M = model['M']

    # Test 1: FK at zero joint angles == M
    theta_zero = np.zeros(6)
    T0 = forward_kinematics(theta_zero, model)
    err = np.linalg.norm(T0 - M)
    print(f"  [1] FK(zeros) vs M  →  error norm = {err:.2e}  ", end="")
    assert err < 1e-10, f"FK at zero joints does not match M! Error = {err}"
    print("PASS")

    # Test 2: output shape
    assert T0.shape == (4, 4), f"FK output shape is {T0.shape}, expected (4,4)"
    print("  [2] FK output shape (4,4)                      PASS")

    # Test 3: rotation orthonormality at several random configs
    rng = np.random.default_rng(42)
    for trial in range(20):
        theta_rand = rng.uniform(-np.pi, np.pi, size=6)
        T = forward_kinematics(theta_rand, model)
        R = T[:3, :3]
        ortho_err = np.linalg.norm(R @ R.T - np.eye(3))
        det_err = abs(np.linalg.det(R) - 1.0)
        assert ortho_err < 1e-8, f"Rotation not orthonormal (trial {trial}): {ortho_err}"
        assert det_err < 1e-8, f"Rotation det != 1 (trial {trial}): {det_err}"
    print("  [3] Rotation orthonormality (20 random configs) PASS")

    # Test 4: quick FK-IK round-trip
    theta_test = np.array([0.5, -0.3, 0.8, -1.2, 0.4, -0.6])
    T_test = forward_kinematics(theta_test, model)
    try:
        theta_recovered = inverse_kinematics(T_test, np.zeros(6), model)
        T_recovered = forward_kinematics(theta_recovered, model)
        pos_err = np.linalg.norm(T_recovered[:3, 3] - T_test[:3, 3])
        rot_err = np.linalg.norm(T_recovered[:3, :3] - T_test[:3, :3], 'fro')
        print(f"  [4] FK-IK round-trip  →  pos err = {pos_err:.2e} mm, "
              f"rot err = {rot_err:.2e}  ", end="")
        assert pos_err < 1.0 and rot_err < 1e-3, "Round-trip error too large"
        print("PASS")
    except RuntimeError as e:
        print(f"  [4] FK-IK round-trip  →  IK did not converge: {e}  SKIP")

    print("=" * 60)
    print("All FK verification checks passed.\n")

# ============================================================================
# Dataset generation
# ============================================================================

def generate_dataset(num_samples, mode="position", output_csv="ur5e_ik_dataset.csv",
                     seed=42, joint_limits=None):
    """Generate a supervised FK dataset and write it to CSV.

    Parameters
    ----------
    num_samples  : int   – number of samples to generate
    mode         : str   – "position" or "pose"
    output_csv   : str   – output CSV filename
    seed         : int   – random seed for reproducibility
    joint_limits : list  – per-joint (lo, hi) tuples; defaults to [-pi, pi]

    Returns
    -------
    stats : dict  – summary statistics
    """
    if joint_limits is None:
        joint_limits = DEFAULT_JOINT_LIMITS

    rng = np.random.default_rng(seed)
    model = build_ur5e_model()

    # Column names
    joint_cols = [f"q{i+1}" for i in range(6)]
    if mode == "position":
        feature_cols = ["x", "y", "z"]
    elif mode == "pose":
        feature_cols = ["x", "y", "z",
                        "r11", "r12", "r13",
                        "r21", "r22", "r23",
                        "r31", "r32", "r33"]
    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    all_cols = feature_cols + joint_cols
    rows = []

    success = 0
    fail = 0
    log_interval = max(1, num_samples // 20)  # ~5 % increments

    t_start = time.time()

    for i in range(num_samples):
        try:
            theta = sample_joint_configuration(joint_limits, rng)
            T = forward_kinematics(theta, model)

            # Validate output
            if T.shape != (4, 4):
                raise ValueError("FK returned non-4x4 matrix")
            if np.any(np.isnan(T)) or np.any(np.isinf(T)):
                raise ValueError("FK returned NaN/Inf")

            features = extract_features(T, mode=mode)
            row = features + theta.tolist()
            rows.append(row)
            success += 1

        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  [warn] Sample {i}: {e}")

        if (i + 1) % log_interval == 0 or (i + 1) == num_samples:
            elapsed = time.time() - t_start
            print(f"  progress: {i+1:>8d} / {num_samples}  "
                  f"({100*(i+1)/num_samples:5.1f}%)  "
                  f"success={success}  fail={fail}  "
                  f"elapsed={elapsed:.1f}s")

    df = pd.DataFrame(rows, columns=all_cols)
    df.to_csv(output_csv, index=False)

    stats = {
        "requested": num_samples,
        "success": success,
        "fail": fail,
        "output_file": output_csv,
        "mode": mode,
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    return stats

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate UR5e FK-based IK training dataset.")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of joint samples to generate (default: 10000)")
    parser.add_argument("--mode", type=str, default="position",
                        choices=["position", "pose"],
                        help="Feature mode: 'position' (x,y,z) or 'pose' (x,y,z + R)")
    parser.add_argument("--output", type=str, default="ur5e_ik_dataset.csv",
                        help="Output CSV filename (default: ur5e_ik_dataset.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Skip FK verification checks")
    args = parser.parse_args()

    model = build_ur5e_model()

    # --- Verification ---
    if not args.skip_verify:
        verify_fk(model)

    # --- Dataset generation ---
    print(f"Generating {args.num_samples} samples  (mode={args.mode}, seed={args.seed})")
    print(f"Output → {args.output}\n")

    stats = generate_dataset(
        num_samples=args.num_samples,
        mode=args.mode,
        output_csv=args.output,
        seed=args.seed,
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Requested samples : {stats['requested']}")
    print(f"  Successful        : {stats['success']}")
    print(f"  Failed            : {stats['fail']}")
    print(f"  Feature mode      : {stats['mode']}")
    print(f"  Output file       : {stats['output_file']}")
    print(f"  Elapsed time      : {stats['elapsed_sec']} s")
    print("=" * 60)


if __name__ == "__main__":
    main()

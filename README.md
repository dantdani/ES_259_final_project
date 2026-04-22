# DNN-Initialized Inverse Kinematics for the UR5e

**Software Usage Guide**

Dandi Desta — ES 259 Final Project — April 22, 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Quick Start (run the pretrained 10M model)](#4-quick-start-run-the-pretrained-10m-model)
5. [Generating a Dataset](#5-generating-a-dataset)
6. [Training](#6-training)
7. [Evaluation and Benchmarking](#7-evaluation-and-benchmarking)
8. [Inference Pipeline (detailed)](#8-inference-pipeline-detailed)
9. [Reproducing the Full Study](#9-reproducing-the-full-study)
10. [Troubleshooting](#10-troubleshooting)
11. [Citation](#11-citation)

---

## 1. Overview

This repository implements a hybrid inverse-kinematics (IK) solver for the
Universal Robots UR5e: a seed-conditioned deep neural network produces a
joint-angle initialization, and a damped Newton–Raphson routine polishes that
guess to sub-millimetre precision.

The software supports the full research pipeline:

1. Synthetic dataset generation via forward kinematics (FK).
2. Training the production v3 ResMLP (seed-conditioned) and the legacy v1
   pose-only MLP / ResMLP variants.
3. Benchmarking DNN-initialized IK vs. random-initialized IK.
4. Running inference on new end-effector targets.
5. Reproducing the data-scaling sweep (10k → 10M) and the differentiable-FK
   composite-loss ablation.

**Headline result:** ~77% DNN-init convergence vs. ~44% random on 1,000 random
UR5e poses, with a ~5× iteration speedup. See
[`methodology.txt`](methodology.txt) for the full writeup.

---

## 2. Repository Layout

```
ES_259_final_project/
├── ik_v3/                      # PRODUCTION code (seed-conditioned ResMLP)
│   ├── model.py                #   SeedConditionedIKModel, ResidualBlock
│   ├── train.py                #   training loop (ik_v3.train CLI)
│   ├── infer.py                #   IKSolver class (DNN + Newton polish)
│   ├── generate_dataset.py     #   vectorised PoE FK data generator
│   ├── representations.py      #   6D rotation, sin/cos encode/decode
│   ├── results/                #   10M model_best.pt + scalers
│   └── ur5e_seed_*.csv         #   datasets (10k ... 10M)
│
├── ik_v2/                      # Exploratory: MoE / region experts / refinement
│
├── model.py                    # v1 legacy model (pose-only MLP/ResMLP)
├── train_ik_model.py           # v1 training script
├── generate_pose_dataset.py    # v1 dataset generator
├── benchmark_ik.py             # v1-style benchmark
├── benchmark_ik2.py            # v3-style benchmark (recommended)
├── evaluate_model.py           # per-joint test metrics + plots
├── infer_ik.py                 # v1 inference helper
│
├── train_ik_model_fk_loss.py   # experimental composite (FK) loss trainer
├── model_fk_loss.py            # FK-loss flat MLP model
├── utils_torch_fk.py           # differentiable torch PoE FK
├── run_fk_study.py             # lambda_pos sweep driver
├── run_study.py / run_study_live.py   # v1 architecture sweep drivers
│
├── results/                    # v1 architecture-sweep artifacts
├── results_v1/                 # v1 10M-trained artifacts
├── pose_results/               # v1 baseline artifacts
├── fk_loss_experiments/        # differentiable-FK ablation artifacts
│
├── methodology.txt             # full research writeup
└── accuracy_vs_dataset_size.png
```

---

## 3. Installation

### 3.1 Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux (CPU-only supported; GPU optional)
- ~4 GB free disk (full 10M-sample CSV is ~1.43 GB)

### 3.2 Clone and set up a virtual environment

```bash
git clone https://github.com/dantdani/ES_259_final_project.git
cd ES_259_final_project

# Create + activate a venv
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

### 3.3 Install Python dependencies

```bash
pip install --upgrade pip
pip install torch numpy pandas scikit-learn matplotlib tqdm joblib
```

GPU users may instead install the CUDA build of PyTorch from
[pytorch.org](https://pytorch.org/get-started/locally/). All scripts default
to CPU.

---

## 4. Quick Start (run the pretrained 10M model)

The 10M-trained production model ships inside `ik_v3/results/`
(`model_best.pt` + `pose_scaler.pkl` + `seed_scaler.pkl`). You can use it
immediately without retraining.

### 4.1 Benchmark on 1,000 random UR5e poses

```bash
python benchmark_ik2.py --model_dir ik_v3/results --n 1000
```

Expected output: ~77% DNN-init convergence vs. ~44% random, with roughly a
5× iteration speedup.

### 4.2 Programmatic inference

```python
import numpy as np
from ik_v3.infer import IKSolver

# 1. Load the pretrained solver (model + both scalers).
solver = IKSolver("ik_v3/results")

# 2. Build a target end-effector transform (4x4 homogeneous).
T_target = np.eye(4)
T_target[:3, 3] = [0.4, 0.1, 0.3]   # xyz in metres

# 3. Provide the robot's current joint configuration (seed, 6 radians).
q_current = np.zeros(6)

# 4. Solve.
q_star, info = solver.solve(T_target, q_current)
print("Joint solution (rad):", q_star)
print("Converged:", info["converged"], " iterations:", info["iters"])
```

---

## 5. Generating a Dataset

### 5.1 v3 seed-conditioned generator (recommended)

```bash
# 10M-row production dataset (~1.43 GB, several minutes on CPU)
python -m ik_v3.generate_dataset \
    --samples 10000000 \
    --out ik_v3/ur5e_seed_10m.csv \
    --seed_noise 0.5
```

**Key flags**

- `--samples` — number of (pose, seed, joints) rows.
- `--seed_noise` — Gaussian sigma (radians) used to perturb ground-truth
  joints into the "current-state" seed. The default 0.5 rad matches the
  production run.
- `--out` — output CSV path.

Smaller slices (10k, 50k, 100k, 200k, 500k, 1M) can be produced the same
way and were used to build the data-scaling sweep.

### 5.2 v1 pose-only generator (legacy)

```bash
python generate_pose_dataset.py --samples 100000 \
       --out ur5e_pose_dataset.csv
```

---

## 6. Training

### 6.1 Train the production v3 model

```bash
python -m ik_v3.train \
    --csv ik_v3/ur5e_seed_10m.csv \
    --epochs 150 \
    --out_dir ik_v3/results
```

**Training recipe (defaults in `ik_v3/train.py`)**

- Split: 80 / 10 / 10 (train / val / test), fixed random seed.
- Loss: `NormalizedSinCosLoss` (L2-normalize each sin/cos pair, then MSE).
- Optimiser: Adam, learning rate 1e-3.
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 5, min_lr 1e-6).
- Batch size 2,048, up to 150 epochs, early-stop patience 20.
- Gradient clipping max-norm 1.0.

**Outputs** are written to `--out_dir`:

```
model_best.pt              # best validation checkpoint
pose_scaler.pkl            # StandardScaler for 9D pose
seed_scaler.pkl            # StandardScaler for 6D seed
training_history.csv       # per-epoch train/val loss + MAE
```

### 6.2 Train a smaller-dataset slice (data-scaling sweep)

To reproduce any row of Phase 1, just swap the CSV:

```bash
python -m ik_v3.train --csv ur5e_seed_100k.csv \
       --epochs 150 --out_dir ik_v3/results_100k
```

### 6.3 Train the v1 legacy model (architectural ablation)

```bash
python train_ik_model.py --csv ur5e_pose_dataset.csv \
       --model_type resmlp_4x256 --epochs 150 \
       --out_dir results/100k/resmlp_4x256
```

Valid `--model_type` values: `mlp_3x128`, `mlp_3x256`, `mlp_4x256`,
`mlp_5x256`, `resmlp_2x128`, `resmlp_4x256`.

### 6.4 Train with differentiable-FK composite loss (experimental)

```bash
python train_ik_model_fk_loss.py --csv ur5e_pose_dataset.csv \
       --lambda_pos 1.0 --epochs 150 \
       --out_dir fk_loss_experiments/fk_loss_lambda_1.0
```

`--lambda_pos` weights the FK-position auxiliary term:

$$
L = \mathrm{MSE}_{\sin\cos} + \lambda_\text{pos}\cdot
    \mathrm{MSE}_{xyz}\!\left(\mathrm{FK}_{\mathrm{torch}}(\hat q),\, p_\text{target}\right).
$$

Sweeping via `python run_fk_study.py` reproduces the Phase 2.3 ablation.

---

## 7. Evaluation and Benchmarking

### 7.1 Per-joint test metrics + prediction plots (v1)

```bash
python evaluate_model.py --model_dir results/100k/resmlp_4x256
```

Produces `test_metrics.txt`, per-joint `pred_vs_true_q{1..6}.png`,
`loss_curve.png`, `lr_schedule.png`, and `val_mae_curve.png`.

### 7.2 Hybrid IK benchmark (v3 recommended)

```bash
python benchmark_ik2.py --model_dir ik_v3/results --n 1000 \
       --seed_noise 0.5 --tol_pos 1e-3 --tol_rot 0.01
```

**Flags**

- `--n` — number of random UR5e poses to test.
- `--seed_noise` — sigma (rad) for the controller-state seed perturbation.
- `--tol_pos` / `--tol_rot` — Newton–Raphson convergence tolerances
  (metres / radians).

Writes `benchmark_summary.json` with DNN-init vs. random-init convergence
rate, median iteration count, and mean speedup.

---

## 8. Inference Pipeline (detailed)

The full algorithm used by `IKSolver.solve(T_target, q_current)`:

1. Extract `pose_9d` from `T_target` using the 6D-rotation representation
   (Zhou et al., 2019).
2. Apply the saved `pose_scaler` and `seed_scaler`, concatenate to a 15D
   input.
3. Run one DNN forward pass → 12D sin/cos.
4. L2-normalize each (sin, cos) pair, then decode via `atan2` to get
   $\hat q \in \mathbb{R}^6$.
5. Damped Newton–Raphson polish:

$$
\Delta q = J^{\top}\!\left(J J^{\top} + \lambda^{2} I\right)^{-1} e,
\qquad \lambda^{2} = 10^{-4}
$$

   with per-step clamp $|\Delta q_i| \leq 0.5$ rad, until
   $\lVert e_\text{pos} \rVert < 1$ mm **and**
   $\lVert e_\text{rot} \rVert < 0.573°$, up to 200 iterations.

6. Return $q^{\star}$ with a convergence flag.

Typical CPU cost per solve: ~0.5 ms (DNN) + 6–10 Newton iterations ≈ 1–2 ms
total.

---

## 9. Reproducing the Full Study

```bash
# --- Phase 1: data-scaling sweep (v3 ResMLP) ---
for N in 10000 50000 100000 200000 500000 1000000 10000000 ; do
  python -m ik_v3.generate_dataset --samples $N \
         --out ik_v3/ur5e_seed_${N}.csv --seed_noise 0.5
  python -m ik_v3.train --csv ik_v3/ur5e_seed_${N}.csv \
         --epochs 150 --out_dir ik_v3/results_${N}
  python benchmark_ik2.py --model_dir ik_v3/results_${N} --n 1000
done

# --- Phase 2.1: architectural ablation (v1) ---
python run_study.py       # all widths / depths, 50k + 100k slices

# --- Phase 2.3: differentiable-FK composite-loss ablation ---
python run_fk_study.py    # sweeps lambda_pos in {0, 0.01, 0.1, 1.0}
```

---

## 10. Troubleshooting

**`FileNotFoundError: pose_scaler.pkl`**
You pointed `--model_dir` at a folder that does not contain the scaler
pickles. Retrain, or copy the scalers from `ik_v3/results/`.

**Benchmark shows 0% DNN convergence**
Almost always a seed-scaling mismatch. The inference scalers must be the
ones produced by the *same* training run as the model checkpoint — never
mix a v1 model with v3 scalers.

**Training loss diverges to NaN**
Check that the CSV is the v3 15D schema (9D pose + 6D seed + 12D sin/cos).
Running `ik_v3.train` on a 12D v1 CSV will silently produce garbage. Also
verify the gradient clip is enabled (default `max_norm=1.0`).

**Newton–Raphson never converges**
Relax `--tol_pos` / `--tol_rot`, or raise the max iteration cap inside
`benchmark_ik2.py`. Very close-to-singular targets can require >200
iterations even from a good DNN seed.

**Out-of-memory on the 10M CSV**
`ik_v3/train.py` streams batches from disk; if you still run out, drop
the batch size from 2,048 to 1,024 via `--batch`.

---

## 11. Citation

If you use this software, please cite:

```bibtex
@misc{desta2026dnnik,
  author = {Dandi Desta},
  title  = {DNN-Initialized Inverse Kinematics for the UR5e},
  year   = {2026},
  note   = {ES 259 Final Project, Stanford University},
  url    = {https://github.com/dantdani/ES_259_final_project}
}
```

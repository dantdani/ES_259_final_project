\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{parskip}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue
}

% ---- listing style for shell / python ----
\definecolor{bg}{rgb}{0.96,0.96,0.96}
\definecolor{kw}{rgb}{0.10,0.20,0.60}
\definecolor{cm}{rgb}{0.25,0.55,0.25}
\definecolor{st}{rgb}{0.60,0.20,0.20}

\lstdefinestyle{shell}{
    backgroundcolor=\color{bg},
    basicstyle=\ttfamily\small,
    breaklines=true,
    columns=fullflexible,
    frame=single,
    framesep=4pt,
    xleftmargin=6pt,
    language=bash,
    keywordstyle=\color{kw}\bfseries,
    commentstyle=\color{cm}\itshape,
    stringstyle=\color{st},
    showstringspaces=false
}

\lstdefinestyle{py}{
    backgroundcolor=\color{bg},
    basicstyle=\ttfamily\small,
    breaklines=true,
    columns=fullflexible,
    frame=single,
    framesep=4pt,
    xleftmargin=6pt,
    language=Python,
    keywordstyle=\color{kw}\bfseries,
    commentstyle=\color{cm}\itshape,
    stringstyle=\color{st},
    showstringspaces=false
}

\title{\textbf{DNN-Initialized Inverse Kinematics for the UR5e}\\
       \large Software Usage Guide}
\author{Dandi Desta \\ ES 259 --- Final Project}
\date{April 22, 2026}

\begin{document}
\maketitle

\tableofcontents
\vspace{1em}
\hrule
\vspace{1em}

% =====================================================================
\section{Overview}
% =====================================================================

This document describes how to install, configure, and use the
\texttt{ES\_259\_final\_project} software package. The project implements a
hybrid inverse-kinematics (IK) solver for the Universal Robots UR5e: a
seed-conditioned deep neural network produces a joint-angle initialization,
and a damped Newton--Raphson routine polishes that guess to
sub-millimetre precision.

The software supports the full research pipeline:
\begin{enumerate}[leftmargin=*]
    \item Synthetic dataset generation via forward kinematics (FK).
    \item Training the production v3 ResMLP (seed-conditioned) and legacy
          v1 pose-only MLP/ResMLP variants.
    \item Benchmarking DNN-initialized IK vs. random-initialized IK.
    \item Running inference on new end-effector targets.
    \item Reproducing the data-scaling sweep (10k $\rightarrow$ 10M) and
          the differentiable-FK composite-loss ablation.
\end{enumerate}

% =====================================================================
\section{Repository Layout}
% =====================================================================

\begin{lstlisting}[style=shell]
ES_259_final_project/
|-- ik_v3/                      # PRODUCTION code (seed-conditioned ResMLP)
|   |-- model.py                #   SeedConditionedIKModel, ResidualBlock
|   |-- train.py                #   training loop (ik_v3.train CLI)
|   |-- infer.py                #   IKSolver class (DNN + Newton polish)
|   |-- generate_dataset.py     #   vectorised PoE FK data generator
|   |-- representations.py      #   6D rotation, sin/cos encode/decode
|   |-- results/                #   10M model_best.pt + scalers
|   +-- ur5e_seed_*.csv         #   datasets (10k ... 10M)
|
|-- ik_v2/                      # Exploratory: MoE / region experts / refinement
|
|-- model.py                    # v1 legacy model (pose-only MLP/ResMLP)
|-- train_ik_model.py           # v1 training script
|-- generate_pose_dataset.py    # v1 dataset generator
|-- benchmark_ik.py             # v1-style benchmark
|-- benchmark_ik2.py            # v3-style benchmark (recommended)
|-- evaluate_model.py           # per-joint test metrics + plots
|-- infer_ik.py                 # v1 inference helper
|
|-- train_ik_model_fk_loss.py   # experimental composite (FK) loss trainer
|-- model_fk_loss.py            # FK-loss flat MLP model
|-- utils_torch_fk.py           # differentiable torch PoE FK
|-- run_fk_study.py             # lambda_pos sweep driver
|-- run_study.py / run_study_live.py  # v1 architecture sweep drivers
|
|-- results/                    # v1 architecture-sweep artifacts
|-- results_v1/                 # v1 10M-trained artifacts
|-- pose_results/               # v1 baseline artifacts
|-- fk_loss_experiments/        # differentiable-FK ablation artifacts
|
|-- methodology.txt             # full research writeup
+-- accuracy_vs_dataset_size.png
\end{lstlisting}

% =====================================================================
\section{Installation}
% =====================================================================

\subsection{Requirements}
\begin{itemize}[leftmargin=*]
    \item Python 3.10 or newer
    \item Windows, macOS, or Linux (CPU-only supported; GPU optional)
    \item $\sim$4~GB free disk (full 10M-sample CSV is $\sim$1.43~GB)
\end{itemize}

\subsection{Clone and set up a virtual environment}

\begin{lstlisting}[style=shell]
git clone https://github.com/dantdani/ES_259_final_project.git
cd ES_259_final_project

# Create + activate a venv
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
\end{lstlisting}

\subsection{Install Python dependencies}

\begin{lstlisting}[style=shell]
pip install --upgrade pip
pip install torch numpy pandas scikit-learn matplotlib tqdm joblib
\end{lstlisting}

\noindent
GPU users may instead install the CUDA build of PyTorch from
\href{https://pytorch.org/get-started/locally/}{pytorch.org}. All scripts
default to CPU.

% =====================================================================
\section{Quick Start (run the pretrained 10M model)}
% =====================================================================

The 10M-trained production model ships inside \texttt{ik\_v3/results/}
(model\_best.pt + pose\_scaler.pkl + seed\_scaler.pkl). You can use it
immediately without retraining.

\subsection{Benchmark on 1,000 random UR5e poses}

\begin{lstlisting}[style=shell]
python benchmark_ik2.py --model_dir ik_v3/results --n 1000
\end{lstlisting}

\noindent
Expected output: $\sim$77\% DNN-init convergence vs. $\sim$44\% random,
with roughly a 5$\times$ iteration speedup.

\subsection{Programmatic inference}

\begin{lstlisting}[style=py]
import numpy as np
from ik_v3.infer import IKSolver

# 1. Load the pretrained solver (model + both scalers).
solver = IKSolver("ik_v3/results")

# 2. Build a target end-effector transform (4x4 homogeneous).
T_target = np.eye(4)
T_target[:3, 3] = [0.4, 0.1, 0.3]  # xyz in metres

# 3. Provide the robot's current joint configuration (seed, 6 radians).
q_current = np.zeros(6)

# 4. Solve.
q_star, info = solver.solve(T_target, q_current)
print("Joint solution (rad):", q_star)
print("Converged:", info["converged"],
      " iterations:", info["iters"])
\end{lstlisting}

% =====================================================================
\section{Generating a Dataset}
% =====================================================================

\subsection{v3 seed-conditioned generator (recommended)}

\begin{lstlisting}[style=shell]
# 10M-row production dataset (~1.43 GB, several minutes on CPU)
python -m ik_v3.generate_dataset ^
    --samples 10000000 ^
    --out ik_v3/ur5e_seed_10m.csv ^
    --seed_noise 0.5
\end{lstlisting}

\noindent\textbf{Key flags}
\begin{itemize}[leftmargin=*]
    \item \texttt{--samples} --- number of (pose, seed, joints) rows.
    \item \texttt{--seed\_noise} --- Gaussian sigma (radians) used to
          perturb ground-truth joints into the ``current-state'' seed.
          The default 0.5~rad matches the production run.
    \item \texttt{--out} --- output CSV path.
\end{itemize}

\noindent Smaller slices (10k, 50k, 100k, 200k, 500k, 1M) can be produced
the same way and were used to build the data-scaling sweep.

\subsection{v1 pose-only generator (legacy)}

\begin{lstlisting}[style=shell]
python generate_pose_dataset.py --samples 100000 \
       --out ur5e_pose_dataset.csv
\end{lstlisting}

% =====================================================================
\section{Training}
% =====================================================================

\subsection{Train the production v3 model}

\begin{lstlisting}[style=shell]
python -m ik_v3.train ^
    --csv ik_v3/ur5e_seed_10m.csv ^
    --epochs 150 ^
    --out_dir ik_v3/results
\end{lstlisting}

\noindent\textbf{Training recipe (defaults in \texttt{ik\_v3/train.py})}
\begin{itemize}[leftmargin=*]
    \item Split: 80 / 10 / 10 (train / val / test), fixed random seed.
    \item Loss: \texttt{NormalizedSinCosLoss} (L2-normalize each sin/cos
          pair, then MSE).
    \item Optimiser: Adam, learning rate 1e-3.
    \item Scheduler: ReduceLROnPlateau (factor 0.5, patience 5,
          min\_lr 1e-6).
    \item Batch size 2{,}048, up to 150 epochs, early-stop patience 20.
    \item Gradient clipping max-norm 1.0.
\end{itemize}

\noindent\textbf{Outputs} are written to \texttt{--out\_dir}:
\begin{lstlisting}[style=shell]
model_best.pt              # best validation checkpoint
pose_scaler.pkl            # StandardScaler for 9D pose
seed_scaler.pkl            # StandardScaler for 6D seed
training_history.csv       # per-epoch train/val loss + MAE
\end{lstlisting}

\subsection{Train a smaller-dataset slice (data-scaling sweep)}

To reproduce any row of Phase~1, just swap the CSV:

\begin{lstlisting}[style=shell]
python -m ik_v3.train --csv ur5e_seed_100k.csv \
       --epochs 150 --out_dir ik_v3/results_100k
\end{lstlisting}

\subsection{Train the v1 legacy model (architectural ablation)}

\begin{lstlisting}[style=shell]
python train_ik_model.py --csv ur5e_pose_dataset.csv \
       --model_type resmlp_4x256 --epochs 150 \
       --out_dir results/100k/resmlp_4x256
\end{lstlisting}

\noindent Valid \texttt{--model\_type} values: \texttt{mlp\_3x128},
\texttt{mlp\_3x256}, \texttt{mlp\_4x256}, \texttt{mlp\_5x256},
\texttt{resmlp\_2x128}, \texttt{resmlp\_4x256}.

\subsection{Train with differentiable-FK composite loss (experimental)}

\begin{lstlisting}[style=shell]
python train_ik_model_fk_loss.py --csv ur5e_pose_dataset.csv \
       --lambda_pos 1.0 --epochs 150 \
       --out_dir fk_loss_experiments/fk_loss_lambda_1.0
\end{lstlisting}

\noindent\texttt{--lambda\_pos} weights the FK-position auxiliary term:
\[
L = \mathrm{MSE}_{\sin\cos} + \lambda_\text{pos}\cdot
    \mathrm{MSE}_{xyz}\!\left(\mathrm{FK}_{\mathrm{torch}}(\hat q),\, p_\text{target}\right).
\]
Sweeping via \texttt{python run\_fk\_study.py} reproduces the Phase~2.3
ablation.

% =====================================================================
\section{Evaluation and Benchmarking}
% =====================================================================

\subsection{Per-joint test metrics + prediction plots (v1)}

\begin{lstlisting}[style=shell]
python evaluate_model.py --model_dir results/100k/resmlp_4x256
\end{lstlisting}

\noindent Produces \texttt{test\_metrics.txt}, per-joint
\texttt{pred\_vs\_true\_q\{1..6\}.png}, \texttt{loss\_curve.png},
\texttt{lr\_schedule.png}, and \texttt{val\_mae\_curve.png}.

\subsection{Hybrid IK benchmark (v3 recommended)}

\begin{lstlisting}[style=shell]
python benchmark_ik2.py --model_dir ik_v3/results --n 1000 \
       --seed_noise 0.5 --tol_pos 1e-3 --tol_rot 0.01
\end{lstlisting}

\noindent\textbf{Flags}
\begin{itemize}[leftmargin=*]
    \item \texttt{--n} --- number of random UR5e poses to test.
    \item \texttt{--seed\_noise} --- sigma (rad) for the controller-state
          seed perturbation.
    \item \texttt{--tol\_pos} / \texttt{--tol\_rot} --- Newton--Raphson
          convergence tolerances (metres / radians).
\end{itemize}

\noindent Writes \texttt{benchmark\_summary.json} with DNN-init vs.
random-init convergence rate, median iteration count, and mean speedup.

% =====================================================================
\section{Inference Pipeline (detailed)}
% =====================================================================

The full algorithm used by \texttt{IKSolver.solve(T\_target, q\_current)}:

\begin{enumerate}[leftmargin=*]
    \item Extract \texttt{pose\_9d} from \texttt{T\_target} using the
          6D-rotation representation (Zhou et al., 2019).
    \item Apply the saved \texttt{pose\_scaler} and \texttt{seed\_scaler},
          concatenate to a 15D input.
    \item Run one DNN forward pass $\rightarrow$ 12D sin/cos.
    \item L2-normalize each (sin, cos) pair, then decode via
          \texttt{atan2} to get $\hat q \in \mathbb{R}^6$.
    \item Damped Newton--Raphson polish:
          \[
              \Delta q = J^{\top}\left(J J^{\top} + \lambda^2 I\right)^{-1} e,
              \qquad \lambda^2 = 10^{-4}
          \]
          with per-step clamp $|\Delta q_i| \leq 0.5$ rad, until
          $\|e_\text{pos}\| < 1$~mm \emph{and}
          $\|e_\text{rot}\| < 0.573^\circ$, up to 200 iterations.
    \item Return $q^\star$ with a convergence flag.
\end{enumerate}

Typical CPU cost per solve: $\sim$0.5 ms (DNN) + 6--10 Newton iterations
$\approx$ 1--2 ms total.

% =====================================================================
\section{Reproducing the Full Study}
% =====================================================================

\begin{lstlisting}[style=shell]
# --- Phase 1: data-scaling sweep (v3 ResMLP) ---
for N in 10000 50000 100000 200000 500000 1000000 10000000 ; do
  python -m ik_v3.generate_dataset --samples $N \
         --out ik_v3/ur5e_seed_${N}.csv --seed_noise 0.5
  python -m ik_v3.train --csv ik_v3/ur5e_seed_${N}.csv \
         --epochs 150 --out_dir ik_v3/results_${N}
  python benchmark_ik2.py --model_dir ik_v3/results_${N} --n 1000
done

# --- Phase 2.1: architectural ablation (v1) ---
python run_study.py      # all widths / depths, 50k + 100k slices

# --- Phase 2.3: differentiable-FK composite-loss ablation ---
python run_fk_study.py   # sweeps lambda_pos in {0, 0.01, 0.1, 1.0}
\end{lstlisting}

% =====================================================================
\section{Troubleshooting}
% =====================================================================

\begin{description}[leftmargin=0pt, style=nextline]

\item[\texttt{FileNotFoundError: pose\_scaler.pkl}]
    You pointed \texttt{--model\_dir} at a folder that does not contain
    the scaler pickles. Retrain, or copy the scalers from
    \texttt{ik\_v3/results/}.

\item[Benchmark shows 0\% DNN convergence]
    Almost always a seed-scaling mismatch. The inference scalers must be
    the ones produced by the \emph{same} training run as the model
    checkpoint --- never mix a v1 model with v3 scalers.

\item[Training loss diverges to NaN]
    Check that the CSV is the v3 15D schema
    (9D pose + 6D seed + 12D sin/cos). Running \texttt{ik\_v3.train} on
    a 12D v1 CSV will silently produce garbage. Also verify the gradient
    clip is enabled (default \texttt{max\_norm=1.0}).

\item[Newton--Raphson never converges]
    Relax \texttt{--tol\_pos} / \texttt{--tol\_rot}, or raise the max
    iteration cap inside \texttt{benchmark\_ik2.py}. Very close-to-
    singular targets can require $>$200 iterations even from a good
    DNN seed.

\item[Out-of-memory on the 10M CSV]
    \texttt{ik\_v3/train.py} streams batches from disk; if you still run
    out, drop the batch size from 2{,}048 to 1{,}024 via \texttt{--batch}.

\end{description}

% =====================================================================
\section{Citation}
% =====================================================================

If you use this software, please cite:

\begin{lstlisting}[style=shell]
@misc{desta2026dnnik,
  author = {Dandi Desta},
  title  = {DNN-Initialized Inverse Kinematics for the UR5e},
  year   = {2026},
  note   = {ES 259 Final Project, Stanford University},
  url    = {https://github.com/dantdani/ES_259_final_project}
}
\end{lstlisting}

\end{document}

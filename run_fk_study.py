import os
import sys
import subprocess
import pandas as pd
import re

datasets = [
    {"size": "50k", "file": "ur5e_pose_dataset.csv", "subset": 500000},
]
lambdas = [0.0, 0.01, 0.1, 1.0]
run_names = {
    0.0: 'baseline_original_loss',
    0.01: 'fk_loss_lambda_0.01',
    0.1: 'fk_loss_lambda_0.1',
    1.0: 'fk_loss_lambda_1.0'
}

os.makedirs('fk_loss_experiments', exist_ok=True)
EPOCHS = 150 # Configure for real evaluation. Use 150 for actual experiments

results = []

for ds in datasets:
    for lam in lambdas:
        run_name = run_names[lam]
        out_dir = f"fk_loss_experiments/{run_name}"
        
        cmd = [
            sys.executable, "train_ik_model_fk_loss.py",
            "--csv", ds["file"],
            "--out_dir", out_dir,
            "--arch", "resmlp",
            "--hidden_dim", "256",
            "--num_blocks", "4",
            "--epochs", str(EPOCHS),
            "--patience", "3",
            "--subset_size", str(ds["subset"]),
            "--lambda_pos", str(lam)
        ]
        
        print(f"Running {run_name} on {ds['size']} (150 epochs)...", end="", flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f" ERROR: {res.stderr}")
            continue
            
        # Parse metrics
        metric_file = f"{out_dir}/test_metrics.txt"
        val_loss = rmse = mae = fk_pos = fk_rot = None
        if os.path.exists(metric_file):
            with open(metric_file, "r") as f:
                m = f.read()
            from re import search
            rmse_m = search(r"rmse_rad:\s+([\d\.]+)", m)
            mae_m = search(r"mae_rad:\s+([\d\.]+)", m)
            fk_m = search(r"mean_pos_error_mm:\s+([\d\.]+)", m)
            fk_rm = search(r"mean_rot_error_rad:\s+([\d\.]+)", m)
            
            rmse = float(rmse_m.group(1)) if rmse_m else None
            mae = float(mae_m.group(1)) if mae_m else None
            fk_pos = float(fk_m.group(1)) if fk_m else None
            fk_rot = float(fk_rm.group(1)) if fk_rm else None

        v_m = re.search(r"Best validation loss:\s+([\d\.]+)", res.stdout)
        val_loss = float(v_m.group(1)) if v_m else None

        results.append({
            "Lambda Target": lam, "Run Name": run_name, "Val Loss": val_loss,
            "RMSE (rad)": rmse, "MAE (rad)": mae, "FK Pos Error (mm)": fk_pos,
            "FK Rot Error (rad)": fk_rot
        })
        print(f" Done. FK Pos Error: {fk_pos} mm")

df = pd.DataFrame(results)
print("\n--- FK LOSS ABLATION RESULTS OVERVIEW ---")
print(df.to_string(index=False))
df.to_csv("fk_loss_experiments/ablation_summary.csv", index=False)

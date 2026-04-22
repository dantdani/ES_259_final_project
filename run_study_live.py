import os
import sys
import subprocess
import pandas as pd
import re

configs = [
    {"name": "mlp_3x128", "arch": "mlp", "num_hidden_layers": 3, "hidden_dim": 128},
    {"name": "mlp_3x256", "arch": "mlp", "num_hidden_layers": 3, "hidden_dim": 256},
    {"name": "mlp_4x256", "arch": "mlp", "num_hidden_layers": 4, "hidden_dim": 256},
    {"name": "mlp_5x256", "arch": "mlp", "num_hidden_layers": 5, "hidden_dim": 256},
    {"name": "resmlp_2x128", "arch": "resmlp", "num_blocks": 2, "hidden_dim": 128},
    {"name": "resmlp_4x256", "arch": "resmlp", "num_blocks": 4, "hidden_dim": 256}
]

datasets = [
    {"size": "50k", "file": "ur5e_pose_dataset.csv", "subset": 50000},
    {"size": "100k", "file": "ur5e_pose_dataset.csv", "subset": 100000}
]

os.makedirs("results", exist_ok=True)
EPOCHS = 10 # Capped so we don't timeout the LLM turn!
results = []

for ds in datasets:
    ds_size = ds["size"]
    ds_file = ds["file"]
    
    if not os.path.exists(ds_file):
        print(f"Skipping {ds_file}")
        continue

    for c in configs:
        out_dir = f"results/{ds_size}/{c['name']}"
        cmd = [
            sys.executable, "train_ik_model.py",
            "--csv", ds_file,
            "--out_dir", out_dir,
            "--arch", c["arch"],
            "--hidden_dim", str(c["hidden_dim"]),
            "--epochs", str(EPOCHS),
            "--patience", "3",
            "--subset_size", str(ds["subset"])
        ]
        
        if c["arch"] == "mlp":
            cmd.extend(["--num_hidden_layers", str(c["num_hidden_layers"])])
        if c["arch"] == "resmlp":
            cmd.extend(["--num_blocks", str(c["num_blocks"])])
            
        print(f"Running {c['name']} on {ds_size} (10 epochs)...", end="", flush=True)
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(" ERROR")
            # print(res.stderr)
            continue
            
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
            "Dataset": ds_size, "Model": c["name"], "Val Loss": val_loss,
            "RMSE (rad)": rmse, "MAE (rad)": mae, "FK Pos Error (mm)": fk_pos,
            "FK Rot Error (rad)": fk_rot
        })
        print(f" Done. Val Loss: {val_loss}, MAE: {mae}")

df = pd.DataFrame(results)
print("\n--- RESULTS OVERVIEW ---")
print(df.to_markdown(index=False))
df.to_csv("results/summary.csv", index=False)

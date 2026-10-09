import os

p_v23 = "/root/Workspace/xy/DiT/archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/eval_samples_ctrl"
p_v21 = "/root/Workspace/xy/DiT/archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/eval_samples_ctrl"
p_stdaug = "/root/Workspace/xy/DiT/assets/results/std_callig_aug/eval_samples_ctrl"

for name, p in [("v23", p_v23), ("v21", p_v21), ("std_aug", p_stdaug)]:
    if os.path.exists(p):
        items = sorted(os.listdir(p))
        print(name, f"{len(items)} items:", items[:8], items[-3:] if len(items) > 3 else [])
    else:
        print(name, "NOT FOUND")

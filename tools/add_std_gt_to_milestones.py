import os
import csv
import shutil

base = "/root/Workspace/xy/DiT"
out_dir = os.path.join(base, "exp_milestones/eval200_outputs")
eval_csv = os.path.join(base, "exp-std/csv/eval200_fixed.csv")

rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))

# 1. 00_input_std
std_dir = os.path.join(out_dir, "00_input_std")
os.makedirs(std_dir, exist_ok=True)
# 2. 99_ground_truth
gt_dir = os.path.join(out_dir, "99_ground_truth")
os.makedirs(gt_dir, exist_ok=True)

for i in range(10):
    r = rows[i]
    char, callig, script = r["character"], r["calligrapher"], r["script"]
    std_src = os.path.join(base, r["std_path"])
    gt_src = os.path.join(base, r["image_path"])
    
    # 复制 std
    if os.path.exists(std_src):
        shutil.copy2(std_src, os.path.join(std_dir, f"{i:02d}.png"))
        shutil.copy2(std_src, os.path.join(std_dir, f"{i:02d}_{char}_{callig}_{script}.png"))
        
    # 复制 gt
    if os.path.exists(gt_src):
        shutil.copy2(gt_src, os.path.join(gt_dir, f"{i:02d}.png"))
        shutil.copy2(gt_src, os.path.join(gt_dir, f"{i:02d}_{char}_{callig}_{script}.png"))

print("✓ std 和 gt 基准样本已整理到 eval200_outputs!")

import os, sys
import subprocess
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create local directory
local_dir = "exp/head_to_head"
os.makedirs(local_dir, exist_ok=True)
os.makedirs(os.path.join(local_dir, "moyi"), exist_ok=True)
os.makedirs(os.path.join(local_dir, "ours"), exist_ok=True)
os.makedirs(os.path.join(local_dir, "std"), exist_ok=True)
os.makedirs(os.path.join(local_dir, "gt"), exist_ok=True)

# 8 Selected Samples
samples = [
    {"idx": 10, "callig": "王羲之", "script": "楷", "char": "旨", "img_id": "017513.png"},
    {"idx": 11, "callig": "王羲之", "script": "行", "char": "好", "img_id": "017687.png"},
    {"idx": 21, "callig": "颜真卿", "script": "楷", "char": "其", "img_id": "020672.png"},
    {"idx": 22, "callig": "颜真卿", "script": "行", "char": "憫", "img_id": "004219.png"},
    {"idx": 13, "callig": "米芾", "script": "行", "char": "墟", "img_id": "018038.png"},
    {"idx": 19, "callig": "赵孟頫", "script": "行", "char": "匠", "img_id": "014290.png"},
    {"idx": 20, "callig": "赵孟頫", "script": "隶", "char": "盤", "img_id": "037234.png"},
    {"idx": 6,  "callig": "柳公权", "script": "楷", "char": "連", "img_id": "010042.png"},
]

# Fetch from 48 (Moyi)
for s in samples:
    idx = s["idx"]
    c = s["callig"]
    sc = s["script"]
    ch = s["char"]
    remote_path = f"48:/home/ds/Workspace/moyi/results/moyi_top10_rf/eval_samples_step20000/moyi_g{idx}_{c}_{sc}_{ch}.png"
    local_path = os.path.join(local_dir, "moyi", f"g{idx}.png")
    subprocess.run(["scp", remote_path, local_path], check=True)

# Fetch from 4090 (Ours, Std, GT)
for s in samples:
    idx = s["idx"]
    img_id = s["img_id"]
    
    # Std input
    std_remote = f"4090:/root/Workspace/xy/DiT/exp-std/runs_purestd/eval_samples_ctrl/eval200fix_input_g/g{idx}.png"
    std_local = os.path.join(local_dir, "std", f"g{idx}.png")
    subprocess.run(["scp", std_remote, std_local], check=True)
    
    # Ours (Step 22500)
    our_remote = f"4090:/root/Workspace/xy/DiT/exp-std/runs_purestd/eval_samples_ctrl/step0022500/eval200fix/g{idx}.png"
    our_local = os.path.join(local_dir, "ours", f"g{idx}.png")
    subprocess.run(["scp", our_remote, our_local], check=True)
    
    # GT
    gt_remote = f"4090:/root/Workspace/xy/DiT/data/top10_style23/imgs/{img_id}"
    gt_local = os.path.join(local_dir, "gt", f"g{idx}.png")
    subprocess.run(["scp", gt_remote, gt_local], check=True)

print("All sample images fetched successfully!")

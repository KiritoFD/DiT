import sys
from pathlib import Path
from PIL import Image
import numpy as np
import csv

indices = [0, 46, 75, 80, 88, 103, 121, 133, 155, 172]
base_dir = Path("exp_img/unet/u100_base_55k")
dic_dir = Path("exp_img/unet/u100_dic_b768_50k")
distill_dir = Path("exp_img/unet/u100_writer_distill_10k")
gt_dir = Path("exp_img/common/eval200fix_gt")

rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))

print(f"{'Idx':<5} {'Char':<4} {'Writer/Script':<10} {'Base Ink':<10} {'DiC Ink':<10} {'Distill Ink':<12} {'GT Ink':<10}")
print("-" * 75)
for idx in indices:
    r = rows[idx]
    ch = r["character"]
    ws = f"{r['calligrapher']}·{r['script']}"
    
    b_path = base_dir / f"g{idx}.png"
    d_path = dic_dir / f"g{idx}.png"
    dist_path = distill_dir / f"g{idx}.png"
    gt_path = gt_dir / f"g{idx}.png"
    
    if not (b_path.exists() and d_path.exists()):
        continue
        
    b_im = np.asarray(Image.open(b_path).convert("L"), dtype=np.float32) / 255.0
    d_im = np.asarray(Image.open(d_path).convert("L"), dtype=np.float32) / 255.0
    dist_im = np.asarray(Image.open(dist_path).convert("L"), dtype=np.float32) / 255.0 if dist_path.exists() else None
    gt_im = np.asarray(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0 if gt_path.exists() else None
    
    # Ink is dark pixels: img < 0.5
    b_ink = float((b_im < 0.5).mean()) * 100.0
    d_ink = float((d_im < 0.5).mean()) * 100.0
    dist_ink = float((dist_im < 0.5).mean()) * 100.0 if dist_im is not None else 0.0
    gt_ink = float((gt_im < 0.5).mean()) * 100.0 if gt_im is not None else 0.0
    
    print(f"#{idx:<4} 【{ch}】 {ws:<10} {b_ink:6.2f}%    {d_ink:6.2f}%    {dist_ink:6.2f}%      {gt_ink:6.2f}%")

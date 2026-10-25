import sys
from pathlib import Path
from PIL import Image
import numpy as np
import csv

indices = [0, 46, 75, 80, 88, 103, 121, 133, 155, 172]
base_dir = Path("exp_img/unet/u100_base_55k")
dic_dir = Path("exp_img/unet/u100_dic_b768_50k")
distill_dir = Path("exp_img/unet/u100_writer_distill_10k")
in_dir = Path("exp_img/common/eval200fix_input_g")
gt_dir = Path("exp_img/common/eval200fix_gt")

rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))

print("Stroke & Topology Analysis (Base vs DiC vs Distill):")
print("-" * 80)

for idx in indices:
    r = rows[idx]
    ch = r["character"]
    ws = f"{r['calligrapher']}·{r['script']}"
    
    b_path = base_dir / f"g{idx}.png"
    d_path = dic_dir / f"g{idx}.png"
    dist_path = distill_dir / f"g{idx}.png"
    in_path = in_dir / f"g{idx}.png"
    
    if not (b_path.exists() and d_path.exists() and in_path.exists()):
        continue
        
    b_im = np.asarray(Image.open(b_path).convert("L"), dtype=np.float32) / 255.0
    d_im = np.asarray(Image.open(d_path).convert("L"), dtype=np.float32) / 255.0
    dist_im = np.asarray(Image.open(dist_path).convert("L"), dtype=np.float32) / 255.0
    in_im = np.asarray(Image.open(in_path).convert("L").resize(b_im.shape[::-1]), dtype=np.float32) / 255.0
    
    # In input_g, skeleton/contour is dark or bright?
    # Usually standard soft glyph has ink as bright or dark:
    # In DiT/Unet, std_soft is ink mass. Let's check:
    in_ink = in_im > 0.1 if in_im.mean() < 0.5 else in_im < 0.9
    
    b_dark = b_im < 0.5
    d_dark = d_im < 0.5
    dist_dark = dist_im < 0.5
    
    # Missing stroke pixels where input skeleton has stroke but generated image is white:
    b_miss = (in_ink & (~b_dark)).mean() * 100.0
    d_miss = (in_ink & (~d_dark)).mean() * 100.0
    dist_miss = (in_ink & (~dist_dark)).mean() * 100.0
    
    # Extra ink outside skeleton (dilation/bleeding):
    b_extra = ((~in_ink) & b_dark).mean() * 100.0
    d_extra = ((~in_ink) & d_dark).mean() * 100.0
    dist_extra = ((~in_ink) & dist_dark).mean() * 100.0
    
    print(f"#{idx:<3} 【{ch}】 {ws:<10} | Skel Missing: Base={b_miss:4.1f}%, DiC={d_miss:4.1f}%, Dist={dist_miss:4.1f}% | Bleed: Base={b_extra:4.1f}%, DiC={d_extra:4.1f}%")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect remote dataset data quality, grey ratio, color contamination, and background inversion.
"""

import subprocess
import json
import sys

REMOTE_HOST = "root@10.176.54.17"
REMOTE_PORT = "36430"

REMOTE_SCRIPT = """
import csv, os, random, json
import numpy as np
from PIL import Image

BASE = '/root/Workspace/xy/DiT'
CSV_PATH = os.path.join(BASE, '5script/train_top30_clean.csv')

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f'Total rows in train_top30_clean.csv: {len(rows)}')

random.seed(42)
sample_rows = random.sample(rows, min(5000, len(rows)))

grey_ratios = []
color_stds = []
dark_bg = 0
light_bg = 0
dirty_samples = []

for r in sample_rows:
    p = r['image_path']
    if p.startswith('final_images/'):
        p = p.replace('final_images/', 'final_imgs_256/', 1)
    full_path = os.path.join(BASE, p)
    if not os.path.exists(full_path):
        continue
    try:
        im = Image.open(full_path).convert('RGB')
        arr = np.asarray(im, dtype=np.float32) / 255.0
        
        # 1. Grey ratio (midtones: 0.15 - 0.85)
        mid_mask = (arr > 0.15) & (arr < 0.85)
        grey_ratio = float(mid_mask.mean())
        grey_ratios.append(grey_ratio)
        
        # 2. Color saturation (std across RGB channels)
        c_std = float(arr.std(axis=2).mean())
        color_stds.append(c_std)
        
        # 3. Corner background brightness
        corners = [arr[:10,:10], arr[:10,-10:], arr[-10:,:10], arr[-10:,-10:]]
        bg_val = float(np.mean([c.mean() for c in corners]))
        if bg_val < 0.3:
            dark_bg += 1
        elif bg_val > 0.7:
            light_bg += 1
            
        if grey_ratio > 0.20 or c_std > 0.04:
            if len(dirty_samples) < 15:
                dirty_samples.append({
                    'calligrapher': r.get('calligrapher', ''),
                    'script': r.get('script', ''),
                    'character': r.get('character', ''),
                    'grey_ratio': round(grey_ratio, 3),
                    'color_std': round(c_std, 3),
                    'path': p
                })
    except Exception as e:
        pass

res = {
    'total_checked': len(grey_ratios),
    'grey_gt_015_pct': round(100.0 * sum(1 for g in grey_ratios if g > 0.15) / len(grey_ratios), 2),
    'grey_gt_020_pct': round(100.0 * sum(1 for g in grey_ratios if g > 0.20) / len(grey_ratios), 2),
    'grey_gt_025_pct': round(100.0 * sum(1 for g in grey_ratios if g > 0.25) / len(grey_ratios), 2),
    'grey_gt_030_pct': round(100.0 * sum(1 for g in grey_ratios if g > 0.30) / len(grey_ratios), 2),
    'color_gt_004_pct': round(100.0 * sum(1 for c in color_stds if c > 0.04) / len(color_stds), 2),
    'dark_bg_pct': round(100.0 * dark_bg / len(grey_ratios), 2),
    'light_bg_pct': round(100.0 * light_bg / len(grey_ratios), 2),
    'dirty_samples': dirty_samples
}

print('__JSON_START__')
print(json.dumps(res, ensure_ascii=False, indent=2))
print('__JSON_END__')
"""

def main():
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-p", REMOTE_PORT, REMOTE_HOST,
        f"/opt/conda/bin/python -c \"{REMOTE_SCRIPT}\""
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print("SSH Error:", p.stderr)
        return
    out = p.stdout
    if "__JSON_START__" in out:
        json_str = out.split("__JSON_START__")[1].split("__JSON_END__")[0]
        data = json.loads(json_str)
        print("Remote Dataset Quality Report:")
        print(f"Total Images Sampled: {data['total_checked']}")
        print(f"  Grey Ratio > 0.15 : {data['grey_gt_015_pct']}%")
        print(f"  Grey Ratio > 0.20 : {data['grey_gt_020_pct']}%  <-- Muddy / Unclean Backgrounds")
        print(f"  Grey Ratio > 0.25 : {data['grey_gt_025_pct']}%")
        print(f"  Grey Ratio > 0.30 : {data['grey_gt_030_pct']}%  <-- Extreme Muddy Noise")
        print(f"  Color Contamination (c_std > 0.04): {data['color_gt_004_pct']}%")
        print(f"  Dark Background (Black Canvas) : {data['dark_bg_pct']}%")
        print(f"  Light Background (White Canvas): {data['light_bg_pct']}%")
        print("\nSample Dirty Entries Found:")
        for s in data['dirty_samples']:
            print(f"  [{s['calligrapher']}] {s['script']}-{s['character']}: grey_ratio={s['grey_ratio']}, color_std={s['color_std']}, path={s['path']}")
    else:
        print(out)

if __name__ == "__main__":
    main()

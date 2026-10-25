import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import csv
import json
import pandas as pd
import numpy as np
from PIL import Image
from tools.analysis.run_local_distribution_alignment import measure_stroke_properties
CSV_PATH = ROOT / "exp-std/csv/eval_top2_100.csv"
GEN_DIR = ROOT / "assets/results/top2_u100_base_50k"
TRAIN_DIST_JSON = ROOT / "assets/stroke_width_trainset_distribution.json"

with open(TRAIN_DIST_JSON, "r", encoding="utf-8") as f:
    train_dist = json.load(f)

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
records = []

for idx, r in enumerate(rows):
    callig = r["calligrapher"]
    script = r["script"]
    char = r["character"]
    slot = f"{callig}_{script}"
    target = train_dist.get(slot, train_dist["_GLOBAL_"])

    gt_path = ROOT / r["image_path"]
    gen_path = GEN_DIR / f"g{idx}.png"

    if not gt_path.exists() or not gen_path.exists():
        continue

    gt_im = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
    gen_im = np.array(Image.open(gen_path).convert("L"), dtype=np.float32) / 255.0

    gt_m = measure_stroke_properties(gt_im)
    gen_m = measure_stroke_properties(gen_im)

    records.append({
        "slot": slot,
        "calligrapher": callig,
        "script": script,
        "char": char,
        "target_p50": target["ref_p50"],
        "target_mean": target["ref_mean"],
        "gt_mean": gt_m["mean"],
        "gen_mean": gen_m["mean"],
        "gt_p50": gt_m["p50"],
        "gen_p50": gen_m["p50"],
        "gt_p10": gt_m["p10"],
        "gen_p10": gen_m["p10"],
        "gt_p90": gt_m["p90"],
        "gen_p90": gen_m["p90"],
        "gt_contrast": gt_m["contrast"],
        "gen_contrast": gen_m["contrast"],
    })

df = pd.DataFrame(records)
print(f"Total evaluated samples: {len(df)}")

slot_summary = []
for slot, g in df.groupby("slot"):
    slot_summary.append({
        "Slot": slot,
        "Count": len(g),
        "Target Mean": f"{g['target_mean'].mean():.2f}px",
        "GT Mean": f"{g['gt_mean'].mean():.2f}px",
        "Gen Mean": f"{g['gen_mean'].mean():.2f}px",
        "Target P50": f"{g['target_p50'].mean():.2f}px",
        "GT P50": f"{g['gt_p50'].mean():.2f}px",
        "Gen P50": f"{g['gen_p50'].mean():.2f}px",
        "GT Contrast": f"{g['gt_contrast'].mean():.2f}",
        "Gen Contrast": f"{g['gen_contrast'].mean():.2f}",
    })

print("\n" + "="*95)
print("             【Top 2 U100 Base 50k 物理笔画粗细对比 (全新 Authentic STD 训练)】")
print("="*95)
print(pd.DataFrame(slot_summary).to_string(index=False))
print("="*95)

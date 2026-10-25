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

CSV_PATH = ROOT / "exp-std/csv/eval_top5_fixed.csv"
GEN_DIR = ROOT / "assets/results/top5_u100_base_50k"
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
        "target_std": target["ref_std"],
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
    callig, script = slot.split("_")
    slot_summary.append({
        "书法家": callig,
        "书体": script,
        "样本数": len(g),
        "目标均值": f"{g['target_mean'].mean():.2f}px",
        "真迹均值": f"{g['gt_mean'].mean():.2f}px",
        "生成均值": f"{g['gen_mean'].mean():.2f}px",
        "目标P50": f"{g['target_p50'].mean():.2f}px",
        "真迹P50": f"{g['gt_p50'].mean():.2f}px",
        "生成P50": f"{g['gen_p50'].mean():.2f}px",
        "真迹提按比": f"{g['gt_contrast'].mean():.2f}",
        "生成提按比": f"{g['gen_contrast'].mean():.2f}",
    })

df_res = pd.DataFrame(slot_summary)
print("\n" + "="*95)
print("             【Top 5 U100 Base 50k 物理笔画粗细细分表 (全新 Authentic STD 训练)】")
print("="*95)
print(df_res.to_string(index=False))
print("="*95)

# Overall
tot_t_mean = df["target_mean"].mean()
tot_gt_mean = df["gt_mean"].mean()
tot_gen_mean = df["gen_mean"].mean()
tot_t_p50 = df["target_p50"].mean()
tot_gt_p50 = df["gt_p50"].mean()
tot_gen_p50 = df["gen_p50"].mean()

print(f"全体平均: 目标均值={tot_t_mean:.2f}px | 真迹均值={tot_gt_mean:.2f}px | 生成均值={tot_gen_mean:.2f}px")
print(f"全体典型P50: 目标P50={tot_t_p50:.2f}px | 真迹P50={tot_gt_p50:.2f}px | 生成P50={tot_gen_p50:.2f}px")
print(f"全体提按比: 真迹提按比={df['gt_contrast'].mean():.2f} | 生成提按比={df['gen_contrast'].mean():.2f}")
print("="*95)

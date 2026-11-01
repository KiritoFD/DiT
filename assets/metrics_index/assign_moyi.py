"""Recover MoYi panel identity by optimal one-to-one assignment of GT to the 187-row CSV."""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = "exp-std/csv/eval200_fixed.csv"
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
OUT = Path("assets/metrics_index/moyi12ch_protocol.csv")


def arr(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((64, 64))).astype(np.float32).ravel() / 255.0


rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
B = np.stack([arr(Path(r["image_path"])) for r in rows])
G = np.stack([arr(PANEL / f"gt{i}.png") for i in range(187)])
print("cost matrix", G.shape, B.shape)
cost = np.abs(G[:, None, :] - B[None, :, :]).mean(2)
ri, ci = linear_sum_assignment(cost)
pairs = list(zip(ri, ci))
mean_d = float(cost[ri, ci].mean())
worst = float(cost[ri, ci].max())
print("assignment mean dist", round(mean_d, 4), "worst", round(worst, 4))

with OUT.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["gt_index", "csv_index", "img_id", "character",
                                      "script", "calligrapher", "match_dist"],
                       lineterminator="\n")
    w.writeheader()
    for gi, ci_ in sorted(pairs):
        r = rows[ci_]
        w.writerow({"gt_index": gi, "csv_index": ci_, "img_id": r["img_id"],
                    "character": r["character"], "script": r["script"],
                    "calligrapher": r["calligrapher"],
                    "match_dist": round(float(cost[gi, ci_]), 4)})
print("wrote", OUT, "mean", round(mean_d, 4))

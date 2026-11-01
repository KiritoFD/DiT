"""Resolve MoYi panels: nearest-neighbour index mapping instead of exact order."""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = "exp-std/csv/eval200_fixed.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))


def arr(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((64, 64))).astype(np.float32).ravel() / 255.0


bank = np.stack([arr(r["image_path"]) for r in rows])

for d in ["assets/server_48_evals/moyi_12ch_eval200fix",
          "assets/style_eval_src/moyi_top10_rf",
          "assets/style_eval_src/moyi_top10_rf_4ch",
          "assets/server_48_evals/eval_ours200fix"]:
    d = Path(d)
    if not d.is_dir():
        print("MISSING", d)
        continue
    idx, ds = [], []
    for i in range(187):
        gp = d / f"gt{i}.png"
        if not gp.is_file():
            break
        dist = np.abs(bank - arr(gp)).mean(1)
        j = int(dist.argmin())
        idx.append(j)
        ds.append(float(dist[j]))
    uniq = len(set(idx))
    print(d.name, "scanned", len(idx), "unique_mapped", uniq,
          "mean_dist", round(float(np.mean(ds)), 4) if ds else None,
          "first10", idx[:10])

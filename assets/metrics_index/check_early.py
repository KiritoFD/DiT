"""Check remaining early-history panels for GT recoverability via the master index."""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
INDEX = "assets/data_clean/top10_20261021_v1/scan/inputs/train.csv"


def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32).ravel() / 255.0


rows = [r for r in csv.DictReader(open(INDEX, encoding="utf-8-sig")) if Path(r["image_path"]).is_file()]
B = np.stack([arr(Path(r["image_path"])) for r in rows])

# early-history dirs that only have generated images
cands = [
    "assets/results/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/20261009-142151-v72-callivae-kl1e6-s12500-frozen-sp-c2ot/eval_samples_ctrl/step0130000/eval200fix",
    "assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/eval_samples_ctrl/step0200000/eval200fix",
    "assets/aligned_eval_results/Base_55k_dist_aligned",
    "assets/style_eval_src/moyi_top10_rf_4ch",
]
for c in cands:
    d = Path(c)
    if not d.is_dir():
        print("MISSING", c)
        continue
    gts = sorted(d.glob("gt*.png"))
    gs = sorted(d.glob("g[0-9]*.png"))
    print(f"{d.name:52s} g={len(gs):3d} gt={len(gts):3d}", end="")
    if len(gts) >= 50:
        ds = []
        for i in range(min(20, len(gts))):
            v = arr(d / f"gt{i}.png")
            ds.append(float(np.abs(B - v).mean(1).min()))
        print(f"  gt-vs-index mean-dist={np.mean(ds):.4f}")
    else:
        print("  (no GT)")

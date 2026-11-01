"""Render GT for panels that lack it, then verify identity against the protocol CSV.

Rendering uses the protocol's own image_path, so the result is by construction the
protocol's GT; identity is still re-checked by nearest-neighbour after writing.
"""
import csv
import io
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = "exp-std/csv/eval200_fixed.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))

TARGETS = [
    "assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/eval_samples_ctrl/step0200000/eval200fix",
    "assets/results/v83A_55f_correct_eval_20261012/eval_samples_ctrl/eval200fix_input_g",
    "assets/results/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix",
    "assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/eval_samples_ctrl/step0010000/eval200fix",
    "assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/eval_samples_ctrl/step0015000/eval200fix",
    "assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/eval_samples_ctrl/step0020000/eval200fix",
    "assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/eval_samples_ctrl/step0030000/eval200fix",
    "assets/results/v83A_stdsoft_zerofix/20261012-213952-v83A-stdsoft-zerofix/eval_samples_ctrl/step0105000/eval200fix",
]


def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32).ravel() / 255.0


B = np.stack([arr(Path(r["image_path"])) for r in rows])

for t in TARGETS:
    d = Path(t)
    if not d.is_dir():
        print("MISSING", t)
        continue
    n_g = len([p for p in d.glob("g[0-9]*.png") if p.stem[1:].isdigit()])
    if n_g != 187:
        print("SKIP n_g", n_g, d.name)
        continue
    have_gt = len(list(d.glob("gt*.png")))
    if have_gt < 187:
        for i, r in enumerate(rows):
            dst = d / f"gt{i}.png"
            if not dst.exists():
                src = Path(r["image_path"])
                # save as RGB PNG to match panel convention
                with Image.open(src) as im:
                    im.convert("RGB").save(dst)
    # verify
    idx = []
    for i in range(187):
        v = arr(d / f"gt{i}.png")
        idx.append(int(np.abs(B - v).mean(1).argmin()))
    print(f"{d.parent.parent.name[:34]:34s} step={d.parent.name} exact={idx == list(range(187))}")

"""Render + verify GT for the remaining early-history panels."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = "exp-std/csv/eval200_fixed.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))


def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32).ravel() / 255.0


B = np.stack([arr(Path(r["image_path"])) for r in rows])

roots = ["assets/results", "assets/style_eval_src", "assets/aligned_eval_results",
         "assets/recent_poster_20261017/png"]
done = []
for root in roots:
    for d in Path(root).rglob("*"):
        if not d.is_dir():
            continue
        gs = [p for p in d.glob("g[0-9]*.png") if p.stem[1:].isdigit()]
        if len(gs) != 187:
            continue
        if len(list(d.glob("gt*.png"))) >= 187:
            continue
        for i, r in enumerate(rows):
            dst = d / f"gt{i}.png"
            if not dst.exists():
                with Image.open(r["image_path"]) as im:
                    im.convert("RGB").save(dst)
        idx = []
        for i in range(187):
            v = arr(d / f"gt{i}.png")
            idx.append(int(np.abs(B - v).mean(1).argmin()))
        ok = idx == list(range(187))
        done.append((str(d), ok))
        print(f"{'OK ' if ok else 'BAD'} {d}")
print("rendered", len(done), "all-exact", all(x[1] for x in done))

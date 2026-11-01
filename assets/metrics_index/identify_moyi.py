"""Recover MoYi panel identity by matching its saved GT against the 26k master index."""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
INDEX = "assets/data_clean/top10_20261021_v1/scan/inputs/train.csv"
OUT = Path("assets/metrics_index/moyi12ch_identity.csv")


def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32).ravel() / 255.0


rows = [r for r in csv.DictReader(open(INDEX, encoding="utf-8-sig")) if Path(r["image_path"]).is_file()]
print("index rows with files", len(rows))
B = np.stack([arr(Path(r["image_path"])) for r in rows])
print("bank", B.shape)

recs = []
dists = []
for i in range(187):
    gp = PANEL / f"gt{i}.png"
    if not gp.is_file():
        break
    v = arr(gp)
    d = np.abs(B - v).mean(1)
    j = int(d.argmin())
    ds = d[j]
    recs.append({"gt_index": i, "index_row": j, "dist": round(float(ds), 4),
                 "img_id": rows[j]["img_id"], "character": rows[j]["character"],
                 "script": rows[j]["script"], "calligrapher": rows[j]["calligrapher"]})
    dists.append(float(ds))

mean = float(np.mean(dists))
uniq = len({r["img_id"] for r in recs})
print("mean dist", round(mean, 4), "max", round(max(dists), 4), "unique ids", uniq, "/", len(recs))
print("sample", [(r["character"], r["calligrapher"], r["dist"]) for r in recs[:5]])

if mean < 0.02 and uniq == len(recs):
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
    print("WRITTEN", OUT)
else:
    print("NOT accepted: identity not proven")

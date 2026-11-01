"""Final alignment: optimal one-to-one assignment on binarised shapes.

If the assignment is near-identity, eval200_fixed writer labels transfer to MoYi.
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))
OUT = Path("assets/metrics_index/moyi12ch_protocol.csv")


def binfeat(p, size=64):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L")).astype(np.float32) / 255.0
    b = (a < np.percentile(a, 30)).astype(np.uint8) * 255
    im2 = Image.fromarray(b).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im2, dtype=np.float32).ravel() / 255.0


B = np.stack([binfeat(Path(r["image_path"])) for r in rows])
G = np.stack([binfeat(PANEL / f"gt{i}.png") for i in range(187)])
cost = np.abs(G[:, None, :] - B[None, :, :]).mean(2)
ri, ci = linear_sum_assignment(cost)
identity = int((ri == ci).sum())
print(f"assignment: rows {len(ri)}, identity matches {identity}/187")

# writer distribution check: does the mapped set cover all 10 writers sensibly?
from collections import Counter
w = Counter(rows[c]["calligrapher"] for c in ci)
print("writers after assignment:", dict(w))

if identity >= 160:
    with OUT.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["gt_index", "csv_row", "img_id", "character",
                                           "script", "calligrapher"], lineterminator="\n")
        wr.writeheader()
        for g_i, c_i in sorted(zip(ri, ci)):
            r = rows[c_i]
            wr.writerow({"gt_index": int(g_i), "csv_row": int(c_i), "img_id": r["img_id"],
                         "character": r["character"], "script": r["script"],
                         "calligrapher": r["calligrapher"]})
    print("WRITTEN", OUT)
else:
    print("rejected: assignment not close enough to identity")

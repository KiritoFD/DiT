"""Verify that a generated panel belongs to eval200_fixed by cross-checking its neighbours.

Some panels have no GT, but the same run's *other* steps do. If any sibling step in the
same run is identity-verified against eval200_fixed, the family protocol is fixed and the
GT for this panel can be rendered locally from that CSV.
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = "exp-std/csv/eval200_fixed.csv"
HIST = json.loads(Path("assets/metrics_index/historical_panels.json").read_text(encoding="utf-8"))


def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32).ravel() / 255.0


rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
B = np.stack([arr(Path(r["image_path"])) for r in rows])

verified = {r["dir"]: r for r in HIST if r.get("usable")}
print("identity-verified panels:", len(verified))

# Which runs already have a verified step?
runs = {}
for d in verified:
    run = Path(d)
    # walk up to the run dir (the one containing eval_samples_ctrl)
    for p in [run, *run.parents]:
        if (p / "eval_samples_ctrl").is_dir():
            runs.setdefault(str(p), []).append(d)
            break
print("runs with a verified step:", len(runs))
for k, v in runs.items():
    print(" ", k.replace("\\", "/"), "->", len(v), "verified steps")

# For unverified panels, check whether siblings exist
print("\nunverified panels and their run:")
for r in HIST:
    if r.get("usable"):
        continue
    d = Path(r["dir"])
    for p in [d, *d.parents]:
        if (p / "eval_samples_ctrl").is_dir():
            cin = str(p)
            mark = "IN-VERIFIED-RUN" if cin in runs else "-"
            print(f"  {r['dir'][:70]:70s} {mark}")
            break

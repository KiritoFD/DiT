"""Verify panel identity by nearest-image matching (not byte equality).

Panels were re-rendered/JPEG-ish so bytes differ, but the correct CSV row is a
clear nearest neighbour under mean-|L|. We accept a panel only if every saved GT
maps to a distinct CSV row in the canonical order.
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
OUT = Path("assets/metrics_index/historical_panels.json")


def arr(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((64, 64))).astype(np.float32) / 255.0


rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
bank = np.stack([arr(r["image_path"]).ravel() for r in rows])
print("bank", bank.shape)

panels = [l.strip().split("|") for l in Path("assets/metrics_index/panels_raw.txt").read_text().splitlines() if l.strip()]
out = []
for ng, ngt, d in panels:
    d = Path(d)
    rec = {"dir": str(d), "n_generated": int(ng), "n_gt": int(ngt)}
    if int(ngt) >= 50:
        n = min(int(ngt), len(rows))
        idx = []
        dists = []
        for i in range(n):
            gp = d / f"gt{i}.png"
            if not gp.is_file():
                break
            v = arr(gp).ravel()
            dist = np.abs(bank - v).mean(1)
            j = int(dist.argmin())
            idx.append(j)
            dists.append(float(dist[j]))
        exact = idx == list(range(n))
        rec.update(gt_scanned=len(idx), order_exact=exact,
                   mean_dist=round(float(np.mean(dists)), 3) if dists else None,
                   identity="eval200_fixed(187)" if exact and n == 187 else
                            ("same_canonical_order" if exact else "mismatch"))
    out.append(rec)

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
ok = [r for r in out if r.get("order_exact")]
print("panels", len(out), "order-exact", len(ok))
for r in ok:
    print("OK", r["dir"], "meandist", r["mean_dist"])

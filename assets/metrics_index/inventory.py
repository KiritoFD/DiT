"""Inventory every local historical panel and classify identity evidence."""
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


def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32).ravel() / 255.0


rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
B = np.stack([arr(Path(r["image_path"])) for r in rows])

panels = [l.strip().split("|") for l in
          Path("assets/metrics_index/panels_raw.txt").read_text().splitlines() if l.strip()]
out = []
for ng, ngt, d in panels:
    d = Path(d)
    rec = {"dir": str(d).replace("\\", "/"), "n_generated": int(ng), "n_gt": int(ngt)}
    if int(ngt) >= 50:
        n = min(int(ngt), 187)
        idx, ds, missing = [], [], 0
        for i in range(n):
            gp = d / f"gt{i}.png"
            if not gp.is_file():
                missing += 1
                continue
            dist = np.abs(B - arr(gp)).mean(1)
            j = int(dist.argmin())
            idx.append(j)
            ds.append(float(dist[j]))
        uniq = len(set(idx))
        exact = idx == list(range(len(idx)))
        rec.update(gt_scanned=len(idx), gt_missing=missing, unique_rows=uniq,
                   order_exact=exact,
                   mean_dist=round(float(np.mean(ds)), 4) if ds else None,
                   usable=bool(exact and uniq == len(idx)))
        rec["identity"] = ("eval200_fixed(187), exact order" if exact
                           else ("own protocol (not eval200_fixed)" if uniq < len(idx)
                                 else "permuted eval200_fixed"))
    else:
        rec["identity"] = "no GT saved; generated images only"
    out.append(rec)

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
usable = [r for r in out if r.get("usable")]
print("panels", len(out), "with GT", sum(1 for r in out if r["n_gt"] >= 50),
      "identity-exact", len(usable), "generated-only", sum(1 for r in out if r["n_gt"] < 50))
for r in usable:
    print("USABLE", r["dir"])

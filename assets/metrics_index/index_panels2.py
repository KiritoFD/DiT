"""Verify panel identity by grayscale-pixel hashing (panels store RGB copies of L data)."""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

CSV = "exp-std/csv/eval200_fixed.csv"
OUT = Path("assets/metrics_index/historical_panels.json")


def ghash(p):
    with Image.open(p) as im:
        return hashlib.sha256(im.convert("L").tobytes()).hexdigest()


rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
m = {}
for i, r in enumerate(rows):
    m.setdefault(ghash(r["image_path"]), []).append(i)
print("csv unique gray hashes", len(m), "of", len(rows))

panels = [l.strip().split("|") for l in Path("assets/metrics_index/panels_raw.txt").read_text().splitlines() if l.strip()]
out = []
for ng, ngt, d in panels:
    d = Path(d)
    rec = {"dir": str(d), "n_generated": int(ng), "n_gt": int(ngt)}
    if int(ngt) >= 50:
        order = []
        ok = True
        for i in range(int(ngt)):
            gp = d / f"gt{i}.png"
            if not gp.is_file():
                ok = False
                break
            h = ghash(gp)
            if h not in m:
                ok = False
                break
            order.append(m[h][0])
        rec["gt_gray_matched"] = ok
        if ok:
            rec["order_is_csv_order"] = order == list(range(len(order)))
            rec["order_matches_187"] = order == list(range(187))
            rec["identity"] = "eval200_fixed(187)" if order == list(range(187)) else "same_panel_reordered"
    out.append(rec)

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
good = [r for r in out if r.get("gt_gray_matched")]
print("panels", len(out), "gray-verified", len(good))
for r in good:
    print(r["identity"], "|", r["dir"])

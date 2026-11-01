"""Collect GAR / GARU panel metrics into one comparison table.

Usage:  python tools/d_20261101/gar_table.py [tag ...]
With no arguments it prints every GAR*/GARU* eval found under the eval root, ordered by
step, next to the D1b reference (whose generator is bit-identical to K).
"""
import json
import sys
from pathlib import Path

ROOT = Path("assets/d_20261101/eval")
REF = "D1b_2500"


def discover():
    tags = []
    if ROOT.is_dir():
        for d in sorted(ROOT.iterdir()):
            if not d.is_dir() or not (d / "eval.json").is_file():
                continue
            n = d.name
            if n.startswith(("GAR_s", "GARL_s", "GARU_s")):
                tags.append(n)
    return sorted(tags, key=lambda t: (t.split("_")[0], int(t.rsplit("_s", 1)[1])))


def row(tag):
    f = ROOT / tag / "eval.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    c, o, w, i = (d.get(k) or {} for k in ("content", "ocr", "writer", "ink"))
    return dict(tag=tag, step=d.get("step"), ssim=c.get("ssim"), lpips=c.get("lpips"),
                ink=c.get("ink_iou"), frag=c.get("frag"), ocr=o.get("top1"), n=o.get("n"),
                d=w.get("direction"), amp=w.get("amplitude"), w50=i.get("width_p50"),
                ir=i.get("ink_ratio"), hole=c.get("hole"))


def fm(v, p=6):
    return "-" if v is None else f"{v:.{p}f}"


tags = sys.argv[1:] or ([REF] + discover())
hdr = ["tag", "step", "ssim", "lpips", "ink_iou", "frag", "ocr", "n", "dir", "amp",
       "w50", "ink_ratio"]
print("".join(f"{h:>12}" for h in hdr))
base = None
for t in tags:
    r = row(t)
    if not r:
        print(f"{t:>12}  MISSING")
        continue
    vals = [r["tag"], r["step"], fm(r["ssim"]), fm(r["lpips"]), fm(r["ink"]), fm(r["frag"], 3),
            fm(r["ocr"], 2), r["n"] or 0, fm(r["d"], 4), fm(r["amp"], 4), fm(r["w50"], 3),
            fm(r["ir"], 4)]
    print("".join(f"{str(v):>12}" for v in vals))
    if t == REF:
        base = r

if base:
    print(f"\ndelta vs {REF} (generator == K exactly):")
    for t in tags:
        if t == REF:
            continue
        r = row(t)
        if not r or r["ssim"] is None:
            continue
        d = lambda a, b: f"{a - b:+.6f}" if (a is not None and b is not None) else "-"
        print(f"  {r['tag']:<14} dSSIM={d(r['ssim'], base['ssim'])}"
              f"  dLPIPS={d(r['lpips'], base['lpips'])}"
              f"  dInkIoU={d(r['ink'], base['ink'])}"
              f"  dOCR={d(r['ocr'], base['ocr'])}"
              f"  dW50={d(r['w50'], base['w50'])}")


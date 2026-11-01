"""FIG13 right panel: the decoder sweep across every axis we actually measured.

The report (docs/reports/2026-09_to_11_full_cycle_review.md, FIG13) asks for
"Delta SSIM / Delta LPIPS curves and the tolerance-skeleton F1 curve" for decoder steps
500/1000/1500/2000/2500. All of that is already in assets/metrics_index/lineage_summary.csv,
so this is pure local plotting -- no new inference, no cross-machine transfer.

It also adds the E-G (writer identity) curve that the original
fig_decoder_sweep.png omitted, because that is the axis where the sweep is NOT
monotonic, and leaving it out would let the figure imply "longer is better".

Everything is plotted as a DELTA against K, since the absolute values sit in a narrow
band and the question is the direction of travel.
"""
import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = Path("assets/metrics_index/lineage_summary.csv")
OUT = Path("assets/metrics_index/fig_decoder_fig13.png")

LABELS = [("K (A@43429)", 0), ("K + D1dec@500", 500), ("K + D1dec@1000", 1000),
          ("K + D1dec@1500", 1500), ("K + D1dec@2000", 2000), ("K + D1dec", 2500)]
# column -> (title, ylabel, higher_is_better)
AXES = [("ssim", "ΔSSIM ↑", "Δ SSIM vs K", True),
        ("lpips", "ΔLPIPS ↓", "Δ LPIPS vs K", False),
        ("iou", "ΔInk-IoU ↑", "Δ Ink-IoU vs K", True),
        ("skel", "Δtolerance-skeleton F1 ↑", "Δ F1 vs K", True),
        ("frag", "Δfragment ratio ↓", "Δ fragments vs K", False),
        ("eg", "ΔE-G pooled AUC ↑", "Δ writer AUC vs K", True)]


def load():
    rows = {r["label"]: r for r in
            csv.DictReader(open(CSV, encoding="utf-8-sig")) if r["panel"] == "val100"}
    out = []
    for label, dstep in LABELS:
        r = rows.get(label)
        if not r:
            print(f"  ! missing {label!r} in {CSV}")
            continue
        out.append(dict(label=label, dstep=dstep, raw=r,
                        ssim=float(r["ssim"]), lpips=float(r["lpips"]),
                        iou=float(r["iou"]), skel=float(r["skel"]),
                        frag=float(r["frag"]), cr=float(r["direction"]),
                        ocr=float(r["ocr"])))
    if len(out) < 3:
        raise SystemExit(f"need the decoder sweep in {CSV}; found {len(out)}")
    return out


def main():
    rows = load()
    # The E-G column is not in lineage_summary; join it from the paired CSV.
    eg = {}
    p = Path("assets/metrics_index/paired_eg_ocr_20261101.csv")
    if p.is_file():
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            eg[(r["arm"], int(r["step"]))] = float(r["eg_auc"])
    names = {"K (A@43429)": ("K_A", 43429), "K + D1dec@500": ("K_plus_D1dec_s0000500", 43429),
             "K + D1dec@1000": ("K_plus_D1dec_s0001000", 43429),
             "K + D1dec@1500": ("K_plus_D1dec_s0001500", 43429),
             "K + D1dec@2000": ("K_plus_D1dec_s0002000", 43429),
             "K + D1dec": ("K_plus_D1dec", 43429)}
    for r in rows:
        r["eg"] = eg.get(names[r["label"]])
    has_eg = all(r["eg"] is not None for r in rows)

    base = rows[0]
    xs = [r["dstep"] for r in rows]
    axes_def = [a for a in AXES if a[0] != "eg" or has_eg]

    n = len(axes_def)
    fig, axs = plt.subplots(1, n, figsize=(3.1 * n + 1.5, 5.0))
    if n == 1:
        axs = [axs]

    for ax, (key, title, ylab, higher) in zip(axs, axes_def):
        ys = [r[key] - base[key] for r in rows]
        ax.axhline(0, color="#2c3e50", lw=1.0, ls="-", alpha=.55)
        ax.plot(xs, ys, "-o", color="#c0392b", lw=1.8, ms=6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        # Mark the best step on this axis, which is NOT always the last one.
        best = (max if higher else min)(range(len(ys)), key=lambda i: ys[i])
        ax.scatter([xs[best]], [ys[best]], s=170, facecolors="none",
                   edgecolors="#27ae60", linewidths=2.0, zorder=6)
        ax.set_title(f"{title}\nbest @{xs[best]} (Δ={ys[best]:+.4g})", fontsize=10.5)
        ax.set_xlabel("decoder steps (generator = K)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=.22, ls=":")
        lo, hi = min(ys), max(ys)
        pad = max((hi - lo) * .35, 1e-9)
        ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle("FIG13 · K's generator fixed, only the decoder changes — every axis measured\n"
                 "quality axes (SSIM/LPIPS/Ink-IoU/skeleton) improve and saturate; "
                 "writer identity peaks at 1000 steps then fades",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}\n")

    hdr = ["label", "step", "dSSIM", "dLPIPS", "dInkIoU", "dSkelF1", "dFrag", "dEG", "dOCR"]
    print("".join(f"{h:>22}" for h in hdr))
    for r in rows:
        v = [r["label"], r["dstep"],
             f"{r['ssim'] - base['ssim']:+.5f}", f"{r['lpips'] - base['lpips']:+.5f}",
             f"{r['iou'] - base['iou']:+.5f}", f"{r['skel'] - base['skel']:+.5f}",
             f"{r['frag'] - base['frag']:+.4f}",
             "-" if r["eg"] is None else f"{r['eg'] - base['eg']:+.4f}",
             f"{r['ocr'] - base['ocr']:+.2f}"]
        print("".join(f"{str(x):>22}" for x in v))


if __name__ == "__main__":
    main()

"""SSIM x LPIPS, curated to a handful of self-explanatory points.

Each label says WHAT the model is (initialisation / architecture / training recipe),
not just a codename. Only the era-defining points are drawn in full colour; the rest
stay as grey context so the labelled ones never collide.
"""
import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = Path("assets/metrics_index/ssim_lpips_20261101.csv")
OUT = Path("assets/metrics_index/fig_ssim_lpips_final.png")

SKIP = {"v83A-diag", "std-input", "X-input-g"}

# (arm, step) -> label lines, dx, dy, ha
HEROES = {
    ("v80", 190000): ("v80 190k — best overall\n(CalliVAE + physics prior)", -30, -52, "right"),
    ("v72", 130000): ("v72 130k — CalliVAE first win\n(no REPA)", -64, 26, "right"),
    ("DiC", 80000): ("DiC 80k — paired-image\nalignment", 18, -54, "left"),
    ("U100-Base", 55000): ("U100 base 55k — non-top10 pretrain\n(no style table)", -66, 40, "right"),
    ("CFG1.6", 0): ("CFG 1.6 — best CFG sweep\n(style gain maxed)", -58, -40, "right"),
    ("v84A", 200000): ("v84A 200k — std-conditioned\n(no style token)", 46, 44, "left"),
    ("v88", 50000): ("v88 50k — 8-step sampler\n(quality tradeoff)", 46, -22, "left"),
    ("L-prior", 256000): ("L-prior 256k — std prior only\n(no character table)", 46, 22, "left"),
    ("style-scan", 20000): ("same recipe @20k — early checkpoint\n(same model, less training)", 46, -24, "left"),
    ("MoYi-4ch", 80000): ("MoYi 4ch 80k — 4-ch Rectified Flow\n(48 host; own GT, self-contained)", 40, -44, "left"),
    ("MoYi-12ch", 50000): ("MoYi 12ch 50k — 12-ch multichannel\n(early 48-host baseline)", 46, -26, "left"),
}


def load():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    best = {}
    for r in rows:
        if r["arm"] in SKIP:
            continue
        try:
            s, l, st = float(r["ssim"]), float(r["lpips"]), int(r["step"])
        except ValueError:
            continue
        k = (r["arm"], st)
        if k not in best or best[k]["ssim"] < s:
            best[k] = {"arm": r["arm"], "step": st, "ssim": s, "lpips": l}
    return list(best.values())


def main():
    rows = load()
    ax_rows = [r for r in rows
               if (r["arm"], r["step"]) not in HEROES]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter([r["ssim"] for r in ax_rows], [r["lpips"] for r in ax_rows],
               s=34, c="#9aa5b1", alpha=.55, edgecolors="white", linewidth=.4,
               label=f"other eras ({len(ax_rows)})")

    for (arm, step), (text, dx, dy, ha) in HEROES.items():
        cand = [r for r in rows if r["arm"] == arm and r["step"] == step]
        if not cand:
            continue
        r = cand[0]
        ax.scatter([r["ssim"]], [r["lpips"]], s=130, c="#c0392b", alpha=.95,
                   edgecolors="white", linewidth=1.0, zorder=9)
        ax.annotate(text, (r["ssim"], r["lpips"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.8, ha=ha, va="center",
                    linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", lw=.8, color="#333",
                                    shrinkA=2, shrinkB=5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb",
                              lw=.7, alpha=.97), zorder=12)

    ax.invert_yaxis()
    ax.set_xlabel("SSIM ↑  — pixel/structure agreement with the GT (187-char protocol)", fontsize=11)
    ax.set_ylabel("LPIPS ↓  — perceptual distance (AlexNet); axis inverted so up = better",
                  fontsize=11)
    ax.set_title("Quality vs fidelity across eras — what each stepping stone actually is\n"
                 f"{len(rows)} identity-verified panels re-measured under one protocol",
                 fontsize=13)
    ax.grid(alpha=.18, linestyle=":")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("total", len(rows), "heroes", len(HEROES), "context", len(ax_rows))


if __name__ == "__main__":
    main()

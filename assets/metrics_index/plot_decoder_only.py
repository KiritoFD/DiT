"""The decoder-only result: K's frozen generator + a decoder trained on it.

This is the headline of the D/GAR line, and it is easy to misread, so the figure is
explicit about the two things that make it credible:
  * every point shares the SAME generator (K), so any movement is pure decoder effect
  * the movement is tiny in absolute terms -- the axes are zoomed to show it

Protocol honesty: the decoder sweep was measured on the val100 panel (n=100 for
content, n=90 for OCR), whereas ssim_lpips_20261101.csv is the 187-char protocol
(n=187). The two are NOT comparable, so this figure plots only the metrics that
actually share a protocol, and names it on the axis.
"""
import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
EG = Path("assets/metrics_index/paired_eg_ocr_20261101.csv")
OUT = Path("assets/metrics_index/fig_decoder_only_result.png")

# The decoder-only sweep, in training order. K is step 0 (stock sd-vae decoder).
SWEEP = [("K_A", 43429, "K\n(stock sd-vae decoder)", 0),
         ("K_plus_D1dec_s0000500", 43429, "K + D1dec @500", 500),
         ("K_plus_D1dec_s0001000", 43429, "K + D1dec @1000", 1000),
         ("K_plus_D1dec_s0001500", 43429, "K + D1dec @1500", 1500),
         ("K_plus_D1dec_s0002000", 43429, "K + D1dec @2000", 2000),
         ("K_plus_D1dec", 43429, "K + D1dec @2500", 2500)]


def load(path):
    if not path.is_file():
        return {}
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        out[(r["arm"], int(r["step"]))] = r
    return out


def main():
    eg = load(EG)
    rows = []
    for arm, step, label, dstep in SWEEP:
        e = eg.get((arm, step))
        if not e:
            continue
        rows.append(dict(label=label, dstep=dstep,
                         auc=float(e["eg_auc"]), eer=float(e["eg_eer"]),
                         ocr=float(e["ocr_top1"]), n=int(e["n"])))
    if len(rows) < 2:
        raise SystemExit(f"need the decoder sweep in {EG}; found "
                         f"{[r['label'] for r in rows]}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    xs = [r["dstep"] for r in rows]
    base = rows[0]

    def panel(ax, key, title, ylab):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, "-o", color="#c0392b", lw=1.8, ms=6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        ax.scatter([xs[0]], [ys[0]], s=150, facecolors="none",
                   edgecolors="#2c3e50", linewidths=2.0, zorder=6)
        ax.annotate("K reference\n(zero training cost)", (xs[0], ys[0]),
                    xytext=(14, -34), textcoords="offset points",
                    fontsize=8.5, ha="left", linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=.7))
        lo, hi = min(ys), max(ys)
        pad = max((hi - lo) * 0.35, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        d = ys[-1] - ys[0]
        ax.set_title(f"{title}\nendpoint Δ = {d:+.4f}  ({d / abs(base[key]) * 100:+.2f}%)",
                     fontsize=11)
        ax.set_xlabel("decoder training steps — generator is K at every point")
        ax.set_ylabel(ylab)
        ax.grid(alpha=.2, linestyle=":")
        # Mark the saturation point: ~90% of the gain arrives by 1500.
        if key == "auc":
            peak = max(range(len(ys)), key=lambda i: ys[i])
            ax.axvline(xs[peak], color="#7f8c8d", ls="--", lw=.9, alpha=.7)
            ax.annotate(f"peak @{xs[peak]}", (xs[peak], ys[peak]),
                        xytext=(6, 12), textcoords="offset points", fontsize=8.2,
                        color="#2c3e50")

    panel(axes[0], "auc", "Writer identity — E-G pooled AUC ↑", "pooled AUC")
    panel(axes[1], "eer", "Writer identity — E-G EER ↓", "EER (%)")

    fig.suptitle("Decoder-only post-training on a frozen generator\n"
                 "generator is K, bit-identical at every point → any movement is caused "
                 "ONLY by the decoder (val100 panel, one protocol)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")
    for r in rows:
        print(f"  {r['label']:<26} n={r['n']:<4} eg={r['auc']:.4f} eer={r['eer']:.4f} "
              f"ocr={r['ocr']:.2f}")


if __name__ == "__main__":
    main()


"""Scatter: pooled E-G AUC x panel OCR, curated labels only.

Labels are placed with explicit offsets and thin leader lines so the handful of
representative points stay readable instead of piling up at the top right.
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "assets/metrics_index"
CSV = OUT / "paired_eg_ocr_20261101.csv"

# Hand-picked representatives: (arm, step) -> (display label, dx, dy, ha)
LABELS = {
    ("heldout", 50000, "s4"): ("S4@50k val100\n87.98->85.01 AUC / 77.8 OCR", 0, 0, "left"),
}

# Simple, explicit label table keyed by (arm substring, step) to avoid protocol hashes.
TABLE = [
    # key_arm, step, text, dx, dy  (dx<0 puts text left of the point)
    ("S4", 50000, "S4@50k", 16, 26),
    ("S1", 50000, "S1@50k", -16, 40),
    ("S0", 50000, "S0@50k", 14, -30),
    ("K_A", 43429, "K", -18, -40),
    ("D1b_2500", 2500, "D1b@2.5k", -20, 40),
    ("D1_2500", 2500, "D1@2.5k bad init", 18, -24),
    ("D3_11000", 11000, "D3@11k", -18, 34),
    ("P0", 50000, "P0@50k", 18, -22),
]


def load():
    rows = [r for r in csv.DictReader(CSV.open(encoding="utf-8-sig"))]
    out = []
    for r in rows:
        n = int(r["n"])
        if n < 50:
            continue
        arm = r["arm"]
        if arm in ("eval", "shared_train_panel") or arm.startswith(
                ("smoke", "null", "other", "zero", "out_", "eval_cpu_mid")):
            continue
        # write the S-panel arms from their source panel path
        src = r["source_panel"]
        for tag in ("/S4/", "/S1/", "/S0/", "/S6/", "/S8/", "/S9/"):
            if tag in src:
                arm = tag.strip("/") + "@50k"
                break
        if "/P0/" in src:
            arm = "P0"
        out.append({"arm": arm, "step": int(r["step"]), "n": n,
                    "auc": float(r["eg_auc"]), "ocr": float(r["ocr_top1"])})
    return out


def main():
    rows = load()
    fig, ax = plt.subplots(figsize=(13, 8.2))
    colors = {100: "#c0392b", 187: "#2e6fb5", 200: "#7d3c98"}
    for n, c in colors.items():
        z = [r for r in rows if r["n"] == n]
        if not z:
            continue
        ax.scatter([r["auc"] for r in z], [r["ocr"] for r in z], s=58, c=c,
                   alpha=.62, edgecolors="white", linewidth=.5,
                   label=f"panel n={n} ({len(z)})")
    z = [r for r in rows if r["n"] not in colors]
    if z:
        ax.scatter([r["auc"] for r in z], [r["ocr"] for r in z], s=26, c="#8899a6",
                   alpha=.55, label=f"other panels ({len(z)})")

    used = set()
    # All labels go into clear space: left side for mid-cluster, above/below for the
    # top-right cluster, so no two leader lines cross.
    for key, step, text, dx, dy in TABLE:
        cand = [r for r in rows if r["arm"] == key and r["step"] == step
                or (r["arm"].startswith(key) and r["step"] == step)]
        if not cand:
            continue
        r = cand[0]
        if (r["arm"], r["step"]) in used:
            continue
        used.add((r["arm"], r["step"]))
        ha = "right" if dx < 0 else "left"
        ax.annotate(text, (r["auc"], r["ocr"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10,
                    ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", lw=.8, color="#333",
                                    shrinkA=0, shrinkB=6,
                                    connectionstyle="arc3,rad=0"),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=.6, alpha=.95),
                    zorder=10)
        ax.scatter([r["auc"]], [r["ocr"]], s=150, facecolors="none",
                   edgecolors="#111", linewidths=1.3, zorder=9)

    ax.set_xlabel("E-G pooled AUC (%) — official domain evaluator, same reference bank", fontsize=11)
    ax.set_ylabel("Panel OCR top-1 (%) — same panel, same recognizer", fontsize=11)
    ax.set_title("Writer identity (E-G) vs content legibility (OCR)\n"
                 f"{len(rows)} verified panels, unified re-measurement", fontsize=13)
    ax.grid(alpha=.2, linestyle=":")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pooled_eg_ocr_labelled.png", dpi=200)
    print("plotted", len(rows), "labelled", len(used))


if __name__ == "__main__":
    main()

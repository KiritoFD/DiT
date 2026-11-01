"""Final scatter: pooled E-G AUC x panel OCR, all eras labelled.

Labels are placed with explicit offsets; only representative/historic points get a
label so the plot stays readable while covering every era from v72 to 2026-11.
"""
import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = Path("assets/metrics_index/paired_eg_ocr_20261101.csv")
OUT = Path("assets/metrics_index/fig_pooled_eg_ocr_final.png")

SKIP = ("smoke", "null", "other", "zero", "out_", "eval_cpu_mid", "eval",
        "X-input-g", "std-input", "L-prior-256",
        # MoYi panels: the saved GT is not the eval200_fixed set and the original
        # protocol CSV is gone, so their identity cannot be proven -> excluded.
        "MoYi")

# (arm, step) -> text, dx, dy, ha, va
LABELS = {
    # --- 2026-11 main line (100-char val100) ---
    ("K_A", 43429): ("K", -58, -44, "right", "center"),
    ("D1b_2500", 2500): ("D1b 2.5k = K", -40, 46, "right", "center"),
    ("D1_2500", 2500): ("D1 2.5k bad init", 30, -16, "left", "center"),
    ("D3_11000", 11000): ("D3 11k", 10, 46, "left", "center"),
    ("P0", 50000): ("P0@50k", 28, -20, "left", "center"),
    # --- S family 50k ---
    ("S4", 50000): ("S4@50k", 46, 44, "left", "center"),
    ("S1", 50000): ("S1@50k", 24, -34, "left", "center"),
    ("S0", 50000): ("S0@50k", -44, -40, "right", "center"),
    ("S6", 50000): ("S6@50k", -52, 22, "right", "center"),
    # --- aligned eval / U100 ---
    ("U100-Base", 55000): ("U100 base 55k", -96, 44, "right", "center"),
    ("Aligned-Arm2", 50000): ("Arm2 50k", -110, 16, "right", "center"),
    ("DiC", 50000): ("DiC 50k", 12, 54, "left", "center"),
    ("ExpA", 70000): ("ExpA 70k", 96, 44, "left", "center"),
    # --- V72..V80 CalliVAE era ---
    ("v80", 200000): ("v80 200k", 96, 20, "left", "center"),
    ("v79", 75000): ("v79 75k", -50, -50, "right", "center"),
    ("v74", 115000): ("v74 115k", -76, -46, "right", "center"),
    ("v72", 130000): ("v72 130k", -46, 44, "right", "center"),
    ("v78", 45000): ("v78 45k", 40, -22, "left", "center"),
    # --- V83..V88 ---
    ("v84A", 200000): ("v84A 200k", -90, -44, "right", "center"),
    ("v85A", 10000): ("v85A 10k", -66, -30, "right", "center"),
    ("v85E", 15000): ("v85E 15k", 40, -50, "left", "center"),
    ("v88", 50000): ("v88 50k", 34, 14, "left", "center"),
    ("v88", 5000): ("v88 5k", 34, -16, "left", "center"),
    # --- CFG sweep ---
    ("CFG1.6", 0): ("CFG1.6 max E-G", 92, 24, "left", "center"),
    ("CFG1.4", 0): ("CFG1.4", 92, -6, "left", "center"),
    ("CFG1.2", 0): ("CFG1.2", 92, -36, "left", "center"),
}


def load():
    rows = []
    for r in csv.DictReader(CSV.open(encoding="utf-8-sig")):
        n = int(r["n"])
        if n < 50:
            continue
        arm = r["arm"]
        if arm.startswith(SKIP):
            continue
        src = r["source_panel"]
        for tag in ("/S4/", "/S1/", "/S0/", "/S6/", "/S8/", "/S9/"):
            if tag in src:
                arm = tag.strip("/")
        if "/P0/" in src:
            arm = "P0"
        rows.append({"arm": arm, "step": int(r["step"]), "n": n,
                     "auc": float(r["eg_auc"]), "ocr": float(r["ocr_top1"])})
    return rows


def main():
    rows = load()
    fig, ax = plt.subplots(figsize=(14, 9))
    colors = {100: "#c0392b", 187: "#2e6fb5", 200: "#7d3c98"}
    for n, c in colors.items():
        z = [r for r in rows if r["n"] == n]
        if z:
            ax.scatter([r["auc"] for r in z], [r["ocr"] for r in z], s=52, c=c,
                       alpha=.5, edgecolors="white", linewidth=.5,
                       label=f"{n}-char panel ({len(z)})")
    z = [r for r in rows if r["n"] not in colors]
    if z:
        ax.scatter([r["auc"] for r in z], [r["ocr"] for r in z], s=24, c="#8899a6",
                   alpha=.5, label=f"other panels ({len(z)})")

    used = set()
    for (arm, step), (text, dx, dy, ha, va) in LABELS.items():
        cand = [r for r in rows if r["arm"] == arm and r["step"] == step]
        if not cand:
            continue
        r = max(cand, key=lambda x: x["auc"])
        if (r["arm"], r["step"]) in used:
            continue
        used.add((r["arm"], r["step"]))
        ax.annotate(text, (r["auc"], r["ocr"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9, ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", lw=.7, color="#333",
                                    shrinkA=1, shrinkB=5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc",
                              lw=.6, alpha=.95), zorder=12)
        ax.scatter([r["auc"]], [r["ocr"]], s=140, facecolors="none",
                   edgecolors="#111", linewidths=1.2, zorder=11)

    ax.set_xlabel("E-G pooled AUC (%) — writer identity (official domain evaluator)", fontsize=11)
    ax.set_ylabel("Panel OCR top-1 (%) — content legibility (same recognizer)", fontsize=11)
    ax.set_title("All eras: style identity vs OCR\n"
                 f"{len(rows)} identity-verified panels (v72 → 2026-11), unified re-measurement",
                 fontsize=13)
    ax.grid(alpha=.18, linestyle=":")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("points", len(rows), "labelled", len(used), "->", OUT)


if __name__ == "__main__":
    main()

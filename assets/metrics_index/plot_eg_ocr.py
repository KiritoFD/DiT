"""E-G x OCR with curated, self-explanatory labels.

Each label states what the model is (initialisation / recipe / failure mode), not a
bare codename. Only the decisive points are coloured; everything else is grey context.
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
        "X-input-g", "std-input")

# (arm, step) -> label, dx, dy, ha
HEROES = {
    ("K_A", 43429): ("K = A@43k — current baseline\n(frozen DiT, style token)", -60, -46, "right"),
    ("D1b_2500", 2500): ("D1b 2.5k — decoder-only finetune\n(generator bit-identical to K)", -50, 46, "right"),
    ("D1_2500", 2500): ("D1 2.5k — WRONG init\n(random style branch, never from K)", 34, -20, "left"),
    ("D3_11000", 11000): ("D3 11k — hard-char FM\n(same wrong init)", 24, 46, "left"),
    ("P0", 50000): ("P0 50k — no top10 pretrain\n(content-only baseline)", 34, -18, "left"),
    ("S4", 50000): ("S4 50k — Stroke x Spatial\n(matrix winner on 100 chars)", 50, 34, "left"),
    ("S0", 50000): ("S0 50k — content baseline\n(no style structure)", -54, -36, "right"),
    ("v80", 190000): ("v80 190k — CalliVAE + physics\n(only model high on BOTH axes)", -60, 44, "right"),
    ("CFG1.6", 0): ("CFG 1.6 — max guidance\n(peak E-G, OCR punished)", 60, -22, "left"),
    ("v88", 5000): ("v88 5k — cold start\n(nothing learned yet)", 34, -18, "left"),
    ("MoYi-12ch", 50000): ("MoYi 12ch 50k — 12-ch multichannel\n(early 48-host baseline)", -56, -30, "right"),
    ("L-prior", 256000): ("L-prior 256k — std prior only\n(no character table)", 46, 30, "left"),
    ("L-prior", 190000): ("L-prior 190k — std prior\n(no character table)", -56, 26, "right"),
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
        rows.append({"arm": arm, "step": int(r["step"]),
                     "auc": float(r["eg_auc"]), "ocr": float(r["ocr_top1"])})
    return rows


def main():
    rows = load()
    others = [r for r in rows if (r["arm"], r["step"]) not in HEROES]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter([r["auc"] for r in others], [r["ocr"] for r in others],
               s=30, c="#9aa5b1", alpha=.5, edgecolors="white", linewidth=.4,
               label=f"other panels ({len(others)})")

    for (arm, step), (text, dx, dy, ha) in HEROES.items():
        cand = [r for r in rows if r["arm"] == arm and r["step"] == step]
        if not cand:
            continue
        r = max(cand, key=lambda x: x["auc"])
        ax.scatter([r["auc"]], [r["ocr"]], s=125, c="#c0392b", alpha=.95,
                   edgecolors="white", linewidth=1.0, zorder=9)
        ax.annotate(text, (r["auc"], r["ocr"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.8, ha=ha, va="center",
                    linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", lw=.8, color="#333",
                                    shrinkA=2, shrinkB=5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb",
                              lw=.7, alpha=.97), zorder=12)

    ax.set_xlabel("E-G pooled AUC ↑ — writer identity (official domain evaluator)", fontsize=11)
    ax.set_ylabel("Panel OCR top-1 ↑ — content legibility (same recogniser)", fontsize=11)
    ax.set_title("Style identity vs content legibility — what each milestone actually is\n"
                 f"{len(rows)} identity-verified panels, one protocol",
                 fontsize=13)
    ax.grid(alpha=.18, linestyle=":")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("total", len(rows), "heroes", len(HEROES), "context", len(others))


if __name__ == "__main__":
    main()

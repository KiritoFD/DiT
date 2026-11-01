"""Build the paired pooled-E-G-AUC x panel-OCR scatter from verified jobs.

Sources of truth:
  pooled AUC  -- assets/eval_domain_pooled_*/<name>/result.json written by
                 tools.eval_unpaired.domain.run (official statistics.verification)
  panel OCR   -- assets/metrics_index/ocr_*.json written by ocr_for_panels
Only rows present in BOTH are plotted; nothing is imputed.
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
JOBS = ROOT / "assets/metrics_index/verified_jobs.json"
D1B_JOBS = ROOT / "assets/metrics_index/d1b_job.json"
OCR_FILES = [ROOT / "assets/metrics_index/ocr_verified_20261101.json",
             ROOT / "assets/metrics_index/ocr_d1b_20261101.json"]
POOLED_ROOTS = [ROOT / "assets/eval_domain_pooled_verified",
                ROOT / "assets/eval_domain_pooled_d1b"]
OUT = ROOT / "assets/metrics_index"
# Diagnostic/sanity panels that are not model evaluation points.
EXCLUDE_ARMS = ("smoke", "null", "other", "zero", "out_", "eval_cpu_mid")
MIN_N_PLOT = 50
IMPORTANT = {"heldout", "D1_2500", "D2_3000", "D3_11000", "D3_11000_plus_D1dec",
             "D1b_2500", "K_A", "K_plus_D1dec", "K_plus_D1dec_s0000500",
             "K_plus_D1dec_s0001000", "K_plus_D1dec_s0001500", "K_plus_D1dec_s0002000",
             "K_P3A", "P0", "P1", "P2", "P3A", "P3B", "S0", "S1", "S4", "S6", "S8", "S9"}
COLORS = {100: "#c0392b", 187: "#2878b5", 200: "#7d3c98", 1046: "#117a65"}


def pooled_auc(name):
    for root in POOLED_ROOTS:
        p = root / name / "result.json"
        if p.is_file():
            d = json.loads(p.read_text())
            v = d.get("g", {}).get("E-G", {}).get("verification", {}).get("pooled", {})
            if v.get("auc") is not None:
                return v
    return None


def main():
    jobs = json.loads(JOBS.read_text())
    for p in [D1B_JOBS]:
        if p.is_file():
            jobs += json.loads(p.read_text())
    ocr = {}
    for f in OCR_FILES:
        if f.is_file():
            ocr.update(json.loads(f.read_text()))
    rows = []
    for j in jobs:
        if j["arm"] == "eval" or j["arm"] == "shared_train_panel":
            continue  # renamed eval_export sets / training panels: identity not eval-grade
        key = f"{j['arm']}|{j['protocol']}"
        ov = ocr.get(key)
        if not ov or "error" in ov or ov.get("top1") is None:
            continue
        v = pooled_auc(f"{j['arm']}_{j['step']}_{j['protocol']}")
        if not v:
            continue
        rows.append({"arm": j["arm"], "step": j["step"], "protocol": j["protocol"],
                     "n": j["n"], "eg_auc": round(100 * v["auc"], 4),
                     "eg_eer": round(100 * v.get("eer", 0), 4),
                     "ocr_top1": ov["top1"], "ocr_n": ov.get("n_ocr"),
                     "source_panel": j["samples"], "source_csv": j["csv"]})
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "step", "protocol", "n", "eg_auc", "eg_eer", "ocr_top1", "ocr_n",
              "source_panel", "source_csv"]
    with (OUT / "paired_eg_ocr_20261101.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["n"], -r["eg_auc"])))
    plot_rows = [r for r in rows if r["n"] >= MIN_N_PLOT
                 and not any(r["arm"].startswith(s) for s in EXCLUDE_ARMS)]
    fig, ax = plt.subplots(figsize=(14.5, 9))
    for n in sorted({r["n"] for r in plot_rows}):
        z = [r for r in plot_rows if r["n"] == n]
        ax.scatter([r["eg_auc"] for r in z], [r["ocr_top1"] for r in z],
                   s=30 if n > 300 else 55, alpha=.7, c=COLORS.get(n, "#68737d"),
                   label=f"panel n={n}", edgecolors="white", linewidth=.35)
    for r in plot_rows:
        if r["arm"] in IMPORTANT and r["n"] <= 200:
            tag = r["arm"] + (f"@{r['step']//1000}k" if r["step"] >= 1000 else f"@{r['step']}")
            ax.annotate(tag, (r["eg_auc"], r["ocr_top1"]), xytext=(5, 5),
                        textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("E-G pooled AUC (%) -- official domain evaluator, same reference bank")
    ax.set_ylabel("Panel OCR top-1 (%) -- same verified panel, Duguang OCR")
    ax.set_title(f"Unified re-measurement: E-G (writer identity) vs OCR (content)\n"
                 f"paired points={len(plot_rows)} / verified jobs={len(jobs)} "
                 "(diagnostic arms and n<50 panels excluded from plot, kept in CSV)")
    ax.grid(alpha=.22, linestyle=":")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pooled_eg_ocr_20261101.png", dpi=180)
    plt.close(fig)
    md = ["# 统一重测：E-G × OCR 散点（2026-11-01）\n\n",
          f"- 配对完成 **{len(plot_rows)}** 点（绘图口径，n≥{MIN_N_PLOT}）；全量 {len(rows)} 点见 CSV。\n",
          "- E-G=官方 pooled ROC AUC（`tools/eval_unpaired/domain/statistics.py::verification`，同参考库、同书体、排除同字、K64）；已用 S4 复现发布值 84.91 验证口径。\n",
          "- OCR=同一面板、同一识别器（`tools/f_20261101/eval_arm.py::ocr_and_ink`）；OCR n 为面板 OCR 可评字数（val100 通常 90）。\n",
          "- 不同 n 的面板不混排：val100(100)/187 字/test200/1046 字历史面板分色展示；诊断臂（smoke/out_* 等）与身份不达标的 eval_export 面板已剔除。\n\n",
          "## 全量配对点（按 n 分组，AUC 降序）\n\n",
          "| 模型/臂 | step | n | E-G AUC | EER | OCR top1 | OCR n |\n|---|---:|---:|---:|---:|---:|---:|\n"]
    for r in sorted(rows, key=lambda x: (x["n"], -x["eg_auc"])):
        md.append(f"| {r['arm']} | {r['step']} | {r['n']} | {r['eg_auc']:.2f} | "
                  f"{r['eg_eer']:.2f} | {r['ocr_top1']:.2f} | {r['ocr_n'] or '—'} |\n")
    md += ["\n## 边界\n\n",
           "- 旧 45 面板表是 retrieval/AFC 口径，不作数；本表全部为 pooled AUC 重测值。\n",
           "- style_matrix 的 S5/J 系历史面板走 discover() 协议，未包含在本批 batch 清单中，待下一批核验后追加。\n",
           "- 缺 E-G 或 OCR 的面板保留缺省，不用旧值或 0 填充。\n"]
    (ROOT / "docs/1031/PAIRED_EG_OCR.md").write_text("".join(md), encoding="utf-8")
    print(f"paired_plot={len(plot_rows)} paired_all={len(rows)} jobs={len(jobs)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读 all_runs_20261005.csv -> 输出 markdown 分析表 (条件对照 / 协议分组)。"""
import csv
import os
from collections import defaultdict

os.chdir("/root/Workspace/xy/DiT")
SRC = "docs/experiments/all_runs_20261005.csv"
OUT = "_sync_work/tables_20261005.md"

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def proto(r):
    p = str(r.get("eval_protocol") or "").strip()
    if p and not p.startswith("unknown"):
        return p
    b = fnum(r.get("best_ssim"))
    if b is None:
        return "no-eval"
    return "OLD(内嵌 json 口径, ssim>=0.65)" if b >= 0.65 else "unknown"


def cond(r):
    skel = "skel:std" if r.get("skel_kind") == "std" else (
        f"skel:{r.get('skel_kind')}" if r.get("skel_cond_on") in ("True", True) else "skel:-")
    ch = ("charTAB" if r.get("char_table") in ("True", True) else "char-") + \
         ("" if r.get("char_cond_on") in ("True", True) else "(off)")
    cal = "callig" if r.get("callig_used") in ("True", True) else "no-callig"
    gj = f"inj{r.get('glyph_inject')}" if str(r.get("glyph_inject", "0")) not in ("0", "", "0.0") else ""
    return "|".join(x for x in (skel, ch, cal, gj) if x)


groups = defaultdict(list)
for r in rows:
    if not r.get("n_eval") or r["n_eval"] == "0":
        continue
    groups[(proto(r), cond(r))].append(r)

lines = ["# 自动分析表 (2026-10-05)", "",
         "> 源: `docs/experiments/all_runs_20261005.csv` (499 run, 71 条有评测)",
         "", "## A. 按协议 × 条件分组 (每组列前 5)", ""]
for key in sorted(groups, key=lambda k: (-len(groups[k]), k[0])):
    pt, cd = key
    rs = sorted(groups[key], key=lambda r: -fnum(r["best_ssim"]))
    lines.append(f"### [{pt}] {cd}  ({len(rs)} runs)")
    lines.append("")
    lines.append("| 实验 | best ssim | @step | n_eval | batch | max_steps | run |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rs[:5]:
        lines.append(f"| {r['experiment_name'][:44]} | {fnum(r['best_ssim']):.4f} | "
                     f"{int(r['best_step'])//1000 if r['best_step'] else 0}k | {r['n_eval']} | "
                     f"{r['global_batch_size']} | {r['max_steps']} | `{r['run_dir'][:52]}` |")
    lines.append("")

# B. 关键配对 (同协议、同代、单变量)
PAIRS = [
    ("字表开关 (skel 条件下)", ["v10a-skel-cond-pretrain", "v10b-skel-only-pretrain"]),
    ("骨架开关 (表条件下)", ["v53-triple-tables-noskel", "v54-minimal-tables-S-noskel"]),
    ("骨架 vs 表 (同协议)", ["v50-A-space-xattn", "v52-noskel-callig-char", "v53-triple", "v54-minimal"]),
]
lines += ["## B. 关键单变量配对", ""]
for title, pats in PAIRS:
    lines.append(f"**{title}**")
    lines.append("")
    lines.append("| 实验 | 协议 | 条件 | best | @step | 曲线关键点 |")
    lines.append("|---|---|---|---|---|---|")
    for pat in pats:
        hits = [r for r in rows if pat.lower() in (r["experiment_name"] or "").lower()
                and r.get("n_eval") and r["n_eval"] != "0"]
        if not hits:
            lines.append(f"| (无匹配 {pat}) | | | | | |")
            continue
        r = max(hits, key=lambda x: fnum(x["best_ssim"]) or 0)
        pts = " ".join(f"{int(k)//1000}k:{fnum(r.get('ssim_at_%d' % k)):.3f}"
                       for k in (5000, 20000, 50000, 65000, 100000)
                       if fnum(r.get(f"ssim_at_{k}")) is not None)
        lines.append(f"| {r['experiment_name'][:40]} | {proto(r)} | {cond(r)} | "
                     f"{fnum(r['best_ssim']):.4f} | "
                     f"{int(r['best_step'])//1000 if r['best_step'] else 0}k | {pts} |")
    lines.append("")

# C. 全量有评测清单
lines += ["## C. 全部有评测的 run (按协议/best 排序)", "",
          "| # | 实验 | 协议 | 条件 | best | @step | n |",
          "|---|---|---|---|---|---|---|"]
allr = sorted([r for r in rows if r.get("n_eval") and r["n_eval"] != "0"],
              key=lambda r: (proto(r), -(fnum(r["best_ssim"]) or 0)))
for i, r in enumerate(allr, 1):
    lines.append(f"| {i} | {r['experiment_name'][:46]} | {proto(r)} | {cond(r)} | "
                 f"{fnum(r['best_ssim']):.4f} | "
                 f"{int(r['best_step'])//1000 if r['best_step'] else 0}k | {r['n_eval']} |")

open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"-> {OUT}  ({len(allr)} 个有评测 run, {len(groups)} 个条件组)")
for k in sorted(groups, key=lambda k: -len(groups[k]))[:8]:
    print(f"   [{k[0]}] {k[1]} : {len(groups[k])} runs")

#!/usr/bin/env python3
"""从 _archive/ 的 log.txt 提取实验记录（配置 + 最终指标），输出 CSV/MD 存档。

用途：删 _archive/ 之前先把性能记录落盘，避免彻底丢失历史结论。
输出：
  _sync_work/archive_record.csv  —— 机器可读
  _sync_work/archive_record.md   —— 人可读
"""
import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path("/root/Workspace/xy/DiT/assets/results/_archive")
OUT_DIR = Path("/root/Workspace/xy/DiT/_sync_work")

# 感兴趣的配置行（正则 -> 字段名）
CONF_PATTERNS = [
    ("model",        r"Building 2-Cond model: (\S+)"),
    ("fusion",       r"fusion=(\w+)"),
    ("dims",         r"dims=(\S+?)[,\s]"),
    ("params",       r"Trainable Parameters: ([\d,]+)"),
    ("lr",           r"\[optim\] AdamW lr=(\S+?) wd=(\S+)"),
    ("repa",         r"Teacher: (\w+).*?w=(\S+?)[,)]"),
    ("struct_loss",  r"\[latent-structure\] enabled: canny=(\S+?), skeleton=(\S+)"),
    ("ema",          r"\[EMA\] enabled with decay=(\S+)"),
    ("imgcount",     r"Dataset contains ([\d,]+) images"),
    ("earlystop",    r"\[early-stop\] metric=(\w+), patience=(\d+)"),
]

# 训练曲线里的指标
STEP_RE = re.compile(r"\(step=(\d+)\)")
METRIC_RES = {
    "diff":  re.compile(r"Diff:\s*([\d.]+)"),
    "repa":  re.compile(r"REPA\(w=[\d.]+\):\s*([\d.]+)"),
    "skel":  re.compile(r"skel_loss\(w=[\d.]+\):\s*([\d.]+)"),
    "canny": re.compile(r"canny_loss\(w=[\d.]+\):\s*([\d.]+)"),
    "steps_sec": re.compile(r"Steps/Sec:\s*([\d.]+)"),
}
# eval 行（如果有）
EVAL_RE = re.compile(
    r"(?:eval|\[eval\]).*?(ssim[=:]\s*([\d.]+))?.*?(lpips[=:]\s*([\d.]+))?",
    re.I)


def extract(log_path: Path):
    rec = {"path": str(log_path.parent.relative_to(ROOT)), "n_steps": 0}
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception as e:
        rec["error"] = str(e)
        return rec

    for name, pat in CONF_PATTERNS:
        m = re.search(pat, txt)
        if m:
            rec[name] = " ".join(g for g in m.groups() if g)

    # 训练曲线：首/末指标
    steps = []
    last = {}
    first = {}
    for m in STEP_RE.finditer(txt):
        steps.append(int(m.group(1)))
    for key, pat in METRIC_RES.items():
        vals = pat.findall(txt)
        if vals:
            try:
                first[key] = float(vals[0])
                last[key] = float(vals[-1])
            except ValueError:
                pass
    rec["n_steps"] = len(steps)
    if steps:
        rec["step_start"] = steps[0]
        rec["step_end"] = steps[-1]
    for k, v in first.items():
        rec[f"{k}_first"] = v
    for k, v in last.items():
        rec[f"{k}_last"] = v

    # eval：抓 (step, ssim) 曲线，取最佳
    evals = re.findall(r"eval step (\d+): ssim=([\d.]+)", txt)
    if evals:
        best = max(evals, key=lambda x: float(x[1]))
        rec["eval_n"] = len(evals)
        rec["ssim_best"] = float(best[1])
        rec["ssim_best_step"] = int(best[0])
        rec["ssim_last"] = float(evals[-1][1])
    rec["size_mb"] = round(
        sum(f.stat().st_size for f in log_path.parent.rglob("*") if f.is_file()
            ) / 1024 / 1024, 1)
    return rec


def main():
    logs = sorted(ROOT.rglob("log.txt"))
    if not logs:
        print("未找到 log.txt", file=sys.stderr)
        return 1
    recs = [extract(p) for p in logs]

    # 字段顺序
    base = ["path", "model", "fusion", "dims", "params", "lr", "repa",
            "struct_loss", "ema", "imgcount", "earlystop",
            "step_start", "step_end", "n_steps",
            "diff_first", "diff_last", "skel_first", "skel_last",
            "canny_first", "canny_last", "steps_sec_first", "steps_sec_last",
            "eval_n", "ssim_best", "ssim_best_step", "ssim_last",
            "size_mb", "error"]
    keys = base + sorted({k for r in recs for k in r} - set(base))

    csv_path = OUT_DIR / "archive_record.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    print(f"CSV -> {csv_path}  ({len(recs)} 条)")

    md_path = OUT_DIR / "archive_record.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# _archive/ 实验记录（删除前存档）\n\n")
        f.write(f"共 {len(recs)} 个实验，来自 `assets/results/_archive/`。\n\n")
        f.write("> 该目录已于 2026-09-23 删除 ckpt；本表保留其配置与最终指标。\n\n")
        f.write("| 实验 | 模型 | fusion | 参数量 | LR | 步数 | Diff首→末 | ssim最佳(steps) | Steps/s |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in sorted(recs, key=lambda x: -(x.get("ssim_best") or 0)):
            name = r["path"].split("/")[-1][:46]
            d1, d2 = r.get("diff_first", "-"), r.get("diff_last", "-")
            d = f"{d1}→{d2}" if d1 != "-" else "-"
            sp = r.get("steps_sec_last", "-")
            sb = r.get("ssim_best")
            sbest = f"{sb:.4f} @{r.get('ssim_best_step','-')}" if sb else "-"
            f.write("| {} | {} | {} | {} | {} | {}→{} | {} | {} | {} |\n".format(
                name,
                r.get("model", "-"),
                r.get("fusion", "-"),
                r.get("params", "-"),
                r.get("lr", "-"),
                r.get("step_start", "-"), r.get("step_end", "-"),
                d, sbest, sp))
    print(f"MD  -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

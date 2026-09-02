# -*- coding: utf-8 -*-
"""
_scan_results.py — 把散落在 225 个日志 + 50+ 实验目录里的评测结果汇成 CSV。

产出（写到 5script/）
  eval_points.csv   每个评测点一行（实验 × step × arm × 指标）
  experiments.csv   每个实验一行（best/last 指标 + 关键配置）

关联方式
--------
历史日志都在 /tmp/*.log，而结果目录在 5script/results/<series>/<run>。
这里扫描每个日志全文，统计其中出现的 5script/results/... 路径，取出现次数最多的
作为该日志所属实验（日志里既有 "results: <dir>"，也有大量
"wrote pending marker: <dir>/checkpoints/..." 行，足以可靠关联）。
"""
import os, sys, re, csv, json, glob, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.getcwd()
RESULTS = "5script/results"
OUT_DIR = "5script"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RUNPATH = re.compile(r"(5script/results/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+)")

# (正则, 字段名元组, source, 默认arm) —— 按特异性从高到低匹配
PATTERNS = [
    (re.compile(r"\[eval-metrics\] step (\d+): MSE=([\d.eE+-]+) SSIM=([\d.eE+-]+) "
                r"SkelIoU=([\d.eE+-]+) LPIPS=([\d.eE+-]+)"),
     ("step", "mse", "ssim", "skel_iou", "lpips"), "eval_metrics", ""),
    (re.compile(r"\[ctrl-metrics\] step (\d+): (base|ctrl)\s+MSE=([\d.eE+-]+) "
                r"SSIM=([\d.eE+-]+) SkelIoU=([\d.eE+-]+)"),
     ("step", "arm", "mse", "ssim", "skel_iou"), "ctrl_metrics", None),
    (re.compile(r"\[early-stop\] eval step (\d+): combo ssim=([\d.eE+-]+) "
                r"skel_iou=([\d.eE+-]+)"),
     ("step", "ssim", "skel_iou"), "early_stop", "combo"),
    (re.compile(r"\[early-stop\] eval step (\d+): ssim_lpips ssim=([\d.eE+-]+), "
                r"lpips=([\d.eE+-]+)"),
     ("step", "ssim", "lpips"), "early_stop", ""),
    (re.compile(r"\[early-stop\] eval step (\d+): ssim=([\d.eE+-]+)"),
     ("step", "ssim"), "early_stop", ""),
    (re.compile(r"\[auto-eval\] step (\d+): free-sampling MSE=([\d.eE+-]+) "
                r"SSIM=([\d.eE+-]+)"),
     ("step", "mse", "ssim"), "auto_eval", "free"),
]

NUMFIELDS = {"step", "mse", "ssim", "skel_iou", "lpips"}


def clean(line):
    return ANSI.sub("", line)


def parse_log(path):
    """返回 (points, runpath_counter)"""
    points = []
    counter = collections.Counter()
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if len(line) > 500:
                    continue
                for m in RUNPATH.finditer(line):
                    counter[m.group(1)] += 1
                if "eval" not in line and "auto-eval" not in line:
                    continue
                s = clean(line)
                if not re.search(r"\[(eval-metrics|ctrl-metrics|early-stop|auto-eval)\]", s):
                    continue
                for pat, fields, source, defarm in PATTERNS:
                    mm = pat.search(s)
                    if not mm:
                        continue
                    d = {k: v for k, v in zip(fields, mm.groups())}
                    rec = {"source": source, "log": os.path.basename(path)}
                    for k, v in d.items():
                        if k in NUMFIELDS:
                            try:
                                rec[k] = float(v)
                            except ValueError:
                                rec[k] = None
                        else:
                            rec[k] = v
                    rec["arm"] = d.get("arm") or (defarm or "")
                    rec["step"] = int(rec["step"])
                    # new best 标记
                    rec["is_best"] = 1 if re.search(r"new best|NEW BEST", s) else 0
                    points.append(rec)
                    break
    except Exception as e:
        print(f"# skip {path}: {e}", file=sys.stderr)
    return points, counter


def main():
    logs = sorted(set(glob.glob("/tmp/*.log")) |
                  set(glob.glob(f"{RESULTS}/*/*/log.txt")) |
                  set(glob.glob(f"{RESULTS}/*/log.txt")))
    print(f"# logs: {len(logs)}")

    # 日志 -> 实验目录
    log2run = {}
    for lp in logs:
        pts, cnt = parse_log(lp)
        if cnt:
            log2run[lp] = cnt.most_common(1)[0][0]
        # 二次解析（已在 parse_log 拿到 points，这里复用需要重读；改为缓存）
    # 重扫一次，缓存 points（避免重复 IO 的逻辑复杂，直接二次解析）
    all_points = []
    for lp in logs:
        pts, _ = parse_log(lp)
        run = log2run.get(lp, "")
        for p in pts:
            p["run_dir"] = run
            all_points.append(p)
    print(f"# raw eval points: {len(all_points)}")

    # 去重：同一 (run_dir, step, source, arm) 保留最后一条
    dedup = {}
    for p in all_points:
        key = (p["run_dir"], p["step"], p["source"], p["arm"])
        dedup[key] = p
    points = sorted(dedup.values(),
                    key=lambda x: (x["run_dir"], x["step"], x["source"], x["arm"]))
    print(f"# deduped eval points: {len(points)}")

    # ── 写 eval_points.csv ────────────────────────────────────────────────
    cols = ["run_dir", "series", "experiment", "step", "source", "arm",
            "ssim", "mse", "lpips", "skel_iou", "is_best", "log"]
    for p in points:
        parts = p["run_dir"].split("/")
        p["series"] = parts[2] if len(parts) > 2 else ""
        p["experiment"] = parts[3] if len(parts) > 3 else ""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/eval_points.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(points)
    print(f"# wrote {OUT_DIR}/eval_points.csv ({len(points)} rows)")

    # ── 聚合 experiments.csv ──────────────────────────────────────────────
    by_run = collections.defaultdict(list)
    for p in points:
        by_run[p["run_dir"]].append(p)

    exp_rows = []
    for run, pts in sorted(by_run.items()):
        if not run:
            continue
        rows = []
        best_ssim, best_step = -1, None
        best_lpips, best_skel, best_mse = None, None, None
        last = max(pts, key=lambda x: x["step"])
        for p in pts:
            if p.get("ssim") and p["ssim"] > best_ssim:
                best_ssim, best_step = p["ssim"], p["step"]
            if p.get("lpips") and (best_lpips is None or p["lpips"] < best_lpips):
                best_lpips = p["lpips"]
            if p.get("skel_iou") and (best_skel is None or p["skel_iou"] > best_skel):
                best_skel = p["skel_iou"]
            if p.get("mse") and (best_mse is None or p["mse"] < best_mse):
                best_mse = p["mse"]
        cfg = {}
        cpath = os.path.join(ROOT, run, "resolved_config.json")
        if os.path.isfile(cpath):
            try:
                cfg = json.load(open(cpath, encoding="utf-8"))
            except Exception:
                pass
        parts = run.split("/")
        exp_rows.append({
            "run_dir": run,
            "series": parts[2] if len(parts) > 2 else "",
            "experiment": parts[3] if len(parts) > 3 else "",
            "model": cfg.get("model", ""),
            "task": ("controlnet" if "ctrl" in run else "pretrain"),
            "data_csv": cfg.get("data_csv") or cfg.get("csv", ""),
            "diffusion_type": cfg.get("diffusion_type", ""),
            "image_size": cfg.get("image_size", ""),
            "batch_size": cfg.get("global_batch_size") or cfg.get("batch_size", ""),
            "lr": cfg.get("lr", ""),
            "max_steps": cfg.get("max_steps", ""),
            "last_step": last["step"],
            "last_ssim": last.get("ssim"),
            "best_step": best_step,
            "best_ssim": round(best_ssim, 4) if best_ssim >= 0 else None,
            "best_lpips": best_lpips,
            "best_mse": best_mse,
            "best_skel_iou": best_skel,
            "n_eval_points": len(pts),
            "sources": "|".join(sorted({p["source"] for p in pts})),
            "arms": "|".join(sorted({p["arm"] for p in pts if p["arm"]})),
        })

    with open(f"{OUT_DIR}/experiments.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exp_rows[0].keys()))
        w.writeheader()
        w.writerows(exp_rows)
    print(f"# wrote {OUT_DIR}/experiments.csv ({len(exp_rows)} rows)")

    # 控制台预览：按 best_ssim 排序
    print("\n# top experiments by best_ssim:")
    for r in sorted(exp_rows, key=lambda x: -(x["best_ssim"] or 0))[:15]:
        print(f"  {r['series']:<26} best_ssim={r['best_ssim']} @step{r['best_step']} "
              f"(last {r['last_step']}, n={r['n_eval_points']})")


if __name__ == "__main__":
    main()

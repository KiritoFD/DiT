# -*- coding: utf-8 -*-
"""
_enrich_experiments.py — 给 experiments.csv 补上「数据集难度」维度。

为什么需要
----------
不同实验用的数据集差异极大，SSIM **跨数据集不可比**：
  - s6_top6_diffonly  best_ssim=0.7322  → 训练集仅 11 位书家
  - s21_fame_flow_v2  best_ssim=0.467   → 训练集 44 位书家、4,765 字
把 0.7322 和 0.467 放在同一列排序会严重误导（看起来像"退步"，实为数据变难）。

本脚本：
  1. 扫描 configs/*.json，建立 series → (data_csv, eval_csv, 关键超参) 映射
     （很多早期实验的 resolved_config.json 未落盘，只能从源 config 反推）
  2. 用 data_assets.csv 把 data_csv 换成 书家数/字数/样本数
  3. 输出 experiments_enriched.csv，增加 data_cal / data_char / data_n /
     eval_set / difficulty_tier 列

difficulty_tier 定义（按训练集书家数）
  small   cal<=12       （top6 / top11 类，最易）
  medium  13<=cal<=50   （fame 44、mid* 等）
  large   cal>50        （top30+ / 全量）
"""
import os, sys, csv, json, glob, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    # ── 1. series -> config 映射 ──────────────────────────────────────────
    series2cfg = {}
    for p in sorted(glob.glob("src/train/configs/*.json")):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(j, dict):
            continue          # 少数 config 是数组（如 sweep 列表），跳过
        # results_dir 形如 5script/results/<series>
        rd = j.get("results_dir", "")
        if not rd:
            continue
        series = rd.rstrip("/").split("/")[-1]
        rec = series2cfg.setdefault(series, {})
        rec.update({
            "config_file": os.path.basename(p),
            "data_csv": j.get("data_csv") or j.get("csv", ""),
            "eval_csv": j.get("eval_csv") or j.get("gpu_eval_csv", ""),
            "model": j.get("model", ""),
            "diffusion_type": j.get("diffusion_type", ""),
            "latent_channels": j.get("latent_channels", ""),
            "batch": j.get("global_batch_size") or j.get("batch_size", ""),
            "lr": j.get("lr", ""),
            "max_steps": j.get("max_steps", ""),
            "image_size": j.get("image_size", ""),
            "task": "controlnet" if ("main_ckpt" in j or "skel_root" in j) else "pretrain",
            "cond_note": "; ".join(
                f"{k}={j[k]}" for k in
                ("w_skel", "w_canny", "w_skel_head", "w_glyph_cond", "w_repa",
                 "use_skel", "use_canny") if k in j),
        })
        # 同名 series 多 config 时，保留信息更全的
        if len(json.dumps(rec, ensure_ascii=False)) < len(json.dumps(j, ensure_ascii=False)):
            rec.update({k: v for k, v in j.items() if k not in rec or not rec[k]})

    print(f"# series with config: {len(series2cfg)}")

    # ── 2. data_csv -> 规模 ───────────────────────────────────────────────
    assets = {}
    ap = "5script/data_assets.csv"
    if os.path.isfile(ap):
        for r in load_csv(ap):
            if r.get("category") == "dataset_csv":
                assets[r["path"]] = {
                    "n": r.get("n_items"), "cal": r.get("n_calligraphers"),
                    "char": r.get("n_characters"), "script": r.get("n_scripts"),
                }
    print(f"# dataset csvs known: {len(assets)}")

    def lookup(csv_path):
        if not csv_path:
            return {}
        base = os.path.basename(csv_path)
        if csv_path in assets:
            return assets[csv_path]
        for k, v in assets.items():
            if os.path.basename(k) == base:
                return v
        # 名字片段匹配（历史 csv 命名不一）
        for k, v in assets.items():
            if base.split(".")[0] in k or k.split("/")[-1].split(".")[0] in base:
                return v
        return {}

    def tier(cal):
        try:
            c = int(cal)
        except (TypeError, ValueError):
            return "unknown"
        if c <= 12:
            return "small"
        if c <= 50:
            return "medium"
        return "large"

    # ── 3. 合并 ───────────────────────────────────────────────────────────
    exps = load_csv("5script/experiments.csv")
    out = []
    for e in exps:
        series = e.get("series", "")
        cfg = series2cfg.get(series, {})
        data_csv = e.get("data_csv") or cfg.get("data_csv", "")
        eval_csv = cfg.get("eval_csv", "")
        st = lookup(data_csv)
        row = dict(e)
        row.update({
            "config_file": cfg.get("config_file", ""),
            "data_csv_resolved": data_csv,
            "eval_csv": eval_csv,
            "task": cfg.get("task") or e.get("task", ""),
            "data_cal": st.get("cal", ""),
            "data_char": st.get("char", ""),
            "data_n": st.get("n", ""),
            "data_script": st.get("script", ""),
            "difficulty_tier": tier(st.get("cal")),
            "model_cfg": cfg.get("model") or e.get("model", ""),
            "diffusion_cfg": cfg.get("diffusion_type") or e.get("diffusion_type", ""),
            "image_size_cfg": cfg.get("image_size", ""),
            "batch_cfg": cfg.get("batch") or e.get("batch_size", ""),
            "lr_cfg": cfg.get("lr") or e.get("lr", ""),
            "max_steps_cfg": cfg.get("max_steps") or e.get("max_steps", ""),
            "cond_note": cfg.get("cond_note", ""),
        })
        out.append(row)

    cols = ["series", "experiment", "run_dir", "task", "difficulty_tier",
            "data_cal", "data_char", "data_n", "data_script",
            "best_ssim", "best_step", "best_lpips", "best_mse", "best_skel_iou",
            "last_step", "last_ssim", "n_eval_points",
            "data_csv_resolved", "eval_csv",
            "model_cfg", "diffusion_cfg", "image_size_cfg",
            "batch_cfg", "lr_cfg", "max_steps_cfg", "cond_note",
            "config_file", "sources", "arms"]
    with open("5script/experiments_enriched.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"# wrote 5script/experiments_enriched.csv ({len(out)} rows)")

    print("\n# 按 difficulty_tier 分组的最佳 SSIM：")
    groups = collections.defaultdict(list)
    for r in out:
        try:
            v = float(r["best_ssim"])
        except (TypeError, ValueError):
            continue
        groups[(r["difficulty_tier"], r["task"])].append((v, r["series"], r["best_step"]))
    for k in sorted(groups):
        vals = sorted(groups[k], reverse=True)
        print(f"\n  [{k[0]}/{k[1]}] n={len(vals)}")
        for v, s, st in vals[:6]:
            print(f"    {v:.4f}  {s:<30} @{st}")


if __name__ == "__main__":
    main()

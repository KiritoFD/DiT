#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量实验 CSV 构建 (2026-10-05)
扫描 exp-std/ , assets/results/ , _archive/ 下**所有** run 目录, 输出:
  docs/experiments/all_runs_20261005.csv        每 run 一行 (条件构成 + 曲线关键点 + best)
  docs/experiments/all_runs_curves_20261005.csv 长表 (run_id, step, ssim) 供画图
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

os.chdir("/root/Workspace/xy/DiT")
ROOTS = ["exp-std", "assets/results", "_archive"]
OUT = "docs/experiments"

COND_KEYS = [
    "model", "experiment_name", "global_batch_size", "max_steps", "diffusion_type",
    "skel_latent_shards_dir", "skel_as_glyph_cond", "eval_skel_latent_shards_dir",
    "inst_skel_shards_dir", "w_latent_skel", "latent_skel_probe",
    "no_char_cond", "char_dino_embeddings", "char_dino_index", "num_characters",
    "use_script_cond", "glyph_inject_layers", "w_glyph_cond",
    "callig_clean_map", "callig_map_json", "callig_script_map", "callig_aug_dirs",
    "freeze_char_table", "repa_layers", "w_repa", "repa_cache_dir",
    "self_cond", "noise_aug", "style_rank_w", "deform_skel", "skel_head",
    "image_channels", "aux_channels", "in_mem_eval_sets", "eval_csv", "eval_sets",
    "gpu_eval_sets", "eval_strict_csv", "train_csv", "hidden_size", "depth",
    "patch_size", "lr", "lr_main", "ema_decay",
]


def find_runs():
    out = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            if "resolved_config.json" in fn or "checkpoints" in dn:
                out.append(dp)
    return sorted(set(out))


def load_cfg(d):
    for cur in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        p = os.path.join(cur, "resolved_config.json")
        if os.path.isfile(p):
            try:
                return json.load(open(p, encoding="utf-8")), p
            except Exception:                                    # noqa: BLE001
                return None, None
    return None, None


def collect_curve(d):
    """{step: (ssim, mse)} — eval_auto json 优先, summary csv 补齐。"""
    c = {}
    for sub in (os.path.join(d, "checkpoints"), d):
        for j in glob.glob(os.path.join(sub, "eval_auto_*.json")):
            try:
                o = json.load(open(j, encoding="utf-8"))
                if o.get("ssim") is not None:
                    c[int(o.get("step", 0))] = (float(o["ssim"]), o.get("mse"))
            except Exception:                                    # noqa: BLE001
                pass
    for f in glob.glob(os.path.join(d, "eval_stdskel_summary.csv")):
        try:
            for r in csv.DictReader(open(f, encoding="utf-8")):
                if (r.get("set") or "").strip() != "eval200fix":
                    continue
                c.setdefault(int(r["step"]), (float(r["ssim_mean"]), None))
        except Exception:                                        # noqa: BLE001
            pass
    return c


def ckpt_steps(d):
    s = []
    for sub in (os.path.join(d, "checkpoints"), d):
        for f in glob.glob(os.path.join(sub, "*.pt")):
            m = re.search(r"(\d+)\.pt$", os.path.basename(f))
            if m:
                s.append(int(m.group(1)))
    return sorted(set(s))


def bname(v):
    v = str(v or "").strip()
    return os.path.basename(v.rstrip("/")) if v else ""


def main():
    runs = find_runs()
    os.makedirs(OUT, exist_ok=True)
    rows, curves = [], []
    for d in runs:
        cfg, cfg_path = load_cfg(d)
        curve = collect_curve(d)
        cks = ckpt_steps(d)
        c = cfg or {}
        exp_dir = os.path.dirname(d) if os.path.basename(d)[:2].isdigit() else d
        rid = d.replace("/", "__")
        bs = max(curve, key=lambda k: curve[k][0]) if curve else None
        row = {
            "run_dir": d,
            "exp_dir": exp_dir,
            "run_id": rid,
            "experiment_name": c.get("experiment_name") or os.path.basename(exp_dir),
            "cfg_path": cfg_path or "",
            "cfg_found": bool(cfg),
            "ckpt_n": len(cks),
            "ckpt_last": cks[-1] if cks else "",
            "n_eval": len(curve),
            "eval_first": min(curve) if curve else "",
            "eval_last": max(curve) if curve else "",
            "best_ssim": round(curve[bs][0], 6) if bs else "",
            "best_step": bs if bs else "",
            "last_ssim": round(curve[max(curve)][0], 6) if curve else "",
        }
        for k in (5000, 10000, 20000, 35000, 50000, 65000, 80000, 100000, 150000):
            row[f"ssim_at_{k}"] = round(curve[k][0], 6) if k in curve else ""
        # 条件列 —— ★ 保留原值 (多值字段如 "eval200fix:a.csv:187,seen:b.csv:20" 不能 basename)
        for k in COND_KEYS:
            row[k] = c.get(k, "")
        row["skel_dir_base"] = bname(c.get("skel_latent_shards_dir"))
        row["inst_skel_dir_base"] = bname(c.get("inst_skel_shards_dir"))
        # 评测协议 (决定数值可比性)
        _sets = " ".join(str(c.get(k, "") or "") for k in
                         ("in_mem_eval_sets", "eval_csv", "eval_sets", "gpu_eval_sets"))
        if "eval200_fixed" in _sets:
            row["eval_protocol"] = "eval200fix+seen"
        elif "seen_v10" in _sets:
            row["eval_protocol"] = "eval_seen_v10"
        elif "fame_strict" in _sets:
            row["eval_protocol"] = "fame_strict_clean_v8"
        elif _sets.strip():
            row["eval_protocol"] = "other:" + _sets[:40]
        else:
            row["eval_protocol"] = "unknown(老 run, 见 eval_auto json)"
        # 派生条件标签
        sd = str(c.get("skel_latent_shards_dir", "") or "")
        row["skel_cond_on"] = bool(sd) or str(c.get("skel_as_glyph_cond", "")).lower() in ("true", "1")
        row["skel_kind"] = ("std" if "std" in sd else ("inst/gt" if ("skel_w7" in sd or "gt" in sd)
                                                       else ("other" if sd else "-")))
        row["char_table"] = bool(c.get("char_dino_embeddings"))
        row["char_cond_on"] = not (c.get("no_char_cond") in (True, "true", "True"))
        row["callig_used"] = any(str(c.get(k, "") or "") for k in
                                 ("callig_clean_map", "callig_map_json", "callig_script_map"))
        row["glyph_inject"] = c.get("glyph_inject_layers", 0) or 0
        rows.append(row)
        for st in sorted(curve):
            curves.append({"run_id": rid, "run_dir": d,
                           "experiment_name": row["experiment_name"],
                           "step": st, "ssim": round(curve[st][0], 6),
                           "mse": curve[st][1] if curve[st][1] is not None else ""})

    cols = list(rows[0].keys())
    with open(f"{OUT}/all_runs_20261005.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["run_dir"]))

    ccols = ["run_id", "run_dir", "experiment_name", "step", "ssim", "mse"]
    with open(f"{OUT}/all_runs_curves_20261005.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ccols)
        w.writeheader()
        w.writerows(curves)

    n_eval = sum(1 for r in rows if r["n_eval"])
    print(f"run 目录: {len(rows)} (有评测曲线: {n_eval}); 曲线点: {len(curves)}")
    print(f"-> {OUT}/all_runs_20261005.csv")
    print(f"-> {OUT}/all_runs_curves_20261005.csv")


if __name__ == "__main__":
    main()

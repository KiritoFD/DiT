# -*- coding: utf-8 -*-
"""
cleanup_old_experiments.py — 旧实验整理 + ckpt 抽稀清理 (约 1/10 保留).

对每个实验 run 目录:
  1. 读 resolved_config.json → 配置摘要
  2. 读 eval 历史 (eval_auto_*.json / log.txt early-stop 行) → 最佳指标与 best ckpt
  3. 保留集: best-2 ckpt + 每 ~10 个取 1 个阶梯点 (含首末) → 其余 .pt/.pt.done 删除
  4. 产出快照 json + 全局 CSV (--apply 才真正删除)

保护: ctrl_skel / s20_ctrl_skel_flow_v2 系列 (在跑), 2h 内有写入的 run,
      被活动训练引用的 ckpt。

用法:
  /opt/conda/bin/python tools/cleanup_old_experiments.py            # dry-run
  /opt/conda/bin/python tools/cleanup_old_experiments.py --apply    # 执行删除
"""
import os
import sys
import csv
import json
import glob
import re
import argparse
import datetime

ROOT = "/root/Workspace/xy/DiT"
OUT_DIR = os.path.join(ROOT, "5script", "results", "_cleanup_20260829")
CSV_OUT = os.path.join(ROOT, "5script", "old_experiments_summary_20260829.csv")
SEARCH_ROOTS = [
    os.path.join(ROOT, "5script", "results"),
    os.path.join(ROOT, "results"),
]
PROTECT_SERIES = {"ctrl_skel", "s20_ctrl_skel_flow_v2"}  # 在跑/用户在用
PROTECT_CKPTS = {  # 被活动训练引用的 ckpt (绝对路径)
    os.path.join(ROOT, "5script/results/s20_midcommon_s_flow_v2/"
                 "20260829-023132-s20-midcommon-s-flow-v2/checkpoints/0102500.pt"),
}
FRESH_HOURS = 2.0

DESC = {
    "s2_fromscratch_2factor": "早期: 二因子从零训练",
    "s5_2factor_top30": "二因子 top30 系列 (基线)",
    "s5_2factor_B_canny05_pixelsk": "二因子 + canny0.5 + pixel-skel head",
    "s5_2factor_B_latentstruct": "二因子 + latent 结构损失",
    "s5_2factor_B_latentstruct_pixelsk_opt": "latent 结构 + pixel-skel 调优",
    "s5_2factor_B_pixelfp32": "二因子 pixel fp32",
    "s6_top6_diffonly": "s6: top6 纯扩散 (diff-only 基线)",
    "s6_top6_struct_fp32": "s6: top6 + 结构损失 fp32",
    "s6_top6_struct_fp32_full": "s6: 结构损失 full 变体",
    "s7_klf4_top30": "s7: klf4 清洗 top30",
    "s7_ramp_b8all": "s7: batch8-all ramp 实验",
    "s8_klf4_clean_dino": "s8: klf4 清洗 + DINO char embedding",
    "s9_skelonly": "s9: 仅 skel 损失",
    "s10_b4_grey_clear": "s10: b4 灰度清洗",
    "s11_top6_p4": "s11: top6 patch4",
    "s12_3top30_dino": "s12: 3top30 + DINO (ddpm)",
    "s13_3top30_dino_xs": "s13: 3top30 DiT-XS (ddpm, 短跑放弃)",
    "s14_xs_ddpm_orig": "s14: XS ddpm 原始数据对照",
    "s15_ws_flow": "s15: WS/2 宽体 + flow (195k)",
    "s17_s_flow": "s17: S/2 + flow, 3top30 (165k)",
    "s18_s_flow_small": "s18: S/2 + flow, top6 小数据 (43k)",
    "s19_midclean_s_flow": "s19: mid-clean 增广 + flow (旧架构末代)",
    "s20_midcommon_s_flow_v2": "s20: 新架构 rms/swiglu/qk_norm/RoPE + mid-common, 当前最优基模 (0.5294)",
    "v3a": "v3a: 早期二因子 XL 雏形",
    "v3a_xl": "v3a XL",
    "v3a_xl_highdim": "v3a XL 高维条件融合",
    "v3a_xl_skelhead": "v3a XL + skel head",
    "v3b_xl_glyphcond": "v3b XL + 标准字形条件",
    "v3c_xl_glyphcond_midstep": "v3c 字形条件 + midstep",
    "compositional": "组合泛化实验",
    "px_s_scratch": "pixel 空间从零实验",
    "exp_s_scratch": "S 从零 (pixel 时代)",
    "exp_xl_head": "XL + skel head",
    "exp_xl_head_r8": "XL skel-head 秩 8 消融",
    "exp_xl_head_r32": "XL skel-head 秩 32 消融",
    "exp_xl_head_r32_attn": "XL skel-head 秩 32 + attn",
    "exp_xl_head_r64": "XL skel-head 秩 64 消融",
    "dit_s_pretrain": "DiT-S 预训练 (pixel 时代)",
    "overfit_500": "500 张过拟合冒烟",
}


def series_desc(series):
    if series in DESC:
        return DESC[series]
    if series.startswith("exp_xl_head"):
        return DESC["exp_xl_head_r8"]
    return ""


def load_eval_history(run_dir, ckpt_dir):
    """返回 [(step, metric_dict)] 排序, 以及主指标名 (ssim 优先, 无则 mse)."""
    hist = {}
    for p in glob.glob(os.path.join(ckpt_dir, "eval_auto_*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
            hist[int(d.get("step", 0))] = d
        except Exception:
            pass
    logp = os.path.join(run_dir, "log.txt")
    if os.path.exists(logp):
        try:
            txt = open(logp, encoding="utf-8", errors="ignore").read()
            for m in re.finditer(r"eval step (\d+): combo ssim=([0-9.]+) skel_iou=([0-9.]+)", txt):
                st = int(m.group(1))
                hist.setdefault(st, {"step": st, "ssim": float(m.group(2)),
                                     "skel_iou": float(m.group(3))})
        except Exception:
            pass
    out = sorted(hist.items())
    has_ssim = any("ssim" in d for _, d in out)
    return out, ("ssim" if has_ssim else "mse")


def config_summary(run_dir):
    p = os.path.join(run_dir, "resolved_config.json")
    if not os.path.exists(p):
        return {}
    try:
        c = json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}
    keys = ["model", "data_csv", "diffusion_type", "global_batch_size",
            "batch_size", "lr", "max_steps", "train_steps",
            "norm_type", "mlp_type", "qk_norm", "rope",
            "eval_csv", "eval_cfg", "condition_fusion", "callig_embed_dim",
            "char_embed_dim", "char_proj_mode", "freeze_char_table",
            "cond_drop_all_prob", "cond_drop_one_prob",
            "cond_drop_which_glyph_prob", "w_canny", "w_skel", "w_repa",
            "experiment_name"]
    return {k: c.get(k) for k in keys if c.get(k) not in (None, "")}


def pick_keeps(steps, best2, protect):
    """只保留 best2 (+ 被活动训练引用的 ckpt)."""
    keep = set(protect)
    keep.update(best2)
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正执行删除")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    now = datetime.datetime.now().timestamp()
    rows = []
    total_freed = 0

    for search_root in SEARCH_ROOTS:
        if not os.path.isdir(search_root):
            continue
        for series in sorted(os.listdir(search_root)):
            sdir = os.path.join(search_root, series)
            if not os.path.isdir(sdir) or series.startswith("_"):
                continue
            protected_series = series in PROTECT_SERIES
            for run_name in sorted(os.listdir(sdir)):
                run_dir = os.path.join(sdir, run_name)
                ckpt_dir = os.path.join(run_dir, "checkpoints")
                if not os.path.isdir(ckpt_dir):
                    continue
                candidates = [run_dir, ckpt_dir] + glob.glob(os.path.join(ckpt_dir, "*"))[:200]
                latest_mtime = max((os.path.getmtime(p) for p in candidates), default=0)
                fresh = (now - latest_mtime) < FRESH_HOURS * 3600
                pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
                if not pts:
                    continue
                steps = [int(os.path.basename(p).split(".")[0]) for p in pts]
                hist, primary = load_eval_history(run_dir, ckpt_dir)
                scored = [(s, d.get(primary)) for s, d in hist
                          if d.get(primary) is not None and s in set(steps)]
                if scored:
                    # ssim/lpips 越大越好 → 取最大两个; mse 越小越好 → 取最小两个
                    rev = primary != "mse"
                    best2 = [s for s, _ in sorted(scored, key=lambda t: t[1], reverse=rev)[:2]]
                else:
                    best2 = [steps[-1], steps[max(0, len(steps) // 2)]]
                protect_steps = {s for s, p in zip(steps, pts) if p in PROTECT_CKPTS}
                keeps = pick_keeps(steps, best2, protect_steps)
                keep_paths = {os.path.join(ckpt_dir, f"{s:07d}.pt") for s in keeps}

                if protected_series or fresh:
                    snapshot = {
                        "series": series, "run": run_name,
                        "protected": "active-series" if protected_series else "fresh(<2h)",
                        "desc": series_desc(series),
                        "config": config_summary(run_dir),
                        "primary_metric": primary,
                        "n_ckpts": len(steps), "kept_steps": sorted(keeps),
                        "eval_history_last8": hist[-8:],
                    }
                    tag = "PROTECTED"
                    n_del, freed = 0, 0
                else:
                    del_paths = [p for p in pts if p not in keep_paths
                                 and p not in PROTECT_CKPTS]
                    freed = sum(os.path.getsize(p) for p in del_paths)
                    hist_map = dict(hist)
                    best_metrics = dict(hist_map.get(best2[-1], hist[-1][1])) if hist else {}
                    snapshot = {
                        "series": series, "run": run_name,
                        "desc": series_desc(series),
                        "config": config_summary(run_dir),
                        "primary_metric": primary,
                        "best2_steps": best2,
                        "best2_metrics": {str(s): hist_map.get(s) for s in best2 if s in hist_map},
                        "eval_history_last5": hist[-5:],
                        "n_ckpts": len(steps), "kept_steps": sorted(keeps),
                        "n_deleted": len(del_paths),
                        "freed_gb": round(freed / 1024**3, 2),
                        "applied": bool(args.apply),
                    }
                    tag = "APPLIED" if args.apply else "DRYRUN"
                    n_del = len(del_paths)
                    total_freed += freed
                    if args.apply:
                        for p in del_paths:
                            os.remove(p)
                            if os.path.exists(p + ".done"):
                                os.remove(p + ".done")
                        total_freed += freed

                with open(os.path.join(OUT_DIR, f"{series}__{run_name}.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=1)
                rows.append({
                    "root": os.path.basename(search_root), "series": series,
                    "run": run_name, "status": tag,
                    "desc": series_desc(series), "n_ckpts": len(steps),
                    "n_keep": len(keeps) if tag != "PROTECTED" else len(steps),
                    "n_delete": n_del, "freed_gb": round(freed / 1024**3, 2),
                    "best_step": best2[-1] if best2 else "", "primary": primary,
                    "best_metrics": json.dumps(
                        dict(dict(hist).get(best2[-1], hist[-1][1])) if hist else {},
                        ensure_ascii=False),
                })

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'APPLIED' if args.apply else 'DRYRUN'}: {len(rows)} runs, "
          f"freed {total_freed/1024**3:.1f} GB")
    n_prot = sum(1 for r in rows if r["status"] == "PROTECTED")
    n_act = sum(1 for r in rows if r["status"] != "PROTECTED")
    print(f"protected: {n_prot}, to-clean: {n_act}")
    print(f"snapshots -> {OUT_DIR}")
    print(f"csv -> {CSV_OUT}")


if __name__ == "__main__":
    main()

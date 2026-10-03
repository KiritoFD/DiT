# -*- coding: utf-8 -*-
"""collect_all_curves.py — 收集全部相关实验的 eval 曲线，按 阶段分组。

分组:
  base : 基模预训练 (s21 / s25 / s28 / s30)
  skel : 骨架 ControlNet (s26 / s29 / s31 / ctrl_fame_1pix_v1)
  repa : repa 微调/增强 (s32 / s32b_repa_strong / s32c_chain / v8_3stage)

对 ControlNet 实验 (eval_auto_ctrl_*.json), 同时记录 base 与 ctrl 两个分支;
dashboard 主曲线用 **ctrl** 分支 (带骨架/条件后的生成质量), 并附 base 参考。

输出: _sync_work/dashboard_data_all.json
"""
import os
import sys
import glob
import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

# (key, 显示名, glob 模式, 分组, 取哪个分支: 'plain'|'ctrl')
EXPS = [
    # ---- base 基模预训练 ----
    ("s21", "s21 真迹DINO(ln_only)", "assets/results/s21_fame_flow_v2/*/checkpoints/eval_auto_*.json", "base", "plain"),
    ("s25", "s25 IDS部件码本", "assets/results/s25_ids_pretrain/*/checkpoints/eval_auto_*.json", "base", "plain"),
    ("s28", "s28 标准字形DINO(PCA+OT)", "assets/results/s28_std_dino_pretrain/*/checkpoints/eval_auto_*.json", "base", "plain"),
    ("s30", "s30 DINOchar-strong(base)", "assets/results/s30_dino_char_strong_pretrain/*/checkpoints/eval_auto_*.json", "base", "plain"),
    # ---- skel 骨架 ControlNet ----
    ("s26", "s26 ctrl-GTskel-1px", "assets/results/s26_ctrl_gt_skel/*/checkpoints/eval_auto_ctrl_*.json", "skel", "ctrl"),
    ("s29", "s29 ctrl-GTskel-1px", "assets/results/s29_ctrl_gt_skel_1px/*/checkpoints/eval_auto_ctrl_*.json", "skel", "ctrl"),
    ("s31", "s31 ctrl-GTskel-1px", "assets/results/s31_ctrl_gt_skel_1px/*/checkpoints/eval_auto_ctrl_*.json", "skel", "ctrl"),
    ("1pix", "ctrl_fame_1pix_v1", "assets/results/ctrl_fame_1pix_v1/*/checkpoints/eval_auto_ctrl_*.json", "skel", "ctrl"),
    # ---- repa ----
    ("s32b", "s32b repa-strong", "assets/results/s32b_repa_strong/*/checkpoints/eval_auto_ctrl_*.json", "repa", "ctrl"),
    ("s32c", "s32c repa-chain", "assets/results/s32c_chain/*/checkpoints/eval_auto_ctrl_*.json", "repa", "ctrl"),
    ("s32", "s32 repa-finetune", "assets/results/s32_repa_finetune/*/checkpoints/eval_auto_ctrl_*.json", "repa", "ctrl"),
]


def load_rows(pattern, branch):
    """返回 {step: row}。branch='plain' 直接读; 'ctrl' 读 ctrl 分支(并记录 base)。"""
    out = {}
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        step = d.get("step", 0)
        if branch == "ctrl":
            v = d.get("ctrl") or {}
            b = d.get("base") or {}
        else:
            v = d
            b = {}
        row = {
            "step": step,
            "ssim": v.get("ssim", 0.0),
            "lpips": v.get("lpips", None),
            "mse": v.get("mse", 0.0),
            "skel_iou": v.get("skel_iou", 0.0),
        }
        if b:
            row["base_ssim"] = b.get("ssim", 0.0)
        out[step] = row
    return out


def main():
    out = {}
    for key, name, pattern, group, branch in EXPS:
        rows_map = load_rows(pattern, branch)
        if not rows_map:
            print(f"{key:6s} ({group}): no eval data")
            continue
        rows = [rows_map[s] for s in sorted(rows_map)]
        out[key] = {"name": name, "group": group, "rows": rows}
        best = max(rows, key=lambda r: r["ssim"])
        last = rows[-1]
        extra = ""
        if "base_ssim" in last:
            extra = f" | base(last)={last['base_ssim']:.4f}"
        print(f"{key:6s} ({group}): {len(rows):>2} pts  step {rows[0]['step']}..{rows[-1]['step']}  "
              f"best_ssim={best['ssim']:.4f}@{best['step']}  last={last['ssim']:.4f}{extra}")

    with open("_sync_work/dashboard_data_all.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> _sync_work/dashboard_data_all.json ({len(out)} experiments)")


if __name__ == "__main__":
    main()

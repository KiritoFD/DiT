#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_cfg_sweep.py —— **冻结某个 ckpt，只扫书家 CFG**（不动 STD/任何其它条件 ✓）。

用途（v85 方案 §六 第 2 步）: 在不重训的前提下，先看"提高书家引导"会怎样影响
风格辨识 / 笔画粗细 / 近白失败 —— 这是最便宜的先行实验 ✓。

为什么与训练内评测**直接可比** ✓
    走的是**同一条代码路径**: `build_model_from_args` → `load_model_from_ckpt(use_ema=True)`
    → `run_in_mem_eval(...)`（同一 `resolve_eval_setup` / 同一落盘布局 ✓）。
    唯一变量 = `args.eval_cfg` ✓；`cfg_glyph_scale` 保持配置里的 None（经典 2 路）✓,
    即**两条 CFG 分支保留同一 STD** ✓。

产出
    <out-root>/cfg<C>/eval_samples_ctrl/step{N}/{set}/g{i}.png (+ gt{i}.png) ✓
    <out-root>/cfg<C>/eval_auto_{N}.json（内容指标 ✓）

用法
    python tools/eval_cfg_sweep.py --run-dir <run> --ckpt <run>/checkpoints/0200000.pt \
        --step 200000 --cfgs 1.2 1.4 1.6 --out-root <out>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_args(run_dir: str, cfg: float, out_dir: str, sets: str) -> Namespace:
    """把 run 的 resolved_config 变成评测 args（唯一改动 = eval_cfg / 落盘目录 ✓）。"""
    p = os.path.join(run_dir, "resolved_config.json")
    if not os.path.isfile(p):
        raise FileNotFoundError("缺 resolved_config.json: %s ✗" % p)
    a = Namespace(**json.load(open(p, encoding="utf-8")))
    a.in_mem_eval = True
    a.in_mem_eval_save_samples = True      # ★ 必须显式开, 否则不落 g{i}.png ✗
    a.eval_dir = str(out_dir)
    a.eval_cfg = float(cfg)                # ★ 唯一变量 ✓
    if sets:
        a.in_mem_eval_sets = sets
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="run 目录（含 resolved_config.json ✓）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, required=True, help="落盘用的 step 号（进文件名/目录 ✓）")
    ap.add_argument("--cfgs", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.6])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--sets", default="eval200fix:exp-std/csv/eval200_fixed.csv:187",
                    help="只跑主评测集（省时间 ✓）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="已有样本也重跑 ✓")
    args = ap.parse_args()

    from src.eval.in_mem_eval import run_in_mem_eval
    from src.eval.model_io import load_model_from_ckpt

    print("[sweep] ckpt=%s step=%d cfgs=%s" % (args.ckpt, args.step, args.cfgs), flush=True)
    # ★ 用 ckpt 自带的 args（= 建模型时那份 ✓，比外部 config 更准 ✓）
    model, a = load_model_from_ckpt(args.ckpt, device=args.device, use_ema=True)
    model.eval()
    # 评测开关（一次性设好 ✓；每个 cfg 只改 eval_cfg / eval_dir ✓）
    a.in_mem_eval = True
    a.in_mem_eval_save_samples = True      # ★ 必须显式开, 否则不落 g{i}.png ✗
    if args.sets:
        a.in_mem_eval_sets = args.sets

    done = []
    for c in args.cfgs:
        out = os.path.join(args.out_root, "cfg%s" % ("%g" % c))
        d = os.path.join(out, "eval_samples_ctrl", "step%07d" % args.step, "eval200fix")
        n_have = len([f for f in os.listdir(d) if f.startswith("g")]) if os.path.isdir(d) else 0
        if n_have >= 100 and not args.force:
            print("[sweep] cfg=%g 已有 %d 张, 跳过 ✓" % (c, n_have), flush=True)
            done.append((c, n_have))
            continue
        a.eval_dir = str(out)
        a.eval_cfg = float(c)              # ★ 唯一变量 ✓
        t0 = time.time()
        res = run_in_mem_eval(model, a, args.step, args.device, results_dir=out, logger=print)
        n = len([f for f in os.listdir(d) if f.startswith("g")]) if os.path.isdir(d) else 0
        _r = (res or {}).get("eval200fix") or {}
        print("[sweep] ✓ cfg=%g -> %d 张 (%.0fs) | ssim=%s ink_iou=%s"
              % (c, n, time.time() - t0, _r.get("ssim"), _r.get("ink_iou")), flush=True)
        done.append((c, n))

    print("\n[sweep] 汇总:")
    for c, n in done:
        print("    cfg=%-5g %d 张" % (c, n))
    return 0 if all(n > 0 for _, n in done) else 1


if __name__ == "__main__":
    sys.exit(main())

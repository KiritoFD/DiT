#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_checkpoint.py — 一个入口，**显式判定**该走哪条条件分支，再交给对应评测器。

分支只有两条（详见 docs/inference_branches.md）:
  · 分支 T (table): ckpt 无 glyph 分支 -> g 必须是 None          -> tools/eval_table_checkpoint.py
  · 分支 S (std)  : ckpt 有 glyph 分支 -> g 必须是非空软掩码     -> tools/eval_std_soft_checkpoint.py

判定依据**只来自 ckpt 自己的 args**，并且会把依据打印出来（不做任何猜测、不静默兜底 ✗）。
判定不明（例如有 glyph 分支但不是 std_soft，而是旧骨架 latent）时**直接报错**，
要求调用者显式说明 —— 这正是 2026-10-12"评测把条件拔掉却照样出分"的反面做法。

用法:
  $PY tools/eval_checkpoint.py --ckpt <ckpt.pt> --out <新的空目录> \
      --vae-path data/pretrained/pretrained_models/sd-vae-ft-ema
  # 想强制指定分支（一般不需要）: --branch std|table
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ckpt_args(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    return vars(a) if hasattr(a, "__dict__") else (a or {})


def decide(args):
    """返回 (branch, 依据字符串)。不确定则抛错——绝不猜 ✗。"""
    std_soft = bool(args.get("std_soft_cond", False))
    skel = bool(args.get("skel_as_glyph_cond", False)) or float(args.get("w_glyph_cond", 0) or 0) > 0
    vec = bool(args.get("glyph_vec_cond", False))

    if std_soft and (skel or vec):
        raise SystemExit("[eval_checkpoint] 同时开了 std_soft 与旧字形条件，判定不明 ✗ 请用 --branch 显式指定")
    if std_soft:
        return "std", "std_soft_cond=True（有 std 字形分支: glyph_embedder 必须收到非空 (N,1,64,64)）"
    if skel or vec:
        raise SystemExit("[eval_checkpoint] 这是**旧骨架/向量字形**分支（skel_as_glyph_cond / w_glyph_cond / glyph_vec_cond），"
                         "本仓库暂无对应的严格评测器 ✗ 请勿用 table 分支去评它")
    return "table", "std_soft_cond 未开（训练时就没有字形分支 -> g 必须是 None，喂 g 才是 bug）"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="必须是新的空目录（禁止混协议）")
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--branch", choices=["std", "table"], default=None)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    args = ckpt_args(a.ckpt)
    auto_branch, why = decide(args)
    branch = a.branch or auto_branch
    if a.branch and a.branch != auto_branch:
        print("[eval_checkpoint] ⚠ 手工指定 %s，但 ckpt 判定是 %s（%s）" % (a.branch, auto_branch, why))
    print("[eval_checkpoint] ckpt      = %s" % a.ckpt)
    print("[eval_checkpoint] 分支判定  = **%s**" % branch)
    print("[eval_checkpoint] 依据      = %s" % why)
    print("[eval_checkpoint] 协议      = gamma=%s steps=%s cfg=%s batch=%s sampler=heun(固定) 权重=ema"
          % (a.gamma, a.steps, a.cfg, a.batch))

    tool = ROOT / ("tools/eval_std_soft_checkpoint.py" if branch == "std"
                   else "tools/eval_table_checkpoint.py")
    if not tool.is_file():
        raise SystemExit("[eval_checkpoint] 缺评测器: %s" % tool)

    cmd = [sys.executable, "-u", str(tool), "--ckpt", a.ckpt, "--out", a.out,
           "--gamma", str(a.gamma), "--steps", str(a.steps), "--cfg", str(a.cfg),
           "--batch", str(a.batch), "--device", a.device, "--vae-path", a.vae_path]
    if a.sets is not None:
        cmd += ["--sets", a.sets]
    if a.preflight_only:
        cmd += ["--preflight-only"]
    print("[eval_checkpoint] 调用      = %s" % " ".join(cmd[1:]))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

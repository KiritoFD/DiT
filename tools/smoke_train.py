#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_train.py — 训练入口的**干跑体检**（不碰真数据、不写 ckpt、几秒钟）

为什么需要它:
  src/train/train.py 的 main() 是 2000+ 行的上帝函数。"没有测试覆盖的核心代码不重构"
  —— 所以在拆分它之前，必须先有一条**廉价、可重复**的路径证明"配置->args->模型->扩散->一步
  前向反传"这条链子是通的。本工具就是那条路径。

它做什么（全部合成数据，不依赖 data/ 与 exp-std/）:
   1. 读 config json -> Namespace（与训练同一套字段名）
   2. 按 args 建 DiT（含 std_soft 字形条件，如果 config 开了）
   3. 建 diffusion（flow / ddim）
   4. 造一个小 batch 的假 latent + 假条件 -> 前向 -> 反传 1~N 步
   5. 断言 loss 有限、参数真的被更新了

用法:
   python tools/smoke_train.py --config src/train/configs/v83A_stdsoft_zerofix.json [--steps 3] [--device cpu]
   # 退出码 0 = 链路通; 非 0 = 链路断了（这就是可回归的"冒烟测试"）
"""
import argparse
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--device", default="cpu", help="cpu（安全，不与训练抢卡）/ cuda:0")
ap.add_argument("--latent", type=int, default=32)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--in-channels", type=int, default=None)
a = ap.parse_args()

dev = torch.device(a.device)
print("[smoke] config = %s" % a.config)
cfg = json.load(open(a.config, encoding="utf-8"))


class _NS:
    pass


args = _NS()
for k, v in cfg.items():
    setattr(args, k, v)
# 干跑必需的兜底（真训练里由 CLI/默认值提供）
for k, v in (("num_calligraphers", 1), ("num_scripts", 3), ("num_characters", 1),
             ("image_size", 256), ("latent_channels", 4), ("learn_sigma", False),
             ("use_script_cond", False), ("num_classes", 1)):
    if not hasattr(args, k):
        setattr(args, k, v)

fail = 0
try:
    # ── 1. 建模型 ──
    from src.eval import model_io
    model = model_io.build_model_from_args(args, dev)
    n_par = sum(p.numel() for p in model.parameters())
    has_glyph = any("glyph" in n.lower() for n, _ in model.named_modules())
    print("[smoke] 模型 OK: %.1fM 参数, glyph 分支=%s" % (n_par / 1e6, has_glyph))
    if getattr(args, "std_soft_cond", False) and not has_glyph:
        raise AssertionError("config 开了 std_soft_cond 但模型里没有 glyph 分支 ✗（接线断了）")

    # ── 2. 建扩散 ──
    from src.eval.in_mem_eval import build_diffusion
    diff = build_diffusion(int(getattr(args, "eval_steps", 50)),
                           str(getattr(args, "diffusion_type", "flow")))
    print("[smoke] diffusion OK: type=%s" % getattr(args, "diffusion_type", "flow"))

    # ── 3. 合成 batch -> 前反向 ──
    in_ch = a.in_channels or int(getattr(args, "latent_channels", 4))
    B, L = a.batch, a.latent
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    x = torch.randn(B, in_ch, L, L, device=dev)
    t = torch.randint(0, 10, (B,), device=dev)
    y_callig = torch.zeros(B, dtype=torch.long, device=dev)
    y_char = torch.zeros(B, dtype=torch.long, device=dev)
    y_script = torch.zeros(B, dtype=torch.long, device=dev)
    g = (torch.rand(B, 1, 64, 64, device=dev) > 0.9).float()
    before = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    losses = []
    for i in range(a.steps):
        # ★ 位置参数陷阱（2026-10-12 干跑实测踩到）:
        #   DiT_2Cond.forward(x, t, y_callig, y_char, return_intermediate_layer=None, ...,
        #                     g=None, y_script=None, ...)
        #   第 5 个位置是 return_intermediate_layer ✗ —— 用位置传 y_script 会报
        #   "use_script_cond=True 但 forward 未收到 y_script"。**必须关键字传 g / y_script** ✓
        out = model(x, t, y_callig, y_char, g=g, y_script=y_script)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        loss = torch.mean((pred - x) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if not torch.isfinite(loss):
            raise AssertionError("loss 非有限: %r" % loss)
        print("[smoke] step %d loss=%.6f" % (i + 1, float(loss)))
    after = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    if torch.equal(before, after):
        raise AssertionError("参数没被更新（反传/优化器链路断了）✗")
    print("[smoke] ✓ 链路通: 前向+反传+优化器都生效 (loss %s)" % ["%.4f" % x for x in losses])
except Exception as e:
    fail = 1
    print("[smoke] ✗ 失败: %s: %s" % (type(e).__name__, e))
    traceback.print_exc()

sys.exit(fail)

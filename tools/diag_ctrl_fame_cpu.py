#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_ctrl_fame_cpu.py — **CPU-only** 可行性测试。

目标: 验证"用 fame GT-skel ControlNet 的 ckpt 做评测 / 换成我们的 predskel"这条路
是否现实 —— 不碰 GPU（当前 GPU 被 serial 训练占满）。

做四件事:
  1. 从 ctrl ckpt 的 args 重建主模型 (DiT-2Cond) 并灌 s21 主权重 (backbone);
  2. 用 ControlNetDiT 包上 ctrl 分支, 灌 ctrl 权重;
  3. 跑一次 forward (cond = skel latent) 验证通路;
  4. 报告耗时 -> 推算 CPU 全量评测的可行性。
"""
import os
import sys
import time

import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

MAIN = ("data/archive/results_legacy/s21_fame_flow_v2/"
        "20260829-232329-s21-fame-flow-v2/checkpoints/0030000.pt")
CTRL = ("data/archive/results_legacy/ctrl_fame_1pix_v1/"
        "20260830-205652-fame-ctrl-skel-1px-v1/checkpoints/0050000.pt")


def main():
    print("=" * 78)
    print("[1] 加载两个 ckpt (CPU)")
    print("=" * 78)
    t0 = time.time()
    mc = torch.load(MAIN, map_location="cpu", weights_only=False)
    cc = torch.load(CTRL, map_location="cpu", weights_only=False)
    print("  主 ckpt keys : %s" % list(mc.keys()))
    print("  ctrl ckpt keys: %s" % list(cc.keys()))
    print("  加载耗时 %.1fs" % (time.time() - t0))

    a = cc["args"]
    print("  args: num_calligraphers=%s num_characters=%s skel_cond_channels=%s"
          % (a.get("num_calligraphers"), a.get("num_characters"),
             a.get("skel_cond_channels")))
    print("        ctrl_depth=%s ctrl_hidden=%s injection=%s"
          % (a.get("ctrl_depth"), a.get("ctrl_hidden"), a.get("injection")))

    print()
    print("=" * 78)
    print("[2] 重建主模型 + 灌 s21 backbone")
    print("=" * 78)
    from src.eval.model_io import build_model_from_args, apply_post_construction
    t0 = time.time()
    # ⚠ 关键坑 1: 必须用 **s21 主模型自己的 args** 建 backbone。ctrl ckpt 的 args
    #   里 char_proj_mode='ln_only', 而 s21 用的是另一种 -> 直接拿 ctrl args 建会
    #   出现 char_proj.weight 缺失 / char_proj.0/1/3.* 意外 (实测 missing=4 unexpected=6)。
    # ⚠ 关键坑 2: fame 线 args **没有** image_size / vae_downscale, build_model_from_args
    #   会退回默认 (256/4) -> latent 64x64 -> pos_embed 1024 tokens, 而 ckpt 是 256 tokens
    #   (latent 32x32)。sd-vae 是 f8, 故必须显式 vae_downscale=8。
    a_main = mc["args"]
    if not isinstance(a_main, dict):
        a_main = vars(a_main)
    print("  s21 char_proj_mode=%s (ctrl args 是 %s)"
          % (a_main.get("char_proj_mode"), a.get("char_proj_mode")))
    main_m = build_model_from_args(a_main, "cpu", vae_downscale=8)
    main_m = apply_post_construction(main_m, a_main, verbose=False)
    n = sum(p.numel() for p in main_m.parameters())
    print("  主模型参数量 %.1fM, 构模 %.1fs" % (n / 1e6, time.time() - t0))
    print("  pos_embed = %s (应为 [1,256,384])" % (tuple(main_m.pos_embed.shape),))

    sd_main = mc.get("ema") or mc.get("model")
    r = main_m.load_state_dict(sd_main, strict=False)
    print("  backbone 载入: missing=%d unexpected=%d"
          % (len(r.missing_keys), len(r.unexpected_keys)))
    if r.missing_keys:
        print("     missing 示例: %s" % r.missing_keys[:6])
    if r.unexpected_keys:
        print("     unexpected 示例: %s" % r.unexpected_keys[:6])

    print()
    print("=" * 78)
    print("[3] 包 ControlNetDiT + 灌 ctrl 权重")
    print("=" * 78)
    from src.model.legacy.controlnet import ControlNetDiT
    t0 = time.time()
    ctrl = ControlNetDiT(
        main_m,
        cond_in_channels=int(a.get("skel_cond_channels", 4) or 4),
        train_ctrl_only=True,
        ctrl_depth=int(a.get("ctrl_depth", 0) or 0) or None,
        ctrl_hidden=int(a.get("ctrl_hidden", 0) or 0) or None,
        ctrl_num_heads=int(a.get("ctrl_num_heads", 0) or 0) or None,
        injection=str(a.get("injection", "modulate")),
        null_cond=str(a.get("null_cond", "gaussian")),
    ).eval()
    nc = sum(p.numel() for p in ctrl.parameters())
    print("  包装完成 %.1fs, 总参数 %.1fM (ctrl 分支 %.1fM)"
          % (time.time() - t0, nc / 1e6, (nc - n) / 1e6))

    sd_ctrl = cc.get("ema") or cc.get("ctrl")
    r2 = ctrl.load_state_dict(sd_ctrl, strict=False)
    print("  ctrl 载入: missing=%d unexpected=%d"
          % (len(r2.missing_keys), len(r2.unexpected_keys)))
    if r2.missing_keys:
        print("     missing 示例: %s" % r2.missing_keys[:6])
    if r2.unexpected_keys:
        print("     unexpected 示例: %s" % r2.unexpected_keys[:6])

    print()
    print("=" * 78)
    print("[4] CPU forward (cond = skel latent)")
    print("=" * 78)
    B = 2
    torch.manual_seed(0)
    x = torch.randn(B, 4, 32, 32)
    t = torch.full((B,), 0.5)
    y_callig = torch.zeros(B, dtype=torch.long)
    y_char = torch.zeros(B, dtype=torch.long)
    cond = torch.randn(B, 4, 32, 32)
    t0 = time.time()
    with torch.no_grad():
        out = ctrl(x, t, y_callig, y_char, cond=cond)
    dt = time.time() - t0
    o = out[0] if isinstance(out, (tuple, list)) else out
    print("  forward OK: %.2fs/batch(B=%d)  输出 shape=%s"
          % (dt, B, tuple(o.shape)))
    print("  输出 |abs| mean = %.4f  (非零 => 通路活着)" % o.abs().mean().item())

    # cond 敏感性: 换 cond 看输出是否变 (ctrl 分支是否真起作用)
    with torch.no_grad():
        o2 = ctrl(x, t, y_callig, y_char, cond=torch.randn(B, 4, 32, 32))
    o2 = o2[0] if isinstance(o2, (tuple, list)) else o2
    print("  换 cond 后 |out1-out2| mean = %.6f  (>0 => ctrl 条件真被消费)"
          % (o.abs() - o2.abs()).abs().mean().item())

    print()
    print("=" * 78)
    print("[5] CPU 全量评测可行性推算")
    print("=" * 78)
    # flow heun 50 步 -> 每次采样约 2*50 次 forward
    fwd_per_img = 2 * 50 / B
    sec_per_img = dt * fwd_per_img
    for n_img in (100, 500):
        tot = sec_per_img * n_img
        print("  %4d 张: 约 %.0f s = %.1f min" % (n_img, tot, tot / 60))
    print("  (注: 这是**极保守**上界, 未含 batch 并行收益与 VAE decode)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_forward_parity.py — **逐位前向黄金对比**（逻辑重构的护栏 ✓）

为什么需要它
------------
`strict=True` 加载只能证明**参数结构**不变 ✗ —— 对 `forward()` / `main()` 这类
**不新增参数**的逻辑重构完全没有保护。本工具: 固定权重 + 固定输入 ⇒ 记录输出的
**逐位哈希**, 重构前后必须完全一致 ✓（技能要求"对比重构前后确认行为一致" ✓）

⚠ 必须用**真实 ckpt 权重** ✗
   DiT 的 `final_layer` 是 zero-init（"step 0 恒等"设计）⇒ **随机初始化的模型输出恒为 0**
   ⇒ 拿随机权重做黄金会得到"0 vs 0"的**假通过** ✗✗（实测踩过）。
   本工具因此强制 `--ckpt`, 并内置 `--selftest` 证明护栏**灵敏**（改动输入必须改变输出 ✓）。

用法:
  python tools/verify_forward_parity.py --ckpt <ckpt> --golden <path> --save
  python tools/verify_forward_parity.py --ckpt <ckpt> --golden <path> --check
  python tools/verify_forward_parity.py --ckpt <ckpt> --selftest
  python tools/verify_forward_parity.py --ckpt <ckpt> --golden <p> --train --save|--check   # train 模式 ✓
"""
from __future__ import annotations

import argparse
import hashlib
import io as _io
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model.build import build_dit_from_args          # noqa: E402


def load(ckpt, device="cpu"):
    """按 ckpt 自带 args 建模 + **strict 加载** EMA 权重（行为等价的前提 ✓）"""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = build_dit_from_args(a, device=device)
    sd = ck.get("ema") if ck.get("ema") is not None else ck.get("model", ck)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, a


def make_inputs(a, m, seed=1234, batch=2):
    """按模型实际启用的通路构造**齐全**的条件输入（缺一个就抛 RuntimeError ✗）"""
    g = torch.Generator().manual_seed(seed)
    D = int(getattr(a, "latent_channels", 4) or 4)
    C = int(getattr(a, "image_size", 256)) // int(getattr(a, "vae_downscale", 8) or 8)
    x = torch.randn(batch, D, C, C, generator=g)
    t = torch.rand(batch, generator=g) * 1000.0
    n_cal = max(1, int(getattr(a, "num_calligraphers", 2) or 2))
    n_chr = max(1, int(getattr(a, "num_characters", 8) or 8))
    yc = torch.randint(0, n_cal, (batch,), generator=g).long()
    ych = torch.randint(0, n_chr, (batch,), generator=g).long()
    kw = {}
    if bool(getattr(m, "std_soft_cond", False)) or bool(getattr(a, "w_glyph_cond", 0)):
        ch = 1 if bool(getattr(m, "std_soft_cond", False)) else int(getattr(a, "glyph_latent_channels", 4) or 4)
        size = 64 if ch == 1 else int(getattr(a, "image_size", 256)) // int(getattr(a, "vae_downscale", 8) or 8)
        kw["g"] = (torch.rand(batch, ch, size, size, generator=g) > 0.7).float()
    if bool(getattr(m, "use_script_cond", False)):
        kw["y_script"] = torch.randint(0, max(1, int(getattr(a, "num_script_classes", 3) or 3)), (batch,), generator=g).long()
    if int(getattr(m, "hier_style", 0) or 0) > 0:
        kw["y_callig_raw"] = torch.randint(0, n_cal, (batch,), generator=g).long()
    if int(getattr(m, "num_pairs", 0) or 0) > 0:
        kw["y_pair"] = torch.randint(0, max(1, int(getattr(a, "num_pairs", 1) or 1)), (batch,), generator=g).long()
    return x, t, yc, ych, kw


def run(m, x, t, yc, ych, kw, seed=None):
    """seed 非空时在 forward 前播种 —— train 模式的 dropout / 条件 drop mask 才能逐位复现 ✓"""
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        out = m(x, t, yc, ych, **kw)
    return out[0] if isinstance(out, tuple) else out


def sha(x):
    b = _io.BytesIO()
    torch.save(x.detach().cpu().contiguous(), b)
    return hashlib.sha256(b.getvalue()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="真实 ckpt（必须 ✗ 随机权重输出恒为 0）")
    ap.add_argument("--golden", help="黄金文件路径")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="证明护栏灵敏（改输入必须改输出 ✓）")
    ap.add_argument("--cfg1", action="store_true",
                    help="★ [审查 CI] 验证 forward_with_cfg(cfg=1.0) 与条件 forward 等价 ✓")
    ap.add_argument("--train", action="store_true",
                    help="train 模式黄金（覆盖 dropout / 条件 drop mask 等 train-only 通路 ✓）")
    ap.add_argument("--seed", type=int, default=1234, help="train 模式的 RNG 种子（可逐位复现 ✓）")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(1)
    m, a = load(args.ckpt)
    if args.cfg1:
        # ★ [审查 CI 项] CFG=1 必须与条件 forward 等价（否则说明 CFG 路径有额外副作用 ✗）
        x, t, yc, ych, kw = make_inputs(a, m, batch=args.batch)
        base = run(m, x, t, yc, ych, kw)
        try:
            cfg = m.forward_with_cfg(x, t, yc, ych, cfg_scale=1.0, **kw)
        except TypeError:
            cfg = m.forward_with_cfg(x, t, yc, ych, cfg_scale=1.0)
        if isinstance(cfg, tuple):
            cfg = cfg[0]
        d = float((cfg - base).abs().max())
        rel = d / max(1e-9, float(base.abs().max()))
        print(f"  CFG=1 vs 条件 forward: max|diff|={d:.3e}  相对={rel:.3e}")
        if rel <= 1e-4:
            print("  ✓ 等价（在 1e-4 相对容差内 ✓）—— 审查 CI 项通过 ✓")
            return 0
        print("  ✗ **不等价** ✗ —— CFG 路径存在额外行为, 需查（审查 CI 项不过 ✗）")
        return 1
    mode = "train" if args.train else "eval"
    m.train() if args.train else m.eval()
    x, t, yc, ych, kw = make_inputs(a, m, batch=args.batch)
    out = run(m, x, t, yc, ych, kw, seed=(args.seed if args.train else None))
    h = sha(out)
    meta = {"ckpt": os.path.basename(args.ckpt), "batch": args.batch, "mode": mode,
            "seed": (args.seed if args.train else None),
            "shape": list(out.shape), "sha256": h,
            "abs_sum": float(out.abs().sum()),
            "std_soft_cond": bool(getattr(m, "std_soft_cond", False)),
            "use_char_cond": bool(getattr(m, "use_char_cond", False))}
    print("  shape=%s sha256=%s abs_sum=%.4f" % (meta["shape"], h[:16], meta["abs_sum"]))
    if meta["abs_sum"] == 0.0:
        print("  ✗ 输出恒为 0 —— 本 ckpt 的 final_layer 仍是 zero-init, 无法用作黄金 ✗")
        return 3

    if args.selftest:
        rc = 0
        # ① 改 y_char -> 输出必须变
        ych2 = (ych + 1) % max(1, int(getattr(a, "num_characters", 8) or 8))
        sd = args.seed if args.train else None
        d1 = float((run(m, x, t, yc, ych2, kw, seed=sd) - out).abs().max())
        # ② 改 t -> 必须变
        d2 = float((run(m, x, t + 1.0, yc, ych, kw, seed=sd) - out).abs().max())
        # ③ 改 g（若启用）-> 必须变
        d3 = None
        if "g" in kw:
            kw2 = dict(kw); kw2["g"] = kw["g"].flip(0)
            d3 = float((run(m, x, t, yc, ych, kw2, seed=sd) - out).abs().max())
        print("  灵敏度: Δy_char=%.4e  Δt=%.4e  Δg=%s" % (d1, d2, ("%.4e" % d3) if d3 is not None else "n/a"))
        if d1 <= 0: print("  ✗ 改字条件输出不变 —— 护栏不灵敏 ✗"); rc = 1
        if d2 <= 0: print("  ✗ 改 t 输出不变 —— 护栏不灵敏 ✗"); rc = 1
        if d3 is not None and d3 <= 0: print("  ✗ 改 g 输出不变 —— 护栏不灵敏 ✗"); rc = 1
        if rc == 0: print("  ✓ 护栏灵敏 ✓（三类输入扰动都能反映到输出）")
        return rc

    if args.save:
        if not args.golden: print("  ✗ --save 需要 --golden"); return 2
        torch.save({"out": out.detach().cpu(), "meta": meta}, args.golden)
        print("  ✓ 黄金已保存 ->", args.golden)
        return 0
    if args.check:
        if not args.golden or not os.path.isfile(args.golden):
            print("  ✗ 黄金不存在:", args.golden); return 2
        g = torch.load(args.golden, map_location="cpu", weights_only=False)
        bit = bool(torch.equal(g["out"], out.detach().cpu()))
        gm = g["meta"]
        if gm.get("mode", "eval") != mode:
            print("  ✗ 黄金模式(%s)与当前(%s)不一致 —— 别混用 ✗" % (gm.get("mode", "eval"), mode))
            return 2
        print("  黄金哈希=%s  当前哈希=%s  (mode=%s)" % (gm["sha256"][:16], h[:16], mode))
        if bit:
            print("  ✓ 前向**逐位一致** —— 重构未改变行为 ✓✓")
            return 0
        print("  ✗ 前向结果变了 (max|diff|=%.3e) —— 重构改变了行为 ✗✗"
              % float((g["out"] - out.detach().cpu()).abs().max()))
        return 1
    print("  （未指定 --save / --check / --selftest）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

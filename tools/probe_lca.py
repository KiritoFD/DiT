#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看 LocalStyleGlyphAdapter 有没有离开 zero-init。

读 ckpt 里 out_proj 的范数, 再跑一次前向, 量适配器输出相对残差流的比例。
0 附近 = 这两层还没起作用。
"""
import argparse
import os
import sys

import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from src.eval.model_io import load_model_from_ckpt
    model, _ = load_model_from_ckpt(a.ckpt, device=a.device, use_ema=True, verbose=False)
    model.eval()
    adapters = list(model.local_ca)
    print(f"layers {len(adapters)}", flush=True)
    for i, lc in enumerate(adapters):
        w = lc.out_proj.weight.detach()
        print(f"  adapter {i}  out_proj ‖W‖ {float(w.norm()):.4e}  "
              f"max {float(w.abs().max()):.4e}", flush=True)

    dev = a.device
    B = a.n
    g = th.randn(B, 4, 32, 32, device=dev)
    x = th.randn(B, 4, 32, 32, device=dev)
    t = th.full((B,), 500.0, device=dev)
    yc = th.arange(B, device=dev) % 45
    yh = th.zeros(B, dtype=th.long, device=dev)
    captured = {}

    def make(i, lc):
        def hook(_m, inputs, output):
            src = inputs[0]
            delta = output - src
            captured[i] = float(delta.norm() / (src.norm() + 1e-8))
        return hook

    handles = [lc.register_forward_hook(make(i, lc)) for i, lc in enumerate(adapters)]
    with th.no_grad():
        model(x, t, y_callig=yc, y_char=yh, g=g)
    for h in handles:
        h.remove()
    for i, v in captured.items():
        print(f"  adapter {i}  ‖Δx‖/‖x‖ {v:.4e}", flush=True)


if __name__ == "__main__":
    main()

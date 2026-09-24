#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针: GlyphQuery 有没有真的在起作用（还是停在 zero-init 当摆设）。

测四类量：
  1. out_log_scale -> exp()      ：LayerScale 的实际值（初值 0.1）。有没有被训动。
  2. 各投影的 ‖W‖                ：out_proj / style_to_q / style_to_k 初值全 0。
  3. 前向 hook 量 ‖Δx‖/‖x‖       ：该层对残差流的**实际相对贡献**（初值 0）。
  4. 分 t 的贡献                 ：低 t（收尾）和高 t（早期）分别贡献多少。

用法: python tools/probe_glyph_query.py --ckpt <path> [--n 16] [--device cpu]
"""
import argparse
import os
import sys

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--n', type=int, default=16)
ap.add_argument('--device', default='cpu')
ap.add_argument('--ts', default='0.1,0.5,0.9')
a = ap.parse_args()

from src.eval.model_io import load_model_from_ckpt  # noqa: E402

dev = a.device
model, _args = load_model_from_ckpt(a.ckpt, device=dev, use_ema=True, verbose=False)
model.eval()
raw = model
for attr in ('module', '_orig_mod'):
    if hasattr(raw, attr):
        raw = getattr(raw, attr)

lca = getattr(raw, 'local_ca', None)
if lca is None:
    raise SystemExit('✗ 该 ckpt 没有 local_ca')
print(f'ckpt = {a.ckpt}')
print(f'local_ca: {len(lca)} 层, 类型 {type(lca[0]).__name__}')
print(f'插入位置 _local_ca_map = {getattr(raw, "_local_ca_map", None)}')
print()

print('=== 1/2. 参数是否离开初值 ===')
print(f'{"层":>4} | {"exp(out_log_scale)":>19} | {"‖out_proj‖":>12} {"‖style_q‖":>12} '
      f'{"‖style_k‖":>12} | {"‖q_proj‖":>11} {"‖v_proj‖":>11}')
for i, lc in enumerate(lca):
    ls = float(lc.out_log_scale.exp()) if hasattr(lc, 'out_log_scale') else float('nan')
    ow = float(lc.out_proj.weight.norm())
    sq = float(lc.style_to_q.weight.norm()) if hasattr(lc, 'style_to_q') else float('nan')
    sk = float(lc.style_to_k.weight.norm()) if hasattr(lc, 'style_to_k') else float('nan')
    qw = float(lc.q_proj.weight.norm())
    vw = float(lc.v_proj.weight.norm())
    print(f'{i:>4} | {ls:19.6f} | {ow:12.4f} {sq:12.4f} {sk:12.4f} | {qw:11.4f} {vw:11.4f}')
print('  （初值: exp(out_log_scale)=0.1；out_proj / style_to_q / style_to_k 的 ‖W‖ = 0）')

print()
print('=== 3/4. 前向 hook: 该层对残差流的相对贡献 ‖Δx‖/‖x‖ ===')
print('       （初值 0；>0 说明真的在改变主干特征）')
captured = {}


def mk(i):
    def hook(_m, inputs, output):
        src = inputs[0]
        captured.setdefault(i, []).append(
            float((output - src).norm() / (src.norm() + 1e-8)))
    return hook


handles = [lc.register_forward_hook(mk(i)) for i, lc in enumerate(lca)]
try:
    B = a.n
    x = torch.randn(B, 4, 32, 32, device=dev)
    g = torch.randn(B, 4, 32, 32, device=dev)
    yc = torch.arange(B, device=dev) % 45
    yh = torch.zeros(B, dtype=torch.long, device=dev)
    for tv in [float(v) for v in a.ts.split(',') if v.strip()]:
        captured.clear()
        t = torch.full((B,), tv * 1000.0, device=dev)
        with torch.no_grad():
            model(x, t, y_callig=yc, y_char=yh, g=g)
        row = ' '.join(f'层{i}={np.mean(v):.5f}' for i, v in sorted(captured.items()) if v)
        print(f'  t={tv:.2f}: {row}')
finally:
    for h in handles:
        h.remove()

print()
print('=== 判读 ===')
moved = []
for i, lc in enumerate(lca):
    ow = float(lc.out_proj.weight.norm())
    moved.append(ow > 1e-6)
print(f'  out_proj 已离开 0 的层: {[i for i, m in enumerate(moved) if m]} / 共 {len(lca)} 层')
if not any(moved):
    print('  ⚠ 全部仍为 0 -> 这两层是摆设（要么没训到，要么梯度没通）')

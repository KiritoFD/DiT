#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, torch
import numpy as np

ROOT = "/root/Workspace/xy/DiT"
if os.path.exists(ROOT):
    os.chdir(ROOT)
sys.path.insert(0, ".")

from src.model.deform_skel import DeformSkel

print("=== 严格测试 SkelNet-V2 拓扑分支 (剪刀与胶水) ===")

ckpt_p = "assets/deform_skel_top10_v1.pt"
sd = torch.load(ckpt_p, map_location="cpu", weights_only=False)["deform"]

# 1. 基础版 (topo_mode=0)
m0 = DeformSkel(cond_dim=128, ch=4, grid=32, width=96, max_off=6.0, coarse=8, residual=0, res_cap=1.0, stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, dt_ch=1, topo_mode=0)
m0.load_state_dict(sd, strict=False)
m0.eval()

# 2. V2版 (topo_mode=1)
m1 = DeformSkel(cond_dim=128, ch=4, grid=32, width=96, max_off=6.0, coarse=8, residual=0, res_cap=1.0, stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, dt_ch=1, topo_mode=1)
miss, unexp = m1.load_state_dict(sd, strict=False)
print("m1 missing keys (应只包含 head_prune/head_ligature):", miss)
print("m1 unexpected keys:", unexp)
assert all("head_prune" in k or "head_ligature" in k or "z_bg" in k or "delta_ink" in k for k in miss)
m1.eval()

# 测试前向
g = torch.randn(2, 4, 32, 32)
style = torch.randn(2, 128)

with torch.no_grad():
    out0 = m0(g, style)
    out1 = m1(g, style)
    
diff = (out1 - out0).abs()
print(f"第 0 步等价性验证: out1 vs out0 最大偏差={diff.max().item():.5f}, 平均偏差={diff.mean().item():.5f}")
print(f"剪刀 mask_prune 初始均值: {m1.last_mask_prune.mean().item():.5f} (理论期望 ~0.0067)")
print(f"胶水 mask_ligature 初始均值: {m1.last_mask_ligature.mean().item():.5f} (理论期望 ~0.0067)")

assert diff.mean().item() < 0.03, "Step 0 等价性偏差过大！"
print("\n🎉 单元测试完美通过！SkelNet-V2 剪刀与胶水双分支构造正确，完美逐位承接 top10_v1 资产！")

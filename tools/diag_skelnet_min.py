#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_skelnet_min.py — 最小化验证 SkelNet std->pred 位移，全部 CPU，避免任何 GPU/VAE 依赖崩溃。"""
import os, sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
torch.set_num_threads(4)

from src.model.deform_skel import DeformSkel

sd = torch.load("assets/deform_skel_top10_v1.pt", map_location="cpu", weights_only=False)
sd = sd.get("deform", sd)

sk = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=96,
                max_off=6.0, coarse=8, residual=0, res_cap=1.0,
                stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, dt_ch=1)
miss, unexp = sk.load_state_dict(sd, strict=False)
print("load missing=%d unexpected=%d" % (len(miss), len(unexp)))
sk.eval()

# 直接从分片取真实 skel latent, 不经过 VAE
d = np.load("data/top10_style23/shards_std/shard_00000.npz")
lat = torch.from_numpy(d["latents"][:4]).float()   # (4,4,32,32)
print("std latent shape", tuple(lat.shape), "mean=%.4f std=%.4f" % (lat.mean(), lat.std()))

emb = torch.load("assets/callig_script_emb_top10.pt", map_location="cpu",
                 weights_only=False)["embedding"].float()
print("style table", tuple(emb.shape))

print()
print("同一 std 骨架 x 不同风格槽位:")
outs = {}
with torch.no_grad():
    for pid in [0, 11, 21, 22]:
        e = emb[pid:pid + 1].repeat(4, 1)
        out = sk(lat, e)
        if isinstance(out, (tuple, list)):
            out = out[0]
        outs[pid] = out
        diff = (out - lat).abs().mean().item()
        print("  slot %2d  |out-std|_mean=%.5f  out_mean=%.4f out_std=%.4f  out_range[%.3f, %.3f]" % (
            pid, diff, out.mean(), out.std(), out.min(), out.max()))

print()
print("不同风格之间输出的差异 (应非零 -> 风格确实驱动形变):")
keys = list(outs.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        dd = (outs[keys[i]] - outs[keys[j]]).abs().mean().item()
        print("  slot %2d vs %2d : |Δ|=%.5f" % (keys[i], keys[j], dd))

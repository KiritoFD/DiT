#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_skelnet_effective.py — 独立验证:
1) v24_frozenskel 启动时 deform 权重是否被完整载入(且未被初始化冲刷);
2) SkelNet 是否真的在做 std->pred 的有效形变(位移场非零、非饱和);
3) 下游 frozen 训练 SSIM 随 step 的走势。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

CKPT = "assets/deform_skel_top10_v1.pt"
print("=" * 80)
print("[A] 离线 SkelNet 权重体检:", CKPT)
print("=" * 80)
d = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = d.get("deform", d) if isinstance(d, dict) else d
print("  顶层键:", list(d.keys()) if isinstance(d, dict) else "raw")
print("  参数量:", sum(v.numel() for v in sd.values() if torch.is_tensor(v)))
print()
print("  关键层模长 (应与离线真值一致; 若 == Xavier 期望则被冲刷):")
for k in ["style_proj.weight", "affine.weight", "style_off.weight", "out.weight"]:
    if k in sd:
        print("    %-24s |W|_sum=%.4f  |W|_mean=%.6f" % (
            k, sd[k].abs().sum(), sd[k].abs().mean()))
keys = list(sd.keys())
print("  全部键(%d): %s" % (len(keys), keys[:12]))

print()
print("=" * 80)
print("[B] 构造 DiT 模型, 检查 deform 是否被 initialize_weights 冲刷")
print("=" * 80)
try:
    from src.eval.model_io import load_model_from_ckpt
    import glob
    cds = glob.glob("assets/results/v24_frozenskel/*/checkpoints")
    print("  ckpt 目录:", cds)
    if cds:
        ck = sorted(glob.glob(cds[0] + "/*.pt"))
        print("  可用 ckpt:", [os.path.basename(c) for c in ck][:12])
        if ck:
            target = ck[-1]
            model, cfg = load_model_from_ckpt(target, device="cpu", use_ema=True, verbose=False)
            ds_ = getattr(model, "deform_skel", None)
            if ds_ is None:
                print("  !! 模型里没有 deform_skel")
            else:
                m2 = dict(ds_.state_dict())
                print("  载入后模长:")
                for k in ["style_proj.weight", "affine.weight", "style_off.weight"]:
                    if k in m2:
                        a = float(m2[k].abs().sum())
                        b = float(sd[k].abs().sum()) if k in sd else float("nan")
                        print("    %-24s 模型=%.4f  离线=%.4f  Δ=%.4f %s" % (
                            k, a, b, abs(a - b), "✓一致" if abs(a-b) < 1e-3 else "✗不一致/被冲刷"))
except Exception as e:
    import traceback; traceback.print_exc()

print()
print("=" * 80)
print("[C] SkelNet 位移场有效性 (std -> pred 是否真在动)")
print("=" * 80)
try:
    from src.model.deform_skel import DeformSkel
    from src.eval.in_mem_eval import _get_vae
    from src.utils.callig_script_map import load_callig_script_map
    import pandas as pd

    csmap = load_callig_script_map("assets/callig_script_id_map_top10.json")
    # 构造一个 skelnet (用评测里同样的参数)
    import inspect
    sk = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=96,
                    max_off=6.0, coarse=8, residual=0, res_cap=1.0,
                    stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, dt_ch=1)
    miss, unexp = sk.load_state_dict(sd, strict=False)
    print("  load: missing=%d unexpected=%d" % (len(miss), len(unexp)))
    sk = sk.to("cuda")
    sk.eval()

    vae = _get_vae("cuda")
    df = pd.read_csv("assets/train_top10_style23.csv")
    # 取几张真实 std latent
    ds_std = np.load("data/top10_style23/shards_std/shard_00000.npz")
    lat = torch.from_numpy(ds_std["latents"][:4]).float().to("cuda")
    sf = 0.18215
    with torch.no_grad():
        g0 = ((vae.decode(lat / sf).sample.clamp(-1, 1) + 1) / 2).mean(1)  # (4,H,W)
    g0 = g0.unsqueeze(1)  # (4,1,H,W)

    # 风格向量: 取两个不同槽位
    emb = torch.load("assets/callig_script_emb_top10.pt", map_location="cpu",
                     weights_only=False)["embedding"].float()
    print()
    print("  同一 std 骨架 x 不同风格槽位 -> 输出位移幅度:")
    base = None
    for pid in [0, 11, 21, 22]:
        e = emb[pid:pid+1].repeat(4, 1).to("cuda")
        with torch.no_grad():
            out = sk(g0, e)
        if isinstance(out, (tuple, list)):
            out = out[0]
        diff = (out - g0).abs().mean().item()
        mm = out.mean().item()
        print("    slot %2d  |mean(out-std)|=%.5f  out均值=%.4f" % (pid, diff, mm))
    print()
    print("  判读: 若 |out-std| 极小(<0.005) -> 形变几乎没生效(实现/权重问题)")
    print("        若 out均值 贴近 0 或 1 且 diff 很大 -> 位移饱和/爆掉")
except Exception as e:
    import traceback; traceback.print_exc()

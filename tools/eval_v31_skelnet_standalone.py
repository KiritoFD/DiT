#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/eval_v31_skelnet_standalone.py — 评估 v31_stage1_skel 骨架生成器的独立预测能力与质量"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval import model_io
from src.eval.inference import sample_latents, build_diffusion
from src.eval.in_mem_eval import _get_vae, _get_cache
from src.utils.callig_script_map import load_callig_script_map
from types import SimpleNamespace

def cl_dice_2d(a_bin, b_bin):
    from scipy.ndimage import binary_dilation
    st = np.ones((3, 3), bool)
    ad = binary_dilation(a_bin, structure=st, iterations=3)
    bd = binary_dilation(b_bin, structure=st, iterations=3)
    inter = (a_bin & bd).sum() + (b_bin & ad).sum()
    return 2.0 * inter / max(a_bin.sum() + b_bin.sum(), 1)

def main():
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("=== 开始评估 v31_stage1_skel @ 80,000 步的独立骨架生成质量 ===")
    
    # 1. 查找 checkpoint
    ckpt_p = "exp/v31_stage1_skel/20261001-005831-v31-stage1-skel/checkpoints/0080000.pt"
    if not os.path.exists(ckpt_p):
        ckpts = sorted(glob.glob("exp/v31_stage1_skel/*/checkpoints/*.pt"))
        ckpt_p = ckpts[-1]
    print(f"载入 Checkpoint: {ckpt_p}")
    
    model, args = model_io.load_model_from_ckpt(ckpt_p, device=dev, use_ema=True)
    model.eval()
    
    csmap = load_callig_script_map("assets/callig_script_id_map_top10.json")
    vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").float()
    diff = build_diffusion(25, "flow")
    
    # 2. 评测集设置: Seen 20 与 Strict 84
    # 注意: v31_stage1_skel 任务是: 条件 g = shards_std, 目标 = shards_gtskel_w3
    for set_name, csv_p, sh_std, sh_gt, n_eval in [
        ("Seen-20", "assets/eval_top10_seen_20.csv", "data/top10_style23/shards_std", "data/top10_style23/shards_gtskel_w3", 20),
        ("Strict-84", "assets/eval_v13_strict84_aligned.csv", "data/50k_v2_glyph15k/shards_std", "data/top10_style23/gt_skel_eval_strict84", 84)
    ]:
        print(f"\n==================== 评估 {set_name} (n={n_eval}) ====================")
        cache_std = _get_cache(csv_p, n_eval, None, sh_std, args)
        cache_gt = _get_cache(csv_p, n_eval, None, sh_gt, args)
        
        g_std = cache_std["skels_latent"].to(dev) # (N, 4, 32, 32)
        g_gt = cache_gt["skels_latent"].to(dev)   # (N, 4, 32, 32)
        conds = cache_std["conds"]
        noise = cache_std["noise"].to(dev)
        
        # Euler 25 步采样生成预测骨架 g_pred
        t0 = time.time()
        with th.no_grad():
            g_pred = sample_latents(model, diff, noise, conds, 1.0, 32, dev, skel=g_std, seed=0)
        t_sample = time.time() - t0
        g_pred = g_pred.to(dev)
        
        # 3. 潜空间 Latent 指标
        mse_std_gt = float((g_std - g_gt).pow(2).mean().item())
        mse_pred_gt = float((g_pred - g_gt).pow(2).mean().item())
        cos_std_gt = float(th.cosine_similarity(g_std.flatten(1), g_gt.flatten(1)).mean().item())
        cos_pred_gt = float(th.cosine_similarity(g_pred.flatten(1), g_gt.flatten(1)).mean().item())
        
        # 4. 闭合率 = 1 - MSE(g_pred, g_gt) / MSE(g_std, g_gt)
        closure_rate = (1.0 - mse_pred_gt / max(mse_std_gt, 1e-6)) * 100.0
        
        print(f"Latent 空间对比:")
        print(f"  基线 MSE(std, gt):    {mse_std_gt:.5f} | 余弦相似度: {cos_std_gt:.4f}")
        print(f"  预测 MSE(pred, gt):   {mse_pred_gt:.5f} | 余弦相似度: {cos_pred_gt:.4f}")
        print(f"  闭合率 (Closure Rate): {closure_rate:.2f}%  (生成耗时: {t_sample:.1f}s)")
        
        # 5. 解码为 256x256 进行像素级 Dice@3px 与墨量比核对
        with th.no_grad():
            dec_s = vae.decode(g_std[:5] / 0.18215).sample
            dec_p = vae.decode(g_pred[:5] / 0.18215).sample
            dec_g = vae.decode(g_gt[:5] / 0.18215).sample
            
        im_s = ((dec_s.clamp(-1, 1) + 1) / 2).cpu().numpy()[:, 0]
        im_p = ((dec_p.clamp(-1, 1) + 1) / 2).cpu().numpy()[:, 0]
        im_g = ((dec_g.clamp(-1, 1) + 1) / 2).cpu().numpy()[:, 0]
        
        df_csv = pd.read_csv(csv_p)
        print(f"\n前 3 个样本的形态学重合度与 ASCII 诊断:")
        for i in range(min(3, n_eval)):
            ch = df_csv.iloc[i]["character"]
            c_name = df_csv.iloc[i]["calligrapher"]
            s_name = df_csv.iloc[i]["script"]
            
            d3_s = cl_dice_2d(im_s[i] < 0.5, im_g[i] < 0.5)
            d3_p = cl_dice_2d(im_p[i] < 0.5, im_g[i] < 0.5)
            ink_s = (im_s[i] < 0.5).mean()
            ink_p = (im_p[i] < 0.5).mean()
            ink_g = (im_g[i] < 0.5).mean()
            
            print(f"\n[{i}] {c_name}·{s_name}·'{ch}':")
            print(f"    Dice@3px vs GT:  std={d3_s:.4f}  ──►  pred={d3_p:.4f} (提升 Δ = {d3_p - d3_s:+.4f})")
            print(f"    墨量比 (Ink%):   std={ink_s*100:.2f}% | pred={ink_p*100:.2f}% | GT={ink_g*100:.2f}%")
            
            # ASCII 预览 pred 骨架
            sub = np.asarray(Image.fromarray((im_p[i] * 255).astype(np.uint8)).resize((24, 24), Image.Resampling.LANCZOS))
            print("    预测骨架 ASCII:")
            for y in range(24):
                line = "".join(["#" if sub[y, x] < 180 else " " for x in range(24)])
                if "#" in line:
                    print(f"      {line}")

if __name__ == "__main__":
    main()

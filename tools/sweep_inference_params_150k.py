#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/sweep_inference_params_150k.py — 对 v24_top10 终点 150k 检查点进行全方位推理参数扫描"""
import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval.inference import make_eval_cache, sample_latents, build_diffusion
from src.eval.in_mem_eval import _get_vae
from src.eval.model_io import load_model_from_ckpt
from src.utils.callig_script_map import load_callig_script_map
from skimage.metrics import structural_similarity as ssim_fn

def count_fragments(img_np):
    bin_inv = (img_np < 0.5).astype(np.uint8)
    if bin_inv.sum() < 10:
        return 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(bin_inv)
    return n - 1

def main():
    ckpt_p = "assets/results/v24_top10_style23/20260927-213913-v24-top10-style23/checkpoints/0150000.pt"
    if not os.path.exists(ckpt_p):
        print("未找到 150k 检查点:", ckpt_p)
        return

    print(f"=== [150k 终点参数扫描] 加载检查点: {ckpt_p} ===")
    model, _ = load_model_from_ckpt(ckpt_p, device="cuda", use_ema=True, verbose=False)
    model.eval()

    csmap = load_callig_script_map("assets/callig_script_id_map_top10.json")

    # 载入全部 84 个严格未见生僻字样本
    eval_csv = "assets/eval_top10_strict_subset84.csv"
    cache = make_eval_cache(
        eval_csv, None, None, 256, 84, 8, 4, 0.18215,
        skel_latent_shards_dir="data/50k_v2_glyph15k/shards_std",
        callig_script_map=csmap
    )
    n_samples = len(cache["conds"])
    print(f"评测样本数: {n_samples} 张")

    vae = _get_vae("cuda")

    # 参数组合网格
    test_grid = [
        # (cfg_scale, steps, skel_mode, desc)
        (0.7, 50, "deform", "基线 (SkelNet + CFG 0.7 + Steps 50)"),
        (0.5, 50, "deform", "低CFG (SkelNet + CFG 0.5 + Steps 50)"),
        (0.9, 50, "deform", "高CFG (SkelNet + CFG 0.9 + Steps 50)"),
        (1.0, 50, "deform", "纯条件 (SkelNet + CFG 1.0 + Steps 50)"),
        (1.0, 25, "deform", "极速纯条件 (SkelNet + CFG 1.0 + Steps 25)"),
        (1.0, 75, "deform", "高精纯条件 (SkelNet + CFG 1.0 + Steps 75)"),
        (0.7, 50, "bypass", "直通骨架 (g_std + CFG 0.7 + Steps 50)"),
        (1.0, 50, "bypass", "直通骨架 (g_std + CFG 1.0 + Steps 50)"),
        (1.0, 50, "boost1.25", "补偿骨架 (SkelNet x1.25 + CFG 1.0 + Steps 50)"),
    ]

    out_records = []
    orig_deform = model.deform_skel

    for cfg, steps, skel_mode, desc in test_grid:
        diff = build_diffusion(steps, "flow")

        # 骨架模式切换
        if skel_mode == "bypass":
            model.deform_skel = None
        else:
            model.deform_skel = orig_deform

        with torch.no_grad():
            cur_skel = cache["skels_latent"].clone()
            if skel_mode == "boost1.25":
                cur_skel = cur_skel * 1.25

            lat = sample_latents(model, diff, cache["noise"], cache["conds"], cfg, 16, "cuda", skel=cur_skel)
            
            # 分批解码防 OOM
            pred_ims = []
            for b_idx in range(0, n_samples, 8):
                sub_lat = (lat[b_idx:b_idx+8] / 0.18215).to("cuda")
                sub_im = (vae.decode(sub_lat).sample.clamp(-1, 1) + 1) / 2
                pred_ims.append(sub_im.cpu())
            pred_im = torch.cat(pred_ims, dim=0)

            gts = (cache["gts"].to("cuda") + 1) / 2

            ssims, mses, frags, ious = [], [], [], []
            for i in range(n_samples):
                p_np = pred_im[i].permute(1, 2, 0).cpu().numpy().mean(-1)
                g_np = gts[i].permute(1, 2, 0).cpu().numpy().mean(-1)

                s = ssim_fn(p_np, g_np, data_range=1.0)
                m = float(np.mean((p_np - g_np) ** 2))
                frag = count_fragments(p_np)
                
                # IoU
                p_bin = p_np < 0.5
                g_bin = g_np < 0.5
                inter = np.logical_and(p_bin, g_bin).sum()
                union = np.logical_or(p_bin, g_bin).sum()
                iou = inter / max(1, union)

                ssims.append(s)
                mses.append(m)
                frags.append(frag)
                ious.append(iou)

            res = {
                "desc": desc,
                "cfg": cfg,
                "steps": steps,
                "skel_mode": skel_mode,
                "ssim_mean": round(float(np.mean(ssims)), 4),
                "ssim_med": round(float(np.median(ssims)), 4),
                "mse_mean": round(float(np.mean(mses)), 4),
                "ink_iou": round(float(np.mean(ious)), 4),
                "avg_fragments": round(float(np.mean(frags)), 1),
            }
            out_records.append(res)
            print(f"[{desc}]: SSIM = {res['ssim_mean']} (中位 {res['ssim_med']}) | MSE = {res['mse_mean']} | 碎片数 = {res['avg_fragments']} (越低越连贯)")

    model.deform_skel = orig_deform

    df_res = pd.DataFrame(out_records)
    out_csv = "assets/results/v24_top10_style23/sweep_inference_150k.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n🎉 150k 推理参数全网格扫描完成！结果已保存至: {out_csv}")

if __name__ == "__main__":
    main()

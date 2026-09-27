#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/fix_and_regenerate_seen_eval.py — 用正确的 top10 骨架分片重构修复 seen 评测与海报"""
import os
import sys
import glob
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval.inference import make_eval_cache, sample_latents, build_diffusion
from src.eval.in_mem_eval import _get_vae, render_poster, save_input_g
from src.eval.model_io import load_model_from_ckpt
from src.utils.callig_script_map import load_callig_script_map
from skimage.metrics import structural_similarity as ssim_fn

def main():
    results_dir = "assets/results/v24_top10_style23"
    csmap = load_callig_script_map("assets/callig_script_id_map_top10.json")

    # 1. 载入正确的 seen cache (指向 data/top10_style23/shards_std)
    print("[1] 载入正确的 seen cache (data/top10_style23/shards_std)...")
    cache = make_eval_cache(
        "assets/eval_top10_seen_20.csv", None, None, 256, 20, 8, 4, 0.18215,
        skel_latent_shards_dir="data/top10_style23/shards_std",
        callig_script_map=csmap
    )

    diff = build_diffusion(50, "flow")
    vae = _get_vae("cuda")

    # 2. 刷新落盘正确的输入骨架图 seen_input_g
    print("[2] 刷新正确的输入骨架到 eval_samples_ctrl/seen_input_g/ ...")
    save_input_g(results_dir, "seen", cache["skels_latent"], vae, 0.18215, batch=16)

    # 3. 对代表性步骤 (如 50000, 75000, 100000, 105000) 重新推断并覆盖生成图
    steps_to_fix = [50000, 75000, 100000, 105000]
    ckpt_dirs = glob.glob(f"{results_dir}/*/checkpoints")
    if not ckpt_dirs:
        print("未找到检查点目录")
        return
    ckpt_dir = ckpt_dirs[0]

    for st in steps_to_fix:
        ckpt_p = os.path.join(ckpt_dir, f"{st:07d}.pt")
        if not os.path.exists(ckpt_p):
            continue
        print(f"\n[3] 重新生成并修正 Step {st} 的 seen 样本...")
        model, _ = load_model_from_ckpt(ckpt_p, device="cuda", use_ema=True, verbose=False)
        model.eval()

        with torch.no_grad():
            lat = sample_latents(model, diff, cache["noise"], cache["conds"], 0.7, 16, "cuda", skel=cache["skels_latent"])
            del model
            torch.cuda.empty_cache()

            pred_ims = []
            for b_idx in range(0, 20, 4):
                sub_lat = (lat[b_idx:b_idx+4] / 0.18215).to("cuda")
                sub_im = (vae.decode(sub_lat).sample.clamp(-1, 1) + 1) / 2
                pred_ims.append(sub_im.cpu())
            pred_im = torch.cat(pred_ims, dim=0)
            gts = (cache["gts"].to("cuda") + 1) / 2

            out_sd = os.path.join(results_dir, "eval_samples_ctrl", f"step{st:07d}", "g")
            os.makedirs(out_sd, exist_ok=True)

            ssims = []
            for i in range(20):
                p_np = pred_im[i].permute(1, 2, 0).cpu().numpy()
                g_np = gts[i].permute(1, 2, 0).cpu().numpy()
                s = ssim_fn(p_np, g_np, channel_axis=-1, data_range=1.0)
                ssims.append(s)

                # 覆盖落盘 g{i}.png 与 gt{i}.png
                Image.fromarray((p_np * 255).astype(np.uint8)).save(os.path.join(out_sd, f"g{i}.png"))
                Image.fromarray((g_np * 255).astype(np.uint8)).save(os.path.join(out_sd, f"gt{i}.png"))

            mean_s = np.mean(ssims)
            print(f"  ✓ Step {st} 真实 Seen SSIM: {mean_s:.4f} (中位数: {np.median(ssims):.4f})")

    # 4. 重新绘制 seen_poster.png
    print("\n[4] 重新合成全景 seen_poster.png ...")
    out_poster = render_poster(
        results_dir=results_dir,
        set_name="seen",
        eval_csv="assets/eval_top10_seen_20.csv",
        train_csv="assets/train_top10_style23.csv",
        n_eval=20
    )
    print(f"🎉 修复完成！全新的真实 seen_poster.png 已更新至: {out_poster}")

if __name__ == "__main__":
    main()

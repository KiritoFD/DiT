#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/eval_ctrl_fame_cpu.py — 在 fame 评测集上复现 GT-skel ControlNet 的指标 (CPU)。

对照两组:
  base : cond=None      (退化为纯主模型)
  ctrl : cond=1px GT skel latent
流程: heun 50 步 + CFG 1.7 (与训练内 gpu_eval 同口径) -> VAE decode -> SSIM/MSE。

用法: N=60 python tools/eval_ctrl_fame_cpu.py
"""
import csv
import os
import sys
import time

import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

MAIN = ("data/archive/results_legacy/s21_fame_flow_v2/"
        "20260829-232329-s21-fame-flow-v2/checkpoints/0030000.pt")
CTRL = ("data/archive/results_legacy/ctrl_fame_1pix_v1/"
        "20260830-205652-fame-ctrl-skel-1px-v1/checkpoints/0050000.pt")
EVAL_CSV = "data/archive/legacy_csv/eval_fame_strict_clean.csv"
IMG_ROOT = "data/archive/legacy_images"
SKEL_DIR = "data/skel/final_skel_latents_fame_1px_v8"

N = int(os.environ.get("N", "60"))
CFG = float(os.environ.get("CFG", "1.7"))
STEPS = int(os.environ.get("STEPS", "50"))
BATCH = int(os.environ.get("BATCH", "8"))
SCALING = 0.18215


def build():
    from src.eval.model_io import build_model_from_args, apply_post_construction
    from src.model.legacy.controlnet import ControlNetDiT
    mc = torch.load(MAIN, map_location="cpu", weights_only=False)
    cc = torch.load(CTRL, map_location="cpu", weights_only=False)
    a_main = mc["args"]
    a_main = a_main if isinstance(a_main, dict) else vars(a_main)
    m = build_model_from_args(a_main, "cpu", vae_downscale=8)
    m = apply_post_construction(m, a_main, verbose=False)
    r = m.load_state_dict(mc["ema"], strict=False)
    print("  backbone missing=%d unexpected=%d" % (len(r.missing_keys), len(r.unexpected_keys)))
    a = cc["args"]
    a = a if isinstance(a, dict) else vars(a)
    wrap = ControlNetDiT(
        m, cond_in_channels=int(a.get("skel_cond_channels", 4) or 4),
        train_ctrl_only=True,
        ctrl_depth=int(a.get("ctrl_depth", 0) or 0) or None,
        ctrl_hidden=int(a.get("ctrl_hidden", 0) or 0) or None,
        ctrl_num_heads=int(a.get("ctrl_num_heads", 0) or 0) or None,
        injection=str(a.get("injection", "modulate")),
        null_cond=str(a.get("null_cond", "gaussian")),
    ).eval()
    r2 = wrap.load_state_dict(cc["ema"], strict=False)
    print("  ctrl missing=%d unexpected=%d" % (len(r2.missing_keys), len(r2.unexpected_keys)))
    return wrap


def main():
    t_all = time.time()
    print("=" * 78)
    print("[1] 构模 (CPU)")
    print("=" * 78)
    model = build()

    print()
    print("=" * 78)
    print("[2] 构建 eval cache (N=%d)" % N)
    print("=" * 78)
    from src.eval.inference import (build_diffusion, load_eval_vae,
                                    make_eval_cache, sample_latents)
    t0 = time.time()
    cache = make_eval_cache(EVAL_CSV, IMG_ROOT, None, 256, N, 8, 4, SCALING,
                            skel_latent_shards_dir=SKEL_DIR)
    print("  cache 就绪 %.1fs, n=%d, missing_skel=%d, skels_latent=%s"
          % (time.time() - t0, cache["n"], cache["missing_skel"],
             tuple(cache["skels_latent"].shape)))

    diff = build_diffusion(STEPS, "flow",
                           {"t_sampler": "logit_normal", "sampler": "heun", "shift": 1.0})
    vae = load_eval_vae("cpu")

    from skimage.metrics import structural_similarity as ssim_fn

    def decode(lat):
        with torch.no_grad():
            return vae.decode((lat.float() / SCALING)).sample.clamp(-1, 1)

    def gray(x01):
        return x01.mean(0).numpy()

    results = {}
    for tag, skel in (("base", None), ("ctrl", cache["skels_latent"])):
        print()
        print("=" * 78)
        print("[3] 采样: %s (cfg=%.2f, steps=%d)" % (tag, CFG, STEPS))
        print("=" * 78)
        t0 = time.time()
        lat = sample_latents(model, diff, cache["noise"], cache["conds"],
                             cfg_scale=CFG, batch=BATCH, device="cpu",
                             skel=skel, seed=0, hier_conds=None)
        print("  采样完成 %.1fs -> latents %s" % (time.time() - t0, tuple(lat.shape)))
        t0 = time.time()
        pred = decode(lat)
        print("  VAE decode %.1fs" % (time.time() - t0))

        ssims, mses = [], []
        for i in range(cache["n"]):
            g = gray((cache["gts"][i] + 1) / 2)
            p = gray((pred[i] + 1) / 2)
            ssims.append(ssim_fn(g, p, data_range=1.0))
            mses.append(float(((g - p) ** 2).mean()))
        results[tag] = (float(np.mean(ssims)), float(np.mean(mses)))
        print("  ==> %s: SSIM=%.4f  MSE=%.5f" % (tag, results[tag][0], results[tag][1]))

    print()
    print("=" * 78)
    print("[4] 汇总 (n=%d, fame eval 前 %d 条)" % (cache["n"], cache["n"]))
    print("=" * 78)
    b, c = results["base"], results["ctrl"]
    print("  base : SSIM=%.4f  MSE=%.5f" % b)
    print("  ctrl : SSIM=%.4f  MSE=%.5f" % c)
    print("  Δssim = %+.4f   Δmse = %+.5f" % (c[0] - b[0], c[1] - b[1]))
    print()
    print("  历史训练内记录 (step 42500/50000, n=100): base_ssim=0.4977 ctrl_ssim=0.7974")
    print("  总耗时 %.1f min" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
eval_s20_ctrl_monitor.py — s20 ControlNet 训练的旁路轻量 eval (单次).

用法: /opt/conda/bin/python tools/eval_s20_ctrl_monitor.py
  * 找 s20_ctrl_skel_flow_v2 系列 run 里最新的 *.pt.done ckpt
  * base (无 skel) vs ctrl (GT skel latent) 各采样 n=100, cfg 1.7, Heun 25 步 (=50 NFE)
  * 指标追加到 5script/eval_s20_ctrl_monitor.csv (小 batch, 不干扰主训练)
"""
import os
import sys
import csv
import glob
import json

import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from src.model.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import build_diffusion, sample_latents, _ssim, _mse

RESULTS_SERIES = "5script/results/s20_ctrl_skel_flow_v2"
EVAL_CSV = "5script/eval_strict_midclean.csv"
SKEL_SHARDS = "final_skel_latents_eval_v2"
IMG_ROOT = "final_imgs_256"
OUT_CSV = "5script/eval_s20_ctrl_monitor.csv"
N = 100
CFG = 1.7
STEPS = 25  # heun@25 = 50 NFE, 与训练 gpu_eval / s20 统一 eval 协议一致
BATCH = 8

# 与训练一致的 flow 参数（推理侧必须与训练侧一致）
FLOW_KWARGS = {"t_sampler": "logit_normal", "sampler": "heun", "shift": 1.0}


def latest_ckpt():
    runs = sorted(glob.glob(os.path.join(RESULTS_SERIES, "2026*")))
    for run in reversed(runs):
        done = sorted(glob.glob(os.path.join(run, "checkpoints", "*.pt.done")))
        if done:
            return run, done[-1][:-5]  # strip .done
    return None, None


def main():
    run, ckpt_path = latest_ckpt()
    if ckpt_path is None:
        print("[monitor] no ckpt yet")
        return
    step = os.path.basename(ckpt_path).split(".")[0]
    print(f"[monitor] run={os.path.basename(run)} ckpt={os.path.basename(ckpt_path)}")

    device = torch.device("cuda")
    # 主模型 = s20 预训练 best ckpt (ctrl ckpt 里没有 main.* 权重!)
    # arch 参数必须与 s20 预训练配置 (s20_midcommon_s_flow_v2.json) 完全一致
    main_ckpt = ("5script/results/s20_midcommon_s_flow_v2/"
                 "20260829-023132-s20-midcommon-s-flow-v2/checkpoints/0102500.pt")
    main_model = load_main_model(
        "DiT-2Cond-S/2", main_ckpt, device=device,
        num_calligraphers=1011, num_characters=35130,
        condition_fusion="factorized_add", callig_embed_dim=128,
        char_embed_dim=384, char_proj_mode="mlp", freeze_char_table=True,
        cond_drop_which_glyph_prob=0.75, diffusion_type="flow",
        norm_type="rms", mlp_type="swiglu", qk_norm=True,
        rope=True, rope_theta=100.0, attn_impl="sdpa")
    main_model.eval()
    ctrl = ControlNetDiT(main_model, cond_in_channels=4, train_ctrl_only=True).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck.get("ctrl") or ck
    sd = {k: v for k, v in sd.items()
          if k.startswith(("ctrl_encoder", "injections"))}
    miss, unexp = ctrl.load_state_dict(sd, strict=False)
    n_inj = len([k for k in sd if k.startswith("injections")])
    print(f"[monitor] ctrl loaded: keys={len(sd)} (injections={n_inj}) unexpected={len(unexp)}")
    ctrl.eval()
    diffusion = build_diffusion(STEPS, "flow", flow_kwargs=FLOW_KWARGS)

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(device).eval()

    from src.eval.inference import make_eval_cache
    cache = make_eval_cache(EVAL_CSV, IMG_ROOT, None, 256, N, 8, 4, 0.18215,
                            skel_latent_shards_dir=SKEL_SHARDS)
    assert cache["skels_latent"] is not None, "skel latent shards missing"
    sk = cache["skels_latent"]
    print(f"[monitor] skel cond: shape={tuple(sk.shape)} absmean={sk.abs().mean():.4f}")
    rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))[:cache["n"]]
    scripts = [r["script"] for r in rows]

    def sample_and_score(with_skel):
        skel = cache["skels_latent"] if with_skel else None
        model = ctrl if with_skel else ctrl  # base 也走 ctrl wrapper (cond=None 退化为主模型)
        latents = sample_latents(model, diffusion, cache["noise"], cache["conds"],
                                 CFG, BATCH, device, skel=skel, seed=0)
        preds = []
        with torch.no_grad():
            for i in range(0, cache["n"], 8):
                rec = vae.decode(latents[i:i+8].to(device) / 0.18215).sample.float().cpu()
                preds.append(((rec.clamp(-1, 1) + 1) / 2))
        preds = torch.cat(preds)
        mses, ssims = [], []
        per_script = {}
        for k in range(cache["n"]):
            p = preds[k].permute(1, 2, 0).numpy()
            g = ((cache["gts"][k].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
            mses.append(_mse(p, g))
            s = _ssim(p, g)
            ssims.append(s)
            per_script.setdefault(scripts[k], []).append(s)
        out = {"mse_x4": float(np.mean(mses)), "ssim": float(np.mean(ssims))}
        for s, v in per_script.items():
            out[f"ssim_{s}"] = float(np.mean(v))
        return out

    res_base = sample_and_score(False)
    print(f"[monitor] base  {res_base}", flush=True)
    res_ctrl = sample_and_score(True)
    print(f"[monitor] ctrl  {res_ctrl}", flush=True)

    header = ["step", "ckpt", "mse_base", "ssim_base", "mse_ctrl", "ssim_ctrl",
              "ssim_ctrl_楷", "ssim_ctrl_行", "ssim_ctrl_隶"]
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([int(step), os.path.basename(ckpt_path),
                    round(res_base["mse_x4"], 4), round(res_base["ssim"], 4),
                    round(res_ctrl["mse_x4"], 4), round(res_ctrl["ssim"], 4),
                    round(res_ctrl.get("ssim_楷", float("nan")), 4),
                    round(res_ctrl.get("ssim_行", float("nan")), 4),
                    round(res_ctrl.get("ssim_隶", float("nan")), 4)])
    print(f"[monitor] appended -> {OUT_CSV}")


if __name__ == "__main__":
    main()

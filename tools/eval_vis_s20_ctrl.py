# -*- coding: utf-8 -*-
"""
eval_vis_s20_ctrl.py — 独立可视化 eval: 加载最新 ctrl ckpt, 复用训练参数,
跑 run_ctrl_pair_eval 落盘 PNG (base + ctrl), 供拉回本地视觉检查.

用法(远程): /opt/conda/bin/python tools/eval_vis_s20_ctrl.py
  * 只读, 不干扰主训练; N 取小值省显存.
"""
import os
import sys
import glob

import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from src.model.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import build_diffusion, make_eval_cache
from src.eval.in_process_ctrl_eval import run_ctrl_pair_eval

RESULTS_SERIES = "5script/results/s20_ctrl_skel_flow_v2"
EVAL_CSV = "5script/eval_strict_midclean.csv"
# 默认 3px; 可用环境变量覆盖为 1px (如 SKEL_SHARDS=final_skel_latents_eval_1px)
SKEL_SHARDS = os.environ.get("SKEL_SHARDS", "final_skel_latents_eval_v2")
# 导出目录后缀, 区分 skel 粗细版本; 默认 skel3px, 1px 时置 skel1px
SKEL_TAG = os.environ.get("SKEL_TAG", "skel3px")
IMG_ROOT = "final_imgs_256"
MAIN_CKPT = ("5script/results/s20_midcommon_s_flow_v2/"
             "20260829-023132-s20-midcommon-s-flow-v2/checkpoints/0102500.pt")

# — 与训练一致的 flow 参数 (推理侧必须与训练侧一致) —
FLOW_KWARGS = {"t_sampler": "logit_normal", "sampler": "heun", "shift": 1.0}

# — 可视化采样参数 (与训练 gpu_eval 同款规模; heun@25 = 50 NFE) —
N = 100         # 与训练 gpu_eval_n 一致, 指标统计更有代表性
DDIM_STEPS = 25
CFG = 1.7
DIT_BATCH = 16
VAE_BATCH = 16


def latest_ckpt():
    runs = sorted(glob.glob(os.path.join(RESULTS_SERIES, "2026*")))
    for run in reversed(runs):
        done = sorted(glob.glob(os.path.join(run, "checkpoints", "*.pt.done")))
        if done:
            return run, done[-1][:-5]  # strip .done → 指向真实 .pt 权重文件
    return None, None


def main():
    run, ckpt_path = latest_ckpt()
    if ckpt_path is None:
        print("[eval_vis] no ckpt.done yet")
        return
    step = int(os.path.basename(ckpt_path).split(".")[0])
    print(f"[eval_vis] run={os.path.basename(run)} ckpt={os.path.basename(ckpt_path)}")

    device = torch.device("cuda")
    # arch 参数必须与 s20 预训练配置 (s20_midcommon_s_flow_v2.json) 完全一致
    main_model = load_main_model(
        "DiT-2Cond-S/2", MAIN_CKPT, device=device,
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
    sd = {k: v for k, v in sd.items() if k.startswith("ctrl_encoder")}
    miss, unexp = ctrl.load_state_dict(sd, strict=False)
    print(f"[eval_vis] ctrl loaded: keys={len(sd)} miss={len(miss)} unexp={len(unexp)}")
    ctrl.eval()

    diffusion = build_diffusion(DDIM_STEPS, "flow", flow_kwargs=FLOW_KWARGS)

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(device).eval()

    cache = make_eval_cache(EVAL_CSV, IMG_ROOT, None, 256, N, 8, 4, 0.18215,
                            skel_latent_shards_dir=SKEL_SHARDS)
    assert cache["skels_latent"] is not None, "skel latent shards missing"
    print(f"[eval_vis] skel cond: shape={tuple(cache['skels_latent'].shape)} "
          f"absmean={cache['skels_latent'].abs().mean():.4f}")

    checkpoint_dir = os.path.join(run, "checkpoints")
    # 输出到标准目录 checkpoints/eval_samples_ctrl/{step_tag}/{base,ctrl}/,
    # 与 eval_ctrl_metrics_daemon 的路径约定一致, 产出 marker 供其消费
    run_ctrl_pair_eval(ctrl, vae, diffusion, cache, device, step, checkpoint_dir,
                       ddim_steps=DDIM_STEPS, cfg_scale=CFG,
                       dit_batch=DIT_BATCH, vae_batch=VAE_BATCH)
    print(f"[eval_vis] saved under {checkpoint_dir}/eval_samples_ctrl/step{int(step):07d}/")


if __name__ == "__main__":
    main()

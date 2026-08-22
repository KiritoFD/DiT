# -*- coding: utf-8 -*-
"""
sample_controlnet.py — ControlNet 推理: skel 条件引导的 latent DiT 采样.

支持两种 skel 来源:
  1. 从真实图片 ID 加载 GT skel (final_skeleton_d3/<id>.png)
  2. 从标准字库加载: 给定 character_id, 用 build_std_skel 生成的标准骨架
     (路径: std_glyphs/skel/<char_id>.png)

流程:
  1. 加载主模型 + ControlNet 权重
  2. 加载 skel 条件 (GT 或标准字库)
  3. DDIM 采样 (50步, CFG)
  4. VAE decode latent → 像素图
  5. 保存结果

用法:
  # 从 GT 图片 skel 采样
  python tools/controlnet/sample_controlnet.py \
      --main-ckpt <main.pt> --ctrl-ckpt <ctrl.pt> \
      --skel-mode gt --img-id 12345 \
      --callig-id 5 --char-id 100 \
      -o /tmp/sample.png

  # 从标准字库 skel 采样
  python tools/controlnet/sample_controlnet.py \
      --main-ckpt <main.pt> --ctrl-ckpt <ctrl.pt> \
      --skel-mode std --char-id 100 \
      --callig-id 5 \
      -o /tmp/sample.png
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import math
from PIL import Image

from models import DiT_2Cond
from diffusion import create_diffusion
from tools.controlnet.controlnet_dit import ControlNetDiT


def load_main_and_ctrl(main_ckpt, ctrl_ckpt, device, num_callig=1011, num_char=35130):
    """加载主模型 + ControlNet 权重."""
    main_model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        hidden_size=384, depth=12, num_heads=6,
        num_calligraphers=num_callig, num_characters=num_char,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=256,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False, learn_sigma=True)
    # 主模型权重
    ck = torch.load(main_ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck.get("delta")
    main_model.load_state_dict(sd, strict=False)
    main_model.to(device).eval()
    print(f"[main] loaded {main_ckpt} (step={ck.get('train_steps')})")

    ctrl = ControlNetDiT(main_model, cond_in_channels=1, train_ctrl_only=True)
    # ControlNet 权重
    if ctrl_ckpt and os.path.exists(ctrl_ckpt):
        cck = torch.load(ctrl_ckpt, map_location="cpu", weights_only=False)
        ctrl_sd = cck.get("ema") or cck.get("ctrl")
        if ctrl_sd:
            ctrl.load_state_dict(ctrl_sd, strict=False)
            print(f"[ctrl] loaded {ctrl_ckpt} (step={cck.get('train_steps')})")
        else:
            print(f"[ctrl] WARNING: no ctrl/ema key in {ctrl_ckpt}, using zero-init (no control)")
    else:
        print(f"[ctrl] no ckpt, using zero-init (equivalent to base model)")
    ctrl.to(device).eval()
    return ctrl


def load_skel(sodel_mode, img_id=None, char_id=None, skel_root="final_skeleton_d3",
              std_skel_root="std_glyphs/skel", device="cpu"):
    """加载 skel 条件 (GT 或标准字库), 返回 (1,1,256,256) float32 0/1."""
    if lodel_mode == "gt":
        path = os.path.join(skel_root, f"{img_id}.png")
    elif lodel_mode == "std":
        path = os.path.join(std_skel_root, f"{char_id}.png")
    else:
        raise ValueError(f"Unknown skel_mode: {sodel_mode}")
    img = Image.open(path).convert("L")
    arr = (np.asarray(img, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,256,256)


@torch.no_grad()
def sample(ctrl_model, skel_cond, callig_id, char_id, device,
           steps=50, cfg_scale=4.0, seed=0):
    """DDIM 采样, 返回 latent (1,4,32,32)."""
    diffusion = create_diffusion(timestep_respacing="")
    torch.manual_seed(seed)
    z = torch.randn(1, 4, 32, 32, device=device)
    yc = torch.tensor([callig_id], dtype=torch.long, device=device)
    ych = torch.tensor([char_id], dtype=torch.long, device=device)
    cond = skel_cond.to(device)  # (1,1,256,256)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = diffusion.p_sample_loop(
            ctrl_model.forward_with_cfg, z.shape, z,
            model_kwargs=dict(y_callig=yc, y_char=ych, cfg_scale=cfg_scale, cond=cond),
            progress=False, device=device)
    return out.float().clamp(-1, 1)


@torch.no_grad()
def decode_latent(vae, latent, device):
    """VAE decode latent → 像素图 (256×256)."""
    z = latent / 0.18215
    with torch.autocast("cuda", dtype=torch.bfloat16):
        img = vae.decode(z).sample
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-ckpt", required=True)
    ap.add_argument("--ctrl-ckpt", default="")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--skel-mode", choices=["gt", "std"], default="gt")
    ap.add_argument("--img-id", type=int, default=None, help="GT 图片 ID (skel-mode=gt)")
    ap.add_argument("--char-id", type=int, required=True, help="字形 ID (用于 std skel 和采样条件)")
    ap.add_argument("--callig-id", type=int, required=True, help="书家 ID")
    ap.add_argument("--skel-root", default="final_skeleton_d3")
    ap.add_argument("--std-skel-root", default="std_glyphs/skel")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-calligraphers", type=int, default=1011)
    ap.add_argument("--num-characters", type=int, default=35130)
    ap.add_argument("-o", "--output", default="/tmp/ctrl_sample.png")
    args = ap.parse_args()

    device = torch.device("cuda")
    ctrl = load_main_and_ctrl(args.main_ckpt, args.ctrl_ckpt, device,
                              args.num_calligraphers, args.num_characters)

    # Load skel
    skel = load_skel(args.skel_mode, img_id=args.img_id, char_id=args.char_id,
                     skel_root=args.skel_root, std_skel_root=args.std_skel_root, device=device)
    print(f"[skel] mode={args.skel_mode} shape={skel.shape}")

    # Sample latent
    latent = sample(ctrl, skel, args.callig_id, args.char_id, device,
                    steps=args.steps, cfg_scale=args.cfg_scale, seed=args.seed)
    print(f"[sample] latent shape={latent.shape}")

    # VAE decode
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    img = decode_latent(vae, latent, device)
    img = ((img.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
    Image.fromarray(img).save(args.output)
    print(f"[save] {args.output}")

    # 同时保存 skel 条件用于对照
    skel_path = args.output.replace(".png", "_skel.png")
    skel_img = (skel.cpu().numpy()[0, 0] * 255).astype(np.uint8)
    Image.fromarray(skel_img).save(skel_path)
    print(f"[save] skel → {skel_path}")


if __name__ == "__main__":
    main()
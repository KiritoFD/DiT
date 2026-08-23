# -*- coding: utf-8 -*-
"""
eval_controlnet_cpu.py — CPU-only eval for ControlNet.

单步重建 (t=T_EVAL) MSE/SSIM, 对比:
  - base:  无 skel 条件 (cond=None)
  - ctrl:  有 skel 条件 (cond=GT skel)

不占 GPU, 不影响训练. 用法:
  python tools/controlnet/eval_controlnet_cpu.py --ctrl-ckpt <path.pt>
  python tools/controlnet/eval_controlnet_cpu.py  # 自动找最新 ckpt
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 强制 CPU
import sys
_s = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if _s not in sys.path:
    sys.path.insert(0, _s)
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import DiT_2Cond_models
from diffusion import create_diffusion
from controlnet_dit import ControlNetDiT, load_main_model

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
T_EVAL = 150


def _gaussian_window(window_size=11, sigma=1.5):
    g = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.reshape(1, 1, window_size, 1) @ g.reshape(1, 1, 1, window_size))


def _ssim(x, y, data_range=1.0, window_size=11, win=None):
    if x.shape[1] == 3:
        return sum(_ssim(x[:, i:i + 1], y[:, i:i + 1], data_range, window_size, win)
                   for i in range(3)) / 3
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, win, padding=window_size // 2)
    mu_y = F.conv2d(y, win, padding=window_size // 2)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx2 = F.conv2d(x * x, win, padding=window_size // 2) - mu_x2
    sy2 = F.conv2d(y * y, win, padding=window_size // 2) - mu_y2
    sxy = F.conv2d(x * y, win, padding=window_size // 2) - mu_xy
    m = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2))
    return float(m.mean().item())


def find_latest_ckpt(ckpt_dir):
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "**/*.pt"), recursive=True))
    return pts[-1] if pts else None


def load_eval_samples(csv_path, latent_dir, skel_root, img_root, n=50):
    """Load n samples: latent + skel + GT image + conditions."""
    import csv, re
    from latent_dataset import MCCDLatentDataset
    ds = MCCDLatentDataset(
        csv_file=csv_path, latent_shards_dir=latent_dir,
        img_root=img_root, skel_root=skel_root,
        image_size=256, load_skel=True, load_image=True,
        is_train=False, preload=False, structure_size=256)
    # Take first n (deterministic)
    samples = []
    for i in range(min(n, len(ds))):
        s = ds[i]
        samples.append(s)
    return samples


@torch.no_grad()
def eval_single_step(model, vae, diffusion, samples, device, t=T_EVAL, use_skel=False):
    """Single-step xstart reconstruction. Returns (mse, ssim)."""
    win = _gaussian_window(11, 1.5).to(device)
    mse_sum, ssim_sum, cnt = 0.0, 0.0, 0
    t_tensor = torch.full((1,), t, dtype=torch.long)

    for s in samples:
        x_lat = s['latent'].unsqueeze(0).to(device)          # (1,4,32,32)
        gt = s['image'].unsqueeze(0).to(device)               # (1,3,256,256) [-1,1]
        yc = torch.tensor([s['y_callig']], dtype=torch.long, device=device)
        yh = torch.tensor([s['y_char']], dtype=torch.long, device=device)

        mk = dict(y_callig=yc, y_char=yh)
        if use_skel:
            skel = s['skeleton'].unsqueeze(0).to(device).float()  # (1,1,256,256)
            mk['cond'] = skel

        ld = diffusion.training_losses(model, x_lat, t_tensor, mk)
        pred = ld["pred_xstart"]                                # (1,4,32,32)
        decoded = vae.decode(pred / 0.18215).sample             # (1,3,256,256)

        mse_sum += F.mse_loss(decoded, gt).item()
        d01 = (decoded + 1) / 2
        g01 = (gt + 1) / 2
        ssim_sum += _ssim(d01, g01, 1.0, 11, win)
        cnt += 1

    return mse_sum / cnt, ssim_sum / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-ckpt",
                    default="5script/results/s6_top6_diffonly/20260820-191536-s6-top6-diffonly-resume/checkpoints/0195000.pt")
    ap.add_argument("--ctrl-ckpt", default="",
                    help="ControlNet ckpt. Empty = auto-find latest.")
    ap.add_argument("--ctrl-dir", default="5script/results/ctrl_skel",
                    help="Search dir for controlnet ckpts")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--csv", default="5script/train_top6.csv")
    ap.add_argument("--latent-dir", default="final_latents")
    ap.add_argument("--skel-root", default="final_skeleton_d3")
    ap.add_argument("--img-root", default="final_images")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--t", type=int, default=T_EVAL)
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"[eval] device={device}, n={args.n}, t={args.t}")

    # Find controlnet ckpt
    ctrl_ckpt = args.ctrl_ckpt
    if not ctrl_ckpt:
        ctrl_ckpt = find_latest_ckpt(args.ctrl_dir)
    if not ctrl_ckpt:
        print("[eval] No controlnet ckpt found. Evaluating base model only.")
        ctrl_step = 0
    else:
        ctrl_step = int(os.path.basename(ctrl_ckpt).replace('.pt', ''))
        print(f"[eval] ctrl ckpt: {ctrl_ckpt} (step={ctrl_step})")

    # Load main model
    print("[eval] loading main model...")
    main_model = load_main_model(
        model_name="DiT-2Cond-S/2", ckpt_path=args.main_ckpt,
        device=device, num_calligraphers=1011, num_characters=35130,
        condition_fusion="factorized_add", callig_embed_dim=128,
        char_embed_dim=256, cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False)
    main_model.eval()

    # Load controlnet
    ctrl = ControlNetDiT(main_model, cond_in_channels=1, train_ctrl_only=True).to(device)
    ctrl.eval()
    if ctrl_ckpt and os.path.exists(ctrl_ckpt):
        ck = torch.load(ctrl_ckpt, map_location="cpu", weights_only=False)
        ctrl_sd = ck.get("ema") or ck.get("ctrl")
        if ctrl_sd:
            ctrl_keys = {k: v for k, v in ctrl_sd.items() if k.startswith("ctrl_encoder")}
            ctrl.load_state_dict(ctrl_keys, strict=False)
            print(f"[eval] loaded ctrl ({len(ctrl_keys)} keys)")

    # Load VAE
    print("[eval] loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()

    # Load eval samples
    print("[eval] loading samples...")
    samples = load_eval_samples(args.csv, args.latent_dir, args.skel_root,
                                args.img_root, n=args.n)
    print(f"[eval] {len(samples)} samples loaded")

    diffusion = create_diffusion(timestep_respacing="")

    # Eval: base (no skel)
    print("\n=== Base (no skel) ===")
    mse_base, ssim_base = eval_single_step(main_model, vae, diffusion,
                                           samples, device, t=args.t, use_skel=False)
    print(f"  MSE={mse_base:.6f}  SSIM={ssim_base:.6f}")

    # Eval: controlnet (with skel)
    print("\n=== ControlNet (with GT skel) ===")
    mse_ctrl, ssim_ctrl = eval_single_step(ctrl, vae, diffusion,
                                            samples, device, t=args.t, use_skel=True)
    print(f"  MSE={mse_ctrl:.6f}  SSIM={ssim_ctrl:.6f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Step={ctrl_step}  n={args.n}  t={args.t}")
    print(f"{'':>20} {'MSE':>10} {'SSIM':>10}")
    print(f"{'base (no skel)':>20} {mse_base:>10.6f} {ssim_base:>10.6f}")
    print(f"{'ctrl (GT skel)':>20} {mse_ctrl:>10.6f} {ssim_ctrl:>10.6f}")
    print(f"{'delta':>20} {mse_ctrl-mse_base:>+10.6f} {ssim_ctrl-ssim_base:>+10.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

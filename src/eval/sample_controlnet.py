# -*- coding: utf-8 -*-
"""
sample_controlnet.py 鈥?ControlNet 鎺ㄧ悊: skel 鏉′欢寮曞鐨?latent DiT 閲囨牱.

鏀寔涓ょ skel 鏉ユ簮:
  1. GT skel: 浠庣湡瀹炲浘鐗?ID 鍔犺浇 (final_skeleton_d3/<id>.png)
  2. 鏍囧噯瀛楀簱 skel: 缁欏畾 character_id, 鐢ㄦ爣鍑嗗瓧搴撶敓鎴愮殑鏍囧噯楠ㄦ灦
     (璺緞: std_glyphs/skel/<char_id>.png)

娴佺▼:
  1. 鍔犺浇涓绘ā鍨?+ ControlNet 鏉冮噸
  2. 鍔犺浇 skel 鏉′欢 (GT 鎴栨爣鍑嗗瓧搴?
  3. DDIM 閲囨牱 (50姝? CFG on callig/char, skel 濮嬬粓鎻愪緵)
  4. VAE decode latent 鈫?鍍忕礌鍥?(256脳256)
  5. 淇濆瓨缁撴灉 + skel 鏉′欢瀵圭収鍥?
鐢ㄦ硶:
  # 浠?GT 鍥剧墖 skel 閲囨牱
  python tools/controlnet/sample_controlnet.py \
      --main-ckpt <main.pt> --ctrl-ckpt <ctrl.pt> \
      --skel-mode gt --img-id 12345 \
      --callig-id 5 --char-id 100 \
      -o /tmp/sample.png

  # 浠庢爣鍑嗗瓧搴?skel 閲囨牱
  python tools/controlnet/sample_controlnet.py \
      --main-ckpt <main.pt> --ctrl-ckpt <ctrl.pt> \
      --skel-mode std --char-id 100 \
      --callig-id 5 \
      -o /tmp/sample.png
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
_s = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if _s not in sys.path:
    sys.path.insert(0, _s)
import argparse
import numpy as np
import torch
from PIL import Image

from src.model import DiT_2Cond_models
from src.loss import create_diffusion
from src.model.controlnet import ControlNetDiT

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding="utf-8")


def load_main_and_ctrl(args, device):
    """鍔犺浇涓绘ā鍨?+ ControlNet 鏉冮噸."""
    main_model = DiT_2Cond_models[args.model](
        num_calligraphers=args.num_calligraphers, num_characters=args.num_characters,
        condition_fusion="factorized_add",
        callig_embed_dim=args.callig_embed_dim, char_embed_dim=args.char_embed_dim,
        cond_drop_all_prob=args.cond_drop_all_prob, cond_drop_one_prob=args.cond_drop_one_prob,
        use_checkpoint=False, learn_sigma=True)
    ck = torch.load(args.main_ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck.get("delta")
    main_model.load_state_dict(sd, strict=False)
    main_model.to(device).eval()
    print(f"[main] loaded {os.path.basename(args.main_ckpt)} (step={ck.get('train_steps')})")

    ctrl = ControlNetDiT(main_model, cond_in_channels=1, train_ctrl_only=True)
    if args.ctrl_ckpt and os.path.exists(args.ctrl_ckpt):
        cck = torch.load(args.ctrl_ckpt, map_location="cpu", weights_only=False)
        ctrl_sd = cck.get("ema") or cck.get("ctrl")
        if ctrl_sd:
            # Only load ctrl_encoder keys (skip main model keys)
            ctrl_keys = {k: v for k, v in ctrl_sd.items() if k.startswith("ctrl_encoder")}
            ctrl.load_state_dict(ctrl_keys, strict=False)
            print(f"[ctrl] loaded {os.path.basename(args.ctrl_ckpt)} "
                  f"({len(ctrl_keys)} ctrl keys, step={cck.get('train_steps')})")
        else:
            print(f"[ctrl] WARNING: no ctrl/ema key in {args.ctrl_ckpt}, using zero-init (no control)")
    else:
        print(f"[ctrl] no ckpt, using zero-init (equivalent to base model)")
    ctrl.to(device).eval()
    return ctrl


def load_skel(skel_mode, img_id=None, char_id=None,
              skel_root="final_skeleton_d3", std_skel_root="std_glyphs/skel", device="cpu"):
    """鍔犺浇 skel 鏉′欢 (GT 鎴栨爣鍑嗗瓧搴?, 杩斿洖 (1,1,256,256) float32 0/1."""
    if skel_mode == "gt":
        if img_id is None:
            raise ValueError("skel-mode=gt requires --img-id")
        path = os.path.join(skel_root, f"{img_id}.png")
    elif skel_mode == "std":
        if char_id is None:
            raise ValueError("skel-mode=std requires --char-id")
        path = os.path.join(std_skel_root, f"{char_id}.png")
    else:
        raise ValueError(f"Unknown skel_mode: {skel_mode}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"skel not found: {path}")
    img = Image.open(path).convert("L")
    arr = (np.asarray(img, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,256,256)


@torch.no_grad()
def sample(ctrl_model, skel_cond, callig_id, char_id, device,
           steps=50, cfg_scale=4.0, seed=0):
    """DDIM 閲囨牱, 杩斿洖 latent (1,4,32,32)."""
    diffusion = create_diffusion(timestep_respacing=str(steps))
    torch.manual_seed(seed)
    z = torch.randn(1, 4, 32, 32, device=device)
    yc = torch.tensor([callig_id], dtype=torch.long, device=device)
    ych = torch.tensor([char_id], dtype=torch.long, device=device)
    cond = skel_cond.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latent = diffusion.ddim_sample_loop(
            ctrl_model.forward_with_cfg, z.shape, z,
            clip_denoised=False,
            model_kwargs=dict(y_callig=yc, y_char=ych, cfg_scale=cfg_scale, cond=cond),
            progress=True, device=device)
    return latent.float()


@torch.no_grad()
def decode_latent(vae_path, latent, device):
    """VAE decode latent 鈫?鍍忕礌鍥?(1,3,256,256) in [-1,1]."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
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
    ap.add_argument("--img-id", type=int, default=None)
    ap.add_argument("--char-id", type=int, required=True)
    ap.add_argument("--callig-id", type=int, required=True)
    ap.add_argument("--skel-root", default="final_skeleton_d3")
    ap.add_argument("--std-skel-root", default="std_glyphs/skel")
    ap.add_argument("--model", default="DiT-2Cond-S/2")
    ap.add_argument("--num-calligraphers", type=int, default=1011)
    ap.add_argument("--num-characters", type=int, default=35130)
    ap.add_argument("--callig-embed-dim", type=int, default=128)
    ap.add_argument("--char-embed-dim", type=int, default=256)
    ap.add_argument("--cond-drop-all-prob", type=float, default=0.05)
    ap.add_argument("--cond-drop-one-prob", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", default="/tmp/ctrl_sample.png")
    args = ap.parse_args()

    device = torch.device("cuda")
    ctrl = load_main_and_ctrl(args, device)

    skel = load_skel(args.skel_mode, img_id=args.img_id, char_id=args.char_id,
                     skel_root=args.skel_root, std_skel_root=args.std_skel_root, device=device)
    print(f"[skel] mode={args.skel_mode} shape={skel.shape} density={skel.mean().item():.3f}")

    latent = sample(ctrl, skel, args.callig_id, args.char_id, device,
                    steps=args.steps, cfg_scale=args.cfg_scale, seed=args.seed)
    print(f"[sample] latent shape={latent.shape}")

    img = decode_latent(args.vae_path, latent, device)
    img_np = ((img.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
    Image.fromarray(img_np).save(args.output)
    print(f"[save] {args.output}")

    # skel 鏉′欢瀵圭収鍥?    skel_path = args.output.replace(".png", "_skel.png")
    skel_img = (skel.cpu().numpy()[0, 0] * 255).astype(np.uint8)
    Image.fromarray(skel_img).save(skel_path)
    print(f"[save] skel 鈫?{skel_path}")


if __name__ == "__main__":
    main()

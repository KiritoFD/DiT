# -*- coding: utf-8 -*-
"""FontDiffuser-Latent sampling on the top10 eval protocol.

DPM-Solver++ (20 steps, cfg 7.5 — official sampler/settings) in latent space,
then ONE VAE decode at the very end (project sd-vae-ft-ema). Output
{slot}__{char}.png at 256.

Run from baseline/FontDiffuser:
  python ../adapter/fontdiffuser/sample_latent.py \
    --ckpt_dir outputs/latent/global_step_150000 --eval_tag strict84 \
    --save_image_dir ../results/fontdiffuser_latent/strict84
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from build_latent import (FontDiffuserModelLatent, build_content_encoder_latent,  # noqa: E402
                          build_style_encoder_latent, build_unet_latent)
from latent_common import (DIT_ROOT, SCALING, blank_latent,  # noqa: E402
                           build_vae, decode_latents)
from latent_dataset import FontLatentDataset  # noqa: E402

EVAL = f"{DIT_ROOT}/baseline/data/eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--eval_tag", default="strict84", choices=["strict84", "seen20"])
    ap.add_argument("--save_image_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    class A:
        resolution = 32
        unet_channels = (64, 128, 256, 512)
        content_encoder_downsample_size = 3
        channel_attn = True
        content_start_channel = 64
        style_start_channel = 64
    cfg = A()

    unet = build_unet_latent(cfg)
    se = build_style_encoder_latent(cfg)
    ce = build_content_encoder_latent(cfg)
    if not args.random_init:
        unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location="cpu"))
        se.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location="cpu"))
        ce.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location="cpu"))
    model = FontDiffuserModelLatent(unet=unet, style_encoder=se,
                                    content_encoder=ce).to(args.device).eval()

    vae = build_vae(args.device)
    drop = blank_latent(args.device)

    # eval items -> latents (targets come from shards; refs by ref char)
    ds = FontLatentDataset(device=args.device)
    with open(f"{EVAL}/refs.json", encoding="utf-8") as f:
        refs = json.load(f)
    rows = ds.rows
    slot_of = {}
    for r in rows:
        slot_of[(r["slot_name"], r["character"])] = int(r["img_id"])

    items = sorted((s, c) for s, d in refs.items() for c in d)
    if args.limit:
        items = items[:args.limit]

    # DPM-Solver++ scheduler over the training betas
    from diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler)
    sched = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="scaled_linear")
    sched.set_timesteps(args.num_inference_steps, device=args.device)

    @torch.no_grad()
    def eps_fn(x_t, t, style, content):
        t_t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_c, _ = model(x_t=x_t, timesteps=t_t, style_images=style,
                         content_images=content,
                         content_encoder_downsample_size=3)
        eps_u, _ = model(x_t=x_t, timesteps=t_t, style_images=drop.expand_as(x_t),
                         content_images=drop.expand_as(x_t),
                         content_encoder_downsample_size=3)
        return eps_u + args.guidance_scale * (eps_c - eps_u)

    out_dir = args.save_image_dir
    os.makedirs(out_dir, exist_ok=True)

    bs = args.batch_size
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        content = torch.stack([ds.con.get(ord(ch)) for _, ch in chunk]).to(args.device)
        style = torch.stack([
            ds.tgt.get(slot_of[(slot, refs[slot][ch][0])]) for slot, ch in chunk
        ]).to(args.device)
        lat = torch.randn(len(chunk), 4, 32, 32, device=args.device)

        for t in sched.timesteps:
            eps = eps_fn(lat, t, style, content)
            lat = sched.step(eps, t, lat).prev_sample

        imgs = decode_latents(vae, lat)                    # (B,3,256,256) [-1,1]
        for (slot, ch), im in zip(chunk, imgs):
            arr = ((im.clamp(-1, 1) + 1) / 2 * 255).byte()
            arr = arr.permute(1, 2, 0).numpy()
            Image.fromarray(arr).convert("L").save(f"{out_dir}/{slot}__{ch}.png")
        print(f"[{i + len(chunk)}/{len(items)}] saved", flush=True)


if __name__ == "__main__":
    main()

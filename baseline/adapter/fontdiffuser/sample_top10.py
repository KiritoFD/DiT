# -*- coding: utf-8 -*-
"""Batch sampling for FontDiffuser on top10 eval protocol.

For every eval item (slot,char) in refs.json: content = Deng render of char,
style ref = refs[slot][char][0] (FontDiffuser is one-shot by design).
Output: {out_dir}/{slot}__{char}.png

Usage (from baseline/FontDiffuser):
  python ../adapter/fontdiffuser/sample_top10.py \
      --ckpt_dir outputs/top10_phase1_256/global_step_100000 \
      --eval_tag strict84 --save_image_dir ../results/fontdiffuser/strict84
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # FontDiffuser repo

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from PIL import Image  # noqa: E402

from src import (FontDiffuserModelDPM, build_ddpm_scheduler, build_unet,  # noqa: E402
                 build_content_encoder, build_style_encoder)
from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline  # noqa: E402

DIT = "/root/Workspace/xy/DiT"
CONTENT = os.path.join(DIT, "baseline/data/content_font/deng")
EVAL = os.path.join(DIT, "baseline/data/eval")
FD_TGT = os.path.join(DIT, "baseline/data/fontdiffuser/train/TargetImage")


def get_args():
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--eval_tag", default="strict84", choices=["strict84", "seen20"])
    parser.add_argument("--save_image_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--random_init", action="store_true",
                        help="skip ckpt loading (inference-path probe only)")
    parser.add_argument("--limit", type=int, default=0, help="only first N items")
    args = parser.parse_args()
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)
    return args


def main():
    args = get_args()

    unet = build_unet(args)
    style_encoder = build_style_encoder(args)
    content_encoder = build_content_encoder(args)
    if not args.random_init:
        unet.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "unet.pth"), map_location="cpu"))
        style_encoder.load_state_dict(
            torch.load(os.path.join(args.ckpt_dir, "style_encoder.pth"), map_location="cpu"))
        content_encoder.load_state_dict(
            torch.load(os.path.join(args.ckpt_dir, "content_encoder.pth"), map_location="cpu"))
    model = FontDiffuserModelDPM(unet=unet, style_encoder=style_encoder,
                                 content_encoder=content_encoder).to(args.device).eval()
    scheduler = build_ddpm_scheduler(args)
    pipe = FontDiffuserDPMPipeline(
        model=model, ddpm_train_scheduler=scheduler, model_type=args.model_type,
        guidance_type=args.guidance_type, guidance_scale=args.guidance_scale)

    c_tf = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    s_tf = transforms.Compose([
        transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    with open(os.path.join(EVAL, "refs.json"), encoding="utf-8") as f:
        refs = json.load(f)
    items = [(slot, ch) for slot, d in refs.items() for ch in d]
    if args.limit:
        items = items[:args.limit]

    out_dir = args.save_image_dir
    os.makedirs(out_dir, exist_ok=True)

    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            content_batch, style_batch = [], []
            for slot, ch in chunk:
                c = Image.open(os.path.join(CONTENT, f"{ch}.png")).convert("RGB")
                ref_ch = refs[slot][ch][0]
                s = Image.open(os.path.join(FD_TGT, slot, f"{slot}+{ref_ch}.png")).convert("RGB")
                content_batch.append(c_tf(c))
                style_batch.append(s_tf(s))
            content_batch = torch.stack(content_batch).to(args.device)
            style_batch = torch.stack(style_batch).to(args.device)

            images = pipe.generate(
                content_images=content_batch,
                style_images=style_batch,
                batch_size=len(chunk),
                order=args.order,
                num_inference_step=args.num_inference_steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start,
                t_end=args.t_end,
                dm_size=(args.resolution, args.resolution),
                algorithm_type=args.algorithm_type,
                skip_type=args.skip_type,
                method=args.method,
                correcting_x0_fn=args.correcting_x0_fn)

            for (slot, ch), im in zip(chunk, images):
                im.save(os.path.join(out_dir, f"{slot}__{ch}.png"))
            print(f"[{i + len(chunk)}/{len(items)}] saved", flush=True)


if __name__ == "__main__":
    main()

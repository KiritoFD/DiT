#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证: 标准字形图经 sd-vae-ft-ema encode 成 latent 是否合理(4,32,32)。
对比: 直接降采样 vs VAE encode 两种条件表示的差异。
"""
import os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

VAE_PATH = r"G:\GitHub\DiT\pretrained_models\sd-vae-ft-ema"

def render(ch, font_path, size=256):
    f = ImageFont.truetype(font_path, int(size*0.85))
    img = Image.new("RGB", (size,size), (255,255,255))
    d = ImageDraw.Draw(img)
    d.text((size*0.03,size*0.03), ch, font=f, fill=(0,0,0))
    return img

def main():
    from diffusers.models import AutoencoderKL
    torch.manual_seed(0)
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to("cpu").eval()
    print("VAE loaded")
    for ch, fp, label in [("永", r"C:\Windows\Fonts\simkai.ttf","楷-永"),
                          ("之", r"C:\Windows\Fonts\simkai.ttf","楷-之"),
                          ("一", r"C:\Windows\Fonts\SIMLI.TTF","隶-一")]:
        img = render(ch, fp)
        img_np = np.asarray(img).astype(np.float32)/127.5-1.0  # [-1,1]
        t = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0)
        # VAE encode
        with torch.no_grad():
            z = vae.encode(t).latent_dist.sample().mul_(0.18215)
        print(f"{label}: latent shape={tuple(z.shape)} std={z.std().item():.3f} mean={z.mean().item():.3f}")
        # 直接降采样
        small = torch.nn.functional.interpolate(t, size=(32,32), mode='bilinear', align_corners=False)
        print(f"  (直接降采样验证: 32x32 shape={tuple(small.shape)})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归因「笔画破碎」：VAE round-trip vs 模型生成。

对 eval_low_ink.csv 里 ink_ssim 最差的样本，拼三列对比:
  列1 = GT 原图 | 列2 = VAE encode→decode 往返(不经任何扩散模型) | 列3 = 模型生成
并算 ink_iou(GT,VAE往返) 与 ink_iou(GT,模型gen)。
判读: 若列2 就已经断笔 / ink_iou(GT,往返) 明显 <1 -> 破碎主要来自 f8 VAE latent 太粗;
      若列2 干净、只有列3 碎 -> 破碎来自扩散模型(训练/容量/条件), 不是 VAE。
"""
import csv, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
DEV = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
GEN_RUN = sys.argv[2] if len(sys.argv) > 2 else "v15b_supcon__0070000"
SZ, PAD, TH = 150, 8, 24

# ---- VAE ----
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("data/pretrained/sd-vae-ft-ema").to(DEV).eval()
for p in vae.parameters():
    p.requires_grad_(False)

def load256(p):
    im = Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im).astype(np.float32) / 127.5 - 1.0)
    return x.permute(2, 0, 1)[None].to(DEV)

@torch.no_grad()
def roundtrip(x):
    z = vae.encode(x).latent_dist.mode()
    return vae.decode(z).sample

def to_np(x):  # (1,3,H,W) [-1,1] -> HWC uint8
    a = ((x[0].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return a

def ink_iou(a, b, thr=128, dil=1):
    def mask(u8):
        g = u8.mean(2) < thr
        if dil:
            g = F.max_pool2d(g[None, None].float(), 2*dil+1, stride=1, padding=dil)[0, 0] > 0
        return g.numpy()
    ma, mb = mask(a), mask(b)
    inter = (ma & mb).sum(); union = (ma | mb).sum()
    return float(inter) / float(union) if union else 1.0

rows = list(csv.DictReader(open("assets/eval_low_ink.csv", encoding="utf-8")))[:N]
gd = f"assets/ink_eval/{GEN_RUN}__strict"
gen_files = sorted(glob.glob(os.path.join(gd, "g[0-9]*.png")),
                   key=lambda p: int(os.path.basename(p)[1:-4]))

try:
    font = ImageFont.truetype("_fonts/msyh.ttc", 15)
except Exception:
    font = ImageFont.load_default()

COLS = 3
W = COLS * SZ + (COLS + 1) * PAD
H = N * (SZ + TH + PAD) + 46
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
d.text((PAD, 6), f"列1=GT  列2=VAE往返(无模型)  列3=模型gen[{GEN_RUN}]   "
                 f"标注: 往返IoU / genIoU   (ink_ssim最差{N})", font=font, fill="black")

for i, r in enumerate(rows):
    iid = int(r["img_id"]); idx = int(r["idx"])
    y0 = 46 + i * (SZ + TH + PAD)
    gt = load256(f"data/50k/imgs/{iid:06d}.png")
    rt = roundtrip(gt)
    gt_np, rt_np = to_np(gt), to_np(rt)
    genp = gen_files[idx] if idx < len(gen_files) else None
    gen_np = np.asarray(Image.open(genp).convert("RGB").resize((256, 256))) if genp else np.full((256,256,3),255,np.uint8)
    iou_rt = ink_iou(gt_np, rt_np); iou_gen = ink_iou(gt_np, gen_np)
    for k, arr in enumerate([gt_np, rt_np, gen_np]):
        x0 = PAD + k * (SZ + PAD)
        canvas.paste(Image.fromarray(arr).resize((SZ, SZ), Image.BICUBIC), (x0, y0))
        d.rectangle([x0, y0, x0+SZ, y0+SZ], outline="gray")
    d.text((PAD, y0+SZ+3), f"{r['char']}({r['script']}) ink_ssim={r['ink']} "
                           f"| 往返IoU={iou_rt:.2f} genIoU={iou_gen:.2f}", font=font, fill="black")

out = "assets/poster_vae_roundtrip.png"
canvas.save(out)
# 汇总
ious = []
print(f"-> {out}  {canvas.size}")

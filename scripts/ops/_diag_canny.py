#!/usr/bin/env python
# -*- coding: utf-8 -*-
# canny 飞掉的排查: 核对 EdgeGradientLoss 输入值域
#   1) GT 图 batch['image'] 的数值范围
#   2) VAE decode 输出 x0_pred 的数值范围
#   3) 同一个 batch 上 EdgeGradientLoss / SobelCannyLoss 的实际数值量级
import os, sys
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
import torch
torch.set_num_threads(8)
import json, numpy as np

cfg = json.load(open("exp_s6_top6_struct_fp32.json"))
from latent_dataset import MCCDLatentDataset

print("loading dataset (preload=False) ...", flush=True)
ds = MCCDLatentDataset(
    csv_file=cfg["data_csv"], latent_shards_dir="final_latents",
    img_root="final_imgs_256",
    canny_root="final_canny", skel_root="final_skeleton",
    load_canny=True, load_skel=True, preload=False)
print("dataset len:", len(ds), flush=True)

# 抽样 4 张看值域
import random
random.seed(0)
idxs = random.sample(range(len(ds)), 4)
for i in idxs:
    b = ds[i]
    for k in ["image", "latent", "canny", "skeleton"]:
        if k in b:
            v = b[k]
            if isinstance(v, torch.Tensor):
                print(f"  sample {i} {k}: shape={tuple(v.shape)} min={float(v.min()):.4f} max={float(v.max()):.4f} mean={float(v.mean()):.4f}", flush=True)
            else:
                print(f"  sample {i} {k}: type={type(v)}", flush=True)

# VAE decode 值域
print("loading VAE (CPU) ...", flush=True)
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained(cfg["vae_path"])
vae.eval()
lat = ds[idxs[0]]["latent"].unsqueeze(0).float()  # (1,4,32,32)
with torch.no_grad():
    out = vae.decode(lat / 0.18215).sample
print(f"VAE decode: shape={tuple(out.shape)} min={float(out.min()):.4f} max={float(out.max()):.4f} mean={float(out.mean()):.4f}", flush=True)

# 真实梯度 loss 量级
gray_pred = (0.299*out[:,0:1] + 0.587*out[:,1:2] + 0.114*out[:,2:3] + 1.0)/2.0
x = ds[idxs[0]]["image"].unsqueeze(0).float()
gray_gt = (0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3] + 1.0)/2.0
sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], dtype=torch.float32).view(1,1,3,3)
sobel_y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], dtype=torch.float32).view(1,1,3,3)
def grad_mag(g):
    gx = torch.nn.functional.conv2d(g, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(g, sobel_y, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-6)
gm_pred, gm_gt = grad_mag(gray_pred), grad_mag(gray_gt)
print(f"grad pred: min={float(gm_pred.min()):.5f} max={float(gm_pred.max()):.5f} mean={float(gm_pred.mean()):.5f}", flush=True)
print(f"grad gt  : min={float(gm_gt.min()):.5f} max={float(gm_gt.max()):.5f} mean={float(gm_gt.mean()):.5f}", flush=True)
print(f"EdgeGradient L1(pred,gt): {float(torch.nn.functional.l1_loss(gm_pred, gm_gt)):.5f}", flush=True)
canny_gt = ds[idxs[0]]["canny"].unsqueeze(0).float()
print(f"canny_gt: shape={tuple(canny_gt.shape)} min={float(canny_gt.min()):.3f} max={float(canny_gt.max()):.3f} frac1={float((canny_gt>0.5).float().mean()):.4f}", flush=True)
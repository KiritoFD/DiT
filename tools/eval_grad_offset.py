# -*- coding: utf-8 -*-
"""关键补充测试: 当 latent 偏离 GT 时 (模拟 DiT pred_xstart), 梯度幅度如何."""
import os, sys
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, "/root/Workspace/xy/DiT/src")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from latent_dataset import MCCDLatentDataset

sys.stdout.reconfigure(encoding="utf-8")
DEVICE = "cuda"
OUT_DIR = "/root/Workspace/xy/DiT/5script/results/struct_decoder_gpu"

class StructDecoder(nn.Module):
    def __init__(self, in_ch=4, base=64, depth=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU())
        blocks = []
        for _ in range(depth):
            blocks += [nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, base, 3, padding=1)]
        self.body = nn.Sequential(*blocks)
        self.up1 = nn.Sequential(nn.Conv2d(base, base*4, 1), nn.PixelShuffle(2), nn.SiLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.Conv2d(base, base*4, 1), nn.PixelShuffle(2), nn.SiLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.up3 = nn.Sequential(nn.Conv2d(base, base*4, 1), nn.PixelShuffle(2), nn.SiLU())
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        f = self.stem(x); f = self.body(f)
        f = self.up1(f); f = self.up2(f); f = self.up3(f)
        return self.head(f)


print("=== 加载数据 ===", flush=True)
ds = MCCDLatentDataset(
    csv_file="5script/train_full.csv", latent_shards_dir="final_latents",
    img_root=None, canny_root="final_canny_d3", skel_root="final_skeleton_d3",
    image_size=256, load_canny=True, load_skel=True,
    is_train=True, preload=True, load_image=False,
    structure_size=256, num_preload_workers=16)
n = min(2000, len(ds))
lat_v = torch.from_numpy(ds._latents[:n].copy()).to(DEVICE)
sk_v = torch.from_numpy((ds._skels[:n] > 127).astype(np.float32)).unsqueeze(1).to(DEVICE)

print("=== 加载 skel_best ===", flush=True)
sk_net = StructDecoder(in_ch=4, base=64, depth=6).to(DEVICE)
ck = torch.load(os.path.join(OUT_DIR, "skel_best.pt"), map_location="cpu", weights_only=False)
sk_net.load_state_dict(ck["model"])
sk_net.eval()

print("\n=== 关键测试: latent 偏离 GT 时的梯度幅度 ===", flush=True)
print("模拟 DiT pred_xstart 在不同噪声水平下偏离 GT latent:", flush=True)
print(f"{'偏移σ':>8s} {'skel IoU':>10s} {'BCE loss':>10s} {'grad_norm':>12s} {'latent_norm':>14s} {'ratio':>10s}", flush=True)

BS = 50
for offset_sigma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
    # latent = GT + 偏移 (模拟 pred_xstart ≠ GT)
    lat_offset = lat_v[:BS] + torch.randn_like(lat_v[:BS]) * offset_sigma
    lat_g = lat_offset.clone().requires_grad_(True)

    sk_logit = sk_net(lat_g)
    sk_loss = F.binary_cross_entropy_with_logits(sk_logit, sk_v[:BS])
    g = torch.autograd.grad(sk_loss, lat_g)[0]

    g_norm = g.view(BS, -1).norm(dim=1).mean().item()
    lat_norm = lat_offset.view(BS, -1).norm(dim=1).mean().item()
    ratio = g_norm / (lat_norm + 1e-8)

    with torch.no_grad():
        sk_p = sk_net(lat_offset).sigmoid()
        p = (sk_p > 0.5).float()
        tp = (p * sk_v[:BS]).sum(); fn = ((1-p)*sk_v[:BS]).sum(); fp = (p*(1-sk_v[:BS])).sum()
        iou = (tp/(tp+fp+fn+1e-6)).item()

    print(f"{offset_sigma:>8.2f} {iou:>10.4f} {sk_loss.item():>10.4f} {g_norm:>12.6f} {lat_norm:>14.4f} {ratio:>10.6f}", flush=True)

print("\n=== 对比: diff loss (eps prediction) 的梯度幅度 ===", flush=True)
# 模拟 diffusion MSE loss: ||eps_pred - eps_gt||^2, grad = 2*(eps_pred - eps_gt)
# 典型: eps_pred 偏离 eps_gt 约 0.1-0.3
for eps_offset in [0.1, 0.2, 0.5, 1.0]:
    diff_grad = 2 * torch.randn(BS, 4, 32, 32, device=DEVICE) * eps_offset
    diff_gnorm = diff_grad.view(BS, -1).norm(dim=1).mean().item()
    print(f"  eps偏移={eps_offset:.1f}: diff_grad_norm={diff_gnorm:.4f}", flush=True)

print("\n=== 结论 ===", flush=True)
print("看 struct_grad_ratio vs diff_grad_ratio, 决定 w_skel 的量级", flush=True)
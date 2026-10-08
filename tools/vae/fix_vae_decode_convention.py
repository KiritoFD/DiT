#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_vae_decode_convention.py — 把一个"只认 sample/sf"的非标准 VAE **解析地**改回
标准约定 decode(sample)，零训练成本、数学精确（不是近似，不是微调）。

原理 (2026-10-09)
-----------------
旧 Calli-VAE 训练写的是 `x_recon = vae.decode(z / sf)`，即 decoder 相对标准约定
**唯一的偏差就是输入尺度**被放大了 1/sf ≈ 5.49 倍。而 diffusers 的
`AutoencoderKL.decode` 实现为：

    dec = self.decoder(self.post_quant_conv(z))

于是只需令

    post_quant_conv.weight <- post_quant_conv.weight / sf     # bias 保持不动

便有
    decode_new(z) = decoder(post_quant_conv_old(z / sf)) = decode_old(z / sf)

**逐元素恒等**。encoder / 潜空间 / 后验完全不动，重建质量与旧的 decode(sample/sf)
一模一样，但 API 变成标准：encode→sample()，扩散目标 z*sf，推理 decode(pred/sf)。

用法
----
    # 只测量 (dry-run)，不改任何东西
    python tools/vae/fix_vae_decode_convention.py --vae <dir> --img-dir <pics> --n 8

    # 应用并另存为标准约定的新 VAE
    python tools/vae/fix_vae_decode_convention.py --vae <dir> --img-dir <pics> --n 8 \
        --apply --out <new_dir>
"""
import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL


def load_images(args, device):
    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if args.random:
        return torch.rand(args.n, 3, args.size, args.size, device=device) * 2 - 1
    paths = []
    exts = (".png", ".jpg", ".jpeg", ".webp")
    for r, _d, fs in os.walk(args.img_dir):
        for fn in sorted(fs):
            if fn.lower().endswith(exts) and fn.startswith(args.prefix):
                paths.append(os.path.join(r, fn))
        if len(paths) >= args.n:
            break
    xs = [tf(Image.open(p).convert("RGB")) for p in paths[:args.n]]
    return torch.stack(xs).to(device)


@torch.no_grad()
def probe(vae, x, sf, tag, sm=None):
    post = vae.encode(x).latent_dist
    s, m = sm if sm is not None else (post.sample(), post.mode())
    tab = {
        "sample": F.l1_loss(vae.decode(s).sample, x).item(),
        "sample/sf": F.l1_loss(vae.decode(s / sf).sample, x).item(),
        "mode": F.l1_loss(vae.decode(m).sample, x).item(),
        "mode/sf": F.l1_loss(vae.decode(m / sf).sample, x).item(),
    }
    print(f"[{tag}] posterior.std={post.std.mean().item():.4f}  L1: "
          + "  ".join(f"{k}={v:.4f}" for k, v in tab.items()), flush=True)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--img-dir", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sf", type=float, default=None, help="默认取 vae.config.scaling_factor")
    ap.add_argument("--apply", action="store_true", help="真正改写权重")
    ap.add_argument("--out", default="", help="--apply 时另存新目录 (不填则原地覆盖)")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    sf = float(args.sf if args.sf is not None else vae.config.scaling_factor)
    print(f"[fix] VAE={args.vae}  scaling_factor={sf}")

    x = load_images(args, device)
    with torch.no_grad():
        post0 = vae.encode(x).latent_dist
    sm = (post0.sample(), post0.mode())   # 固定同一次采样，保证 before/after 可比
    before = probe(vae, x, sf, "before", sm)

    # 判定：只有"只认 sample/sf"这种被训歪的才需要修
    inverted = before["sample/sf"] < before["sample"] * 0.5
    if not inverted:
        print("[fix] 该 VAE 已是标准约定 (decode(sample) 最优)，无需修改。")
        return
    if not args.apply:
        print("[fix] 判定: INVERTED (只认 sample/sf)。加 --apply 即解析修正 "
              "(post_quant_conv.weight /= sf)。")
        return

    with torch.no_grad():
        W = vae.post_quant_conv.weight.data
        old_scale = W.abs().mean().item()
        vae.post_quant_conv.weight.data = W / sf
    print(f"[fix] post_quant_conv.weight 已缩放 1/{sf} = {1/sf:.4f} "
          f"(|W| 均值 {old_scale:.5f} -> {vae.post_quant_conv.weight.data.abs().mean().item():.5f})")

    after = probe(vae, x, sf, "after ", sm)
    # 实测断言：after 的 decode(sample) 必须 == before 的 decode(sample/sf)
    delta = abs(after["sample"] - before["sample/sf"])
    print(f"[fix] 校验: |after.decode(sample) - before.decode(sample/sf)| = {delta:.6f} "
          f"({'✓ 恒等' if delta < 1e-4 else '✗ 有残差，需微调'})")

    out = args.out or args.vae
    vae.save_pretrained(out)
    print(f"[fix] 已保存标准约定 VAE -> {out}")


if __name__ == "__main__":
    main()

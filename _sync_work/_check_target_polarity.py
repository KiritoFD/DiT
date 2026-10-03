# -*- coding: utf-8 -*-
"""_check_target_polarity.py — 验证 12ch 训练目标 (img/canny/skel) 的极性是否一致.

方法: 直接读训练实际用的 wz latent shard -> 加回白底 -> decode -> 看是不是**白底**;
      并打印 per-ch mean (白底空间里 背景≈0, 笔画/边缘≈负值)。
另: 同时 decode PNG 侧 (final_canny_base / final_skel3_base) 做对照, 看哪一侧反了。
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from diffusers.models import AutoencoderKL  # noqa: E402
SF = 0.18215
dev = "cuda"
vae = AutoencoderKL.from_pretrained("data/pretrained/pretrained_models/sd-vae-ft-ema").to(dev).eval()
wl = torch.from_numpy(np.load("data/white_latent.npy")).float().to(dev)
os.makedirs("_otout4", exist_ok=True)

DIRS = {
    "img":   "data/latents/final_latents_base_wz",
    "canny": "data/aux/aux_canny_latents_base_wz",
    "skel3": "data/skel/aux_skel3_latents_base_wz",
}


def dec(lat):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        d = vae.decode(lat / SF).sample
    return ((d[0].float().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(1, 2, 0)


print("=== 训练目标 (wz latent) 极性检查: 加回白底后 decode ===")
for tag, d in DIRS.items():
    sp = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not sp:
        print(f"  [{tag}] NO SHARD in {d}")
        continue
    z = np.load(sp[0])
    lat = torch.from_numpy(np.array(z["latents"][:1], copy=True)).float().to(dev)
    per_ch = lat.mean(dim=(2, 3))[0].cpu().numpy()
    img = dec(lat + wl[None])
    Image.fromarray((img * 255).astype(np.uint8)).save(f"_otout4/target_{tag}.png")
    r, g, b = (img.reshape(-1, 3).mean(0) * 255)
    # 亮区(背景) RGB
    lum = img.mean(-1)
    m = lum >= np.percentile(lum, 70)
    bg = img.reshape(-1, 3)[m.reshape(-1)].mean(0) * 255
    print(f"  [{tag:5s}] lat_wz per-ch={np.round(per_ch,4)}")
    print(f"           decode 全图 RGB={np.round([r,g,b],1)} 亮区(背景)RGB={np.round(bg,1)} "
          f"R-B={bg[0]-bg[2]:+.1f}")

print("\n=== PNG 侧对照 (未经过 wz) ===")
for tag, pat in (("canny_png", "data/aux/final_canny_base/*.png"),
                 ("skel3_png", "data/skel/final_skel3_base/*.png")):
    fs = sorted(glob.glob(pat))
    if not fs:
        print(f"  [{tag}] none")
        continue
    a = np.asarray(Image.open(fs[0]).convert("L"), dtype=np.float32)
    print(f"  [{tag}] {os.path.basename(fs[0])} gray mean={a.mean():.1f} "
          f"前景(>127)占比={(a > 127).mean():.3f}  <- 白底则前景占比低")
    Image.open(fs[0]).convert("RGB").save(f"_otout4/png_{tag}.png")
print("\n[done] -> _otout4/")

# -*- coding: utf-8 -*-
"""_diag_wz.py — 白底归零全链路诊断.

1) 原始图底色体检 (用户怀疑: 有黄底/深底图没二值化)
2) wz latent round-trip: encode -> 减白底 -> (加回白底) -> decode, 看图是否还原
3) 各类 latent 的 per-ch mean, 与 white_latent 对比
"""
import csv
import os
import sys

import numpy as np
import torch as th
from PIL import Image
from torchvision import transforms as T

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SF = 0.18215
TF = T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
print(f"rows={len(rows)}")

# ---------- 1) 原图底色体检 ----------
import re
stats = []
for r in rows[:4000]:
    p = r["image_path"]
    if not os.path.exists(p):
        continue
    im = Image.open(p).convert("L")
    a = np.asarray(im.resize((256, 256), Image.BICUBIC), dtype=np.float32)
    # 背景 = 亮度最高的 40%
    bg = a[a >= np.percentile(a, 60)]
    stats.append((a.mean(), float(np.percentile(a, 95)), float(bg.mean()),
                  float((a < 127).mean())))
s = np.array(stats)
print(f"\n[原图底色体检] n={len(s)}")
print(f"  global mean   : p5={np.percentile(s[:,0],5):.1f} p50={np.percentile(s[:,0],50):.1f} p95={np.percentile(s[:,0],95):.1f}")
print(f"  p95 亮度(近底): p5={np.percentile(s[:,1],5):.1f} p50={np.percentile(s[:,1],50):.1f} p95={np.percentile(s[:,1],95):.1f}")
print(f"  背景均亮      : p5={np.percentile(s[:,2],5):.1f} p50={np.percentile(s[:,2],50):.1f} p95={np.percentile(s[:,2],95):.1f}")
print(f"  a<127(前景)占比: p5={np.percentile(s[:,3],5):.3f} p50={np.percentile(s[:,3],50):.3f} p95={np.percentile(s[:,3],95):.3f}")
# 深底嫌疑: 背景均亮 < 128 -> skeletonize(a<127) 会整片判为前景
bad = s[s[:, 2] < 128]
print(f"  ⚠ 背景均亮<128 (深底, skeletonize 会翻车) 的样本: {len(bad)}/{len(s)} = {100*len(bad)/len(s):.1f}%")
dark = s[s[:, 0] < 100]
print(f"  ⚠ 全图均亮<100 (整体偏暗): {len(dark)}/{len(s)} = {100*len(dark)/len(s):.1f}%")

# ---------- 2) white latent ----------
from diffusers.models import AutoencoderKL  # noqa: E402
dev = "cuda"
vae = AutoencoderKL.from_pretrained("data/pretrained/pretrained_models/sd-vae-ft-ema").to(dev).eval()

wl_file = th.from_numpy(np.load("data/white_latent.npy")).float()
print(f"\n[data/white_latent.npy] shape={tuple(wl_file.shape)} mean={wl_file.mean():.4f} "
      f"per-ch={np.round(wl_file.mean(dim=(1,2)).numpy(),4)}")

# ---------- 3) round-trip ----------
def rt(tag, src, invert=False):
    im = Image.open(src).convert("L")
    a = np.asarray(im.resize((256, 256), Image.BICUBIC), dtype=np.float32)
    if invert:
        a = 255.0 - a
    x = th.from_numpy(a).float().unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x / 255.0 - 0.5) / 0.5
    with th.no_grad():
        lat = (vae.encode(x.to(dev)).latent_dist.mode() * SF).float()
        wl = (vae.encode(x.to(dev) * 0 + 1.0).latent_dist.mode() * SF).float()  # 白图 latent
        lat_wz = lat - wl
        # a) 原 latent decode
        d0 = vae.decode(lat / SF).sample
        # b) wz latent 直接 decode (不加回) -> 应偏暗
        d1 = vae.decode(lat_wz / SF).sample
        # c) wz + 白底 加回 decode -> 应还原
        d2 = vae.decode((lat_wz + wl) / SF).sample
    for nm, d in (("orig", d0), ("wz_only", d1), ("wz+white", d2)):
        arr = ((d[0].float().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((arr * 255).astype(np.uint8)).save(f"_otout/{tag}_{nm}.png")
    print(f"  [{tag}] src={src}")
    print(f"     原图灰度 mean={a.mean():.1f}")
    print(f"     lat     per-ch={np.round(lat.mean(dim=(2,3))[0].cpu().numpy(),4)} mean={lat.mean():.4f}")
    print(f"     lat_wz  per-ch={np.round(lat_wz.mean(dim=(2,3))[0].cpu().numpy(),4)} mean={lat_wz.mean():.4f}")
    print(f"     decode: orig mean={((d0.float().clamp(-1,1)+1)/2).mean()*255:.1f} "
          f"| wz_only={((d1.float().clamp(-1,1)+1)/2).mean()*255:.1f} "
          f"| wz+white={((d2.float().clamp(-1,1)+1)/2).mean()*255:.1f}")

os.makedirs("_otout", exist_ok=True)
print("\n[round-trip]")
r0 = rows[0]
iid = int(re.search(r"(\d+)\.png", r0["image_path"]).group(1))
rt("img", r0["image_path"])
for nm, d, inv in (("skel3", "data/skel/final_skel3_base", False),
                   ("canny", "data/aux/final_canny_base", True)):
    p = f"{d}/{iid}.png"
    if os.path.exists(p):
        rt(nm, p, inv)
    else:
        print(f"  [{nm}] MISSING {p}")
# 也看 raw canny (未反相)
p = f"data/aux/final_canny_base/{iid}.png"
if os.path.exists(p):
    rt("canny_raw", p, False)

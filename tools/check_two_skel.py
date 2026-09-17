# -*- coding: utf-8 -*-
"""check_two_skel.py — 实测指定 id 的骨架 latent 统计与 decode。

用法: python tools/check_two_skel.py 967376 975597 ...
"""
import glob
import os
import sys

import numpy as np
import torch as th
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SF = 0.18215
IDS = [int(x) for x in sys.argv[1:]] or [967376, 975597]

loc = {}
for sub in ("data/skel/std_skel3_latents_base_sym",
            "data/latents/final_latents_base_sym"):
    for sp in sorted(glob.glob(os.path.join(sub, "shard_*.npz"))):
        with np.load(sp) as d:
            for j, iid in enumerate(d["img_ids"]):
                loc.setdefault((sub, int(iid)), (sp, int(j)))

from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained(
    "data/pretrained/pretrained_models/sd-vae-ft-ema").to("cuda").eval()

out = "/root/Workspace/xy/DiT/_otout_two"
os.makedirs(out, exist_ok=True)
for iid in IDS:
    for sub in ("data/skel/std_skel3_latents_base_sym",
                "data/latents/final_latents_base_sym"):
        if (sub, iid) not in loc:
            print(f"  {iid} {sub.split('/')[-1]}: 不存在")
            continue
        sp, j = loc[(sub, iid)]
        with np.load(sp) as d:
            lat = np.array(d["latents"][j], dtype=np.float32)
        nm = sub.split("/")[-1]
        print(f"  {iid} {nm:32s} shape={lat.shape} "
              f"|mean|={np.abs(lat).mean():.4f} std={lat.std():.4f} "
              f"min={lat.min():.3f} max={lat.max():.3f} "
              f"per-ch={np.round(lat.mean(axis=(1,2)),3)}")
        x = th.from_numpy(lat)[None].to("cuda")
        with th.no_grad():
            dec = vae.decode(x / SF).sample
        dec = ((dec.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        im = (dec * 255).astype(np.uint8)
        Image.fromarray(im).save(f"{out}/{iid}_{nm[:12]}.png")
        print(f"        decode RGB mean={im.reshape(-1,3).mean(axis=0).round(1)}")

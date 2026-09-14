# -*- coding: utf-8 -*-
"""
rebuild_latents_wz.py — 重建三类 latent: (a) canny 转白底 (b) 减去"白底 latent"。

动机 (用户 2026-09-14 裁定):
  1) 实测我们三个通道底色/极性不一致: image 与 skel3 是白底, 而 canny 是**黑底**
     (per-channel mean: img [1.02,0.65,0.15,-0.63] / skel [1.33,1.16,-0.01,-0.90]
      / canny [-0.88,-1.66,0.64,0.76]) -> 模型要额外学"极性翻转", 统一为白底。
  2) 减去白底 latent (ref 里叫 custom_zero): 让"空白区域"在 latent 空间归零,
     笔画/边缘信号变成显著偏离, 不被白底常数 bias 淹没。
     ⚠ 注意: ref 的启动脚本 custom_zero 全为 0 (从未启用); 本脚本按用户要求实现。

产物 (写入新目录, 不覆盖旧 shards):
  data/latents/final_latents_base_wz/         image  latent (白底归零)
  data/skel/aux_skel3_latents_base_wz/        skel3  latent (白底归零)
  data/aux/aux_canny_latents_base_wz/         canny  latent (先转白底, 再归零)

⚠ 推理侧必须配套: 生成 latent 后要 **加回白底** 再 VAE decode, 否则整体偏色。
   见 config 的 aux_zero_white / 推理脚本。

用法: python tools/rebuild_latents_wz.py [--batch 64] [--limit N]
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np
import torch as th
from PIL import Image
from torchvision import transforms as T

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"
SF = 0.18215
CSV = "assets/train_base_noaug.csv"
SHARD_N = 5000

TF = T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

# (任务名, 源目录, 输出目录, 是否反转极性)
JOBS = [
    ("img",   None,                          "data/latents/final_latents_base_wz",  False),
    ("skel3", "data/skel/final_skel3_base",  "data/skel/aux_skel3_latents_base_wz", False),
    ("canny", "data/aux/final_canny_base",   "data/aux/aux_canny_latents_base_wz",  True),
]


def load_gray(path, invert):
    im = Image.open(path).convert("L")
    a = np.asarray(im.resize((256, 256), Image.BICUBIC), dtype=np.float32)
    if invert:
        a = 255.0 - a                      # 黑底白线 -> 白底黑线
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from diffusers.models import AutoencoderKL
    dev = "cuda"
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(dev).eval()

    # 1) 白底 latent (确定性 mode, 与训练侧 build_*_latents 的 mode() 一致)
    white = TF(Image.new("RGB", (256, 256), "white")).unsqueeze(0).to(dev)
    with th.no_grad():
        wl = (vae.encode(white).latent_dist.mode() * SF)[0].float().cpu().numpy()
    print(f"[white] latent mean={wl.mean():.4f} per-ch={np.round(wl.mean(axis=(1,2)),4)}",
          flush=True)

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[rows] {len(rows)}", flush=True)

    for name, src_dir, out_dir, invert in JOBS:
        os.makedirs(out_dir, exist_ok=True)
        buf, ids = [], []
        n_shard, t0 = 0, time.time()
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            paths, iids = [], []
            for r in chunk:
                p = r["image_path"]
                m = re.search(r"(\d+)\.png", p)
                if m is None:
                    continue
                iid = int(m.group(1))
                if name == "img":
                    src = p
                else:
                    src = f"{src_dir}/{iid}.png"
                    if not os.path.exists(src):
                        continue
                if not os.path.exists(src):
                    continue
                paths.append(src)
                iids.append(iid)
            if not paths:
                continue
            arr = np.stack([load_gray(p, invert) for p in paths])
            x = th.from_numpy(arr).float().unsqueeze(1).repeat(1, 3, 1, 1)
            x = (x / 255.0 - 0.5) / 0.5                 # [0,1] -> [-1,1]
            with th.no_grad():
                lat = (vae.encode(x.to(dev)).latent_dist.mode() * SF).float().cpu().numpy()
            lat = lat - wl[None]                        # ★ 减白底
            buf.extend(lat.astype(np.float16))
            ids.extend(iids)
            if len(buf) >= SHARD_N:
                np.savez(os.path.join(out_dir, f"shard_{n_shard:05d}.npz"),
                         latents=np.stack(buf), img_ids=np.array(ids, dtype=np.int64))
                n_shard += 1
                buf, ids = [], []
                print(f"  [{name}] shard {n_shard} written ({i}/{len(rows)}) "
                      f"{time.time()-t0:.0f}s", flush=True)
        if buf:
            np.savez(os.path.join(out_dir, f"shard_{n_shard:05d}.npz"),
                     latents=np.stack(buf), img_ids=np.array(ids, dtype=np.int64))
            n_shard += 1
        print(f"[{name}] DONE -> {out_dir} ({n_shard} shards, {time.time()-t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()

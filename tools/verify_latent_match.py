# -*- coding: utf-8 -*-
"""verify_latent_match.py — 验证 csv 里的图与 latent shard 是否一一对应.

做法: 随机取 N 行 -> 按 img_id 从 shard 取 latent -> VAE decode -> 与 csv 里
     image_path 指向的原图比对 (SSIM/MSE)。若明显不一致 => latent 与 id 错位。
同时导出对照图供目测。
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import torch as th
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"
SF = 0.18215


def ssim(a, b):
    from src.eval.inference import _ssim
    return _ssim(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_base_sym_clean.csv")
    ap.add_argument("--latents", default="data/latents/final_latents_base_sym")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--aug", default="all",
                    help="all / plain / tp / tn —— 抽哪一类样本 (重点验证增强图)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    if a.aug != "all":
        want = {"plain": ""}.get(a.aug, a.aug)
        rows = [r for r in rows if (r.get("aug") or "") == want]
    print(f"[verify] {a.csv}: 取 {len(rows)} 行 (aug={a.aug})  latents={a.latents}")
    step = a.stride or max(1, len(rows) // a.n)

    # 建 id -> (shard, row)
    import glob
    id2loc = {}
    for sp in sorted(glob.glob(os.path.join(a.latents, "shard_*.npz"))):
        with np.load(sp) as d:
            for j, iid in enumerate(d["img_ids"]):
                id2loc[int(iid)] = (sp, int(j))
    print(f"[verify] shard index: {len(id2loc)} ids")

    from diffusers.models import AutoencoderKL
    dev = "cuda"
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(dev).eval()

    out = "/root/Workspace/xy/DiT/_otout_match"
    os.makedirs(out, exist_ok=True)
    picks = [rows[i] for i in range(0, len(rows), step)][:a.n]
    for k, r in enumerate(picks):
        p = r["image_path"]
        m = re.search(r"(\d+)\.png", p)
        if not m:
            continue
        iid = int(m.group(1))
        if iid not in id2loc:
            print(f"  [{k}] id={iid} 不在 shard 中!  {p}")
            continue
        sp, j = id2loc[iid]
        with np.load(sp) as d:
            lat = np.array(d["latents"][j], dtype=np.float32)
        x = th.from_numpy(lat)[None].to(dev)
        with th.no_grad():
            dec = vae.decode(x / SF).sample
        dec = ((dec.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        gt = np.asarray(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        s = ssim(dec, gt)
        mse = float(((dec - gt) ** 2).mean())
        print(f"  [{k}] id={iid} aug={r.get('aug') or '-':3s} "
              f"char={r.get('character','')} ssim={s:.4f} mse={mse:.5f}  {p[-34:]}")
        Image.fromarray((dec * 255).astype(np.uint8)).save(f"{out}/{k}_dec_{iid}.png")
        Image.fromarray((gt * 255).astype(np.uint8)).save(f"{out}/{k}_gt_{iid}.png")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

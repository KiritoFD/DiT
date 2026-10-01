"""直接解码落盘的 latent shards, 出原生 256 骨架图 + 墨占比。

用途: 绕开 eval/poster 那条路径 (已验证它对 *_pred 会写出全白 png),
用**训练自己 dump 的产物**判断生成器到底吐了什么。
用法: python tools/decode_dump.py --gen <目录> --ref <目录> --n 8
"""
import argparse
import glob
import os

import numpy as np
import torch as th
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw


def load_dir(d, cap=None):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                m[int(i)] = np.asarray(z["latents"][j], np.float32)
        if cap and len(m) >= cap:
            break
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="生成骨架的 latent shards 目录")
    ap.add_argument("--ref", default="", help="参照 (如 std 条件) 目录, 可选")
    ap.add_argument("--tgt", default="", help="目标 (如 gtskel_w7) 目录, 可选")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--sf", type=float, default=0.18215)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--out", default="_ot_scratch/decode_dump.png")
    a = ap.parse_args()

    dev = "cuda" if th.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
    gen = load_dir(a.gen)
    ids = sorted(gen)
    ids = ids[:a.n]
    rows = [("gen(dump)", gen, ids)]
    if a.ref:
        r = load_dir(a.ref)
        rows.append(("ref(std)", r, [i for i in ids if i in r]))
    if a.tgt:
        t = load_dir(a.tgt)
        rows.append(("tgt(w7)", t, [i for i in ids if i in t]))

    imgs = {}
    for nm, m, ii in rows:
        if not ii:
            continue
        x = th.from_numpy(np.stack([m[i] for i in ii])).to(dev)
        with th.no_grad():
            d = vae.decode(x / a.sf).sample
        d = ((d.float().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
        # 骨架 latent 的"墨"= 偏暗像素; 二值化阈值取 0.5 与 <0(等价于 decode 前的负值)各打一份
        g = d[..., :3].mean(-1)
        print(f"[{nm}] n={len(ii)}  ink<0.5 {float((g < 0.5).mean()):.4f}  "
              f"ink<0.25 {float((g < 0.25).mean()):.4f}  "
              f"img mean {float(g.mean()):.4f}  min {float(g.min()):.3f}")
        imgs[nm] = [(ii[k], (g[k] < 0.5)) for k in range(len(ii))]

    cell = a.cell
    labels = list(imgs)
    n = min(len(v) for v in imgs.values())
    W, H = cell * n + 8, (cell + 22) * len(labels) + 8
    canvas = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(canvas)
    for r, nm in enumerate(labels):
        y = r * (cell + 22)
        dr.text((4, y), f"{nm}  n={len(imgs[nm])}", fill=0)
        for c in range(n):
            bits = imgs[nm][c][1]
            canvas.paste(Image.fromarray(np.where(bits, 0, 255).astype(np.uint8)),
                         (4 + c * cell, y + 20))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    canvas.save(a.out)
    print(f"poster -> {a.out}")


if __name__ == "__main__":
    main()

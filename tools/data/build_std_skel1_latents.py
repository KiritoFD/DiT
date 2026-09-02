# -*- coding: utf-8 -*-
"""build_std_skel1_latents.py — 渲染标准字形 -> 1px 骨架 -> VAE latent (一条龙).

输出两件套:
  1) std_skel1_latents_fame/  训练 shards (shard_XXXXX.npz, latents+img_ids)  <- s27 训练条件
  2) skel_bank_std1.npz       推理骨架库 (keys: "楷|字", latents)            <- gradio zero-shot 直出

1px = skeletonize 后不做 dilation (与 final_skel1_fame 一致).
CPU 渲染/骨架化 + GPU VAE encode (batch 小, 不影响并行训练).
"""
import os
import sys
import csv
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_fame.csv")
    ap.add_argument("--out-shards", default="std_skel1_latents_fame")
    ap.add_argument("--out-bank", default="skel_bank_std1.npz")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--font-dir", default="/tmp")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--font-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=2592)
    return ap.parse_args()


SCRIPT_FONT = {
    "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
    "行": ["STXINGKA.TTF", "FZSTK.TTF"],
    "隶": ["SIMLI.TTF", "STLITI.TTF"],
}


def skeletonize_1px(binary):
    """zhang-suen via skimage; fallback scipy erosion loop. binary: HxW bool (True=笔画)"""
    try:
        from skimage.morphology import skeletonize
        return skeletonize(binary)
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        img = binary.copy()
        skel = np.zeros_like(binary)
        st = generate_binary_structure(2, 2)
        while img.any():
            er = binary_erosion(img, structure=st)
            skel |= img & ~er
            img = er
        return skel


def pick_font(script, font_dir, cache={}):
    cands = SCRIPT_FONT.get(script, SCRIPT_FONT["楷"])
    for f in cands:
        p = os.path.join(font_dir, f)
        if os.path.isfile(p):
            if f not in cache:
                cache[f] = ImageFont.truetype(p, 200)
            return cache[f]
    return None


def render_skel1(ch, script, size=256, font_size=200):
    """渲染字 -> 1px 骨架 uint8 (255=白底, 0=黑线)"""
    font = pick_font(script, FONT_DIR)
    if font is None:
        return None
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    d.text((size // 2, size // 2), ch, font=font, fill=0, anchor="mm")
    a = np.asarray(img)
    if (a < 250).sum() < 10:
        return None  # 字体缺字
    sk = skeletonize_1px(a < 127)
    return np.where(sk, 0, 255).astype("uint8")


FONT_DIR = "/tmp"


def main():
    global FONT_DIR
    args = parse_args()
    FONT_DIR = args.font_dir

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    print(f"[csv] {len(rows)} rows")

    # 去重: 每个 (script, character) 渲染一次
    pairs = sorted({(r["script"], r["character"]) for r in rows})
    print(f"[pairs] {len(pairs)} unique (script, char)")

    # 渲染 1px 骨架
    skels, keys = [], []
    miss_font = 0
    for i, (script, ch) in enumerate(pairs):
        arr = render_skel1(ch, script, args.img_size, args.font_size)
        if arr is None:
            miss_font += 1
            continue
        skels.append(arr)
        keys.append(f"{script}|{ch}")
        if (i + 1) % 1000 == 0:
            print(f"  render {i+1}/{len(pairs)} (miss_font={miss_font})", flush=True)
    print(f"[render] done: {len(skels)} ok, {miss_font} font-miss")

    # 找每个样本对应的骨架索引 (训练 csv 顺序)
    key2idx = {k: i for i, k in enumerate(keys)}
    sample_idx = []
    for r in rows:
        k = f'{r["script"]}|{r["character"]}'
        sample_idx.append(key2idx.get(k, -1))
    n_have = sum(1 for x in sample_idx if x >= 0)
    print(f"[match] {n_have}/{len(rows)} rows have std skel")

    # VAE encode (按唯一骨架编码一次, 样本行复用索引)
    import torch
    from diffusers import AutoencoderKL
    dev = "cuda"
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(dev).eval()
    latents = np.zeros((len(skels), 4, 32, 32), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(skels), args.batch):
            chunk = skels[s:s + args.batch]
            x = torch.from_numpy(np.stack(chunk).astype(np.float32) / 255.0 * 2 - 1)
            x = x[:, None].repeat(1, 3, 1, 1).to(dev)
            lat = (vae.encode(x).latent_dist.mode() * 0.18215).half().cpu().numpy()
            latents[s:s + len(chunk)] = lat
            if (s // args.batch) % 100 == 0:
                print(f"  encode {s + len(chunk)}/{len(skels)}", flush=True)
    print(f"[encode] {latents.shape}")

    # 1) bank 保存
    np.savez(args.out_bank, keys=np.array(keys), latents=latents)
    print(f"[bank] {args.out_bank}: {len(keys)} entries")

    # 2) shards 保存 (按训练 csv 行展开, 缺失行跳过 -> 与 final_latents_fame 对齐)
    os.makedirs(args.out_shards, exist_ok=True)
    shard, ids = [], []
    n_shard = 0

    def flush_shard():
        nonlocal shard, ids, n_shard
        if not shard:
            return
        np.savez(os.path.join(args.out_shards, f"shard_{n_shard:05d}.npz"),
                 latents=np.stack(shard).astype(np.float16),
                 img_ids=np.array(ids, dtype=np.int64))
        n_shard += 1
        shard, ids = [], []

    for row_i, sk_i in enumerate(sample_idx):
        if sk_i < 0:
            continue
        shard.append(latents[sk_i])
        ids.append(row_i)
        if len(shard) >= args.shard_size:
            flush_shard()
    flush_shard()
    print(f"[shards] {n_shard} shards -> {args.out_shards}/")
    print("done.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
先 VAE encode，再对 latent 做 hash（不对像素做任何 hash）。
用于对比本地官方图与远程 dataset 图是否相同。

用法：
  python vae_hash.py --root <images_root> --out <index.json> [--official] [--batch 64]

--official : 官方 trainset_dataset 结构 {split}/{字}/{字-字体-朝代-出处-id}.png
             （默认按递归遍历所有图片）
输出 JSON 列表：每项 {path, latent_fp}
  latent_fp : VAE latent(mean) 量化指纹
"""
import os, sys, hashlib, json, argparse, time
import numpy as np
import cv2
import torch
from diffusers.models import AutoencoderKL
from concurrent.futures import ThreadPoolExecutor

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

VAE_PATH = "pretrained_models/sd-vae-ft-ema"
SIZE = 256
LATENT_SCALE = 0.18215


def read_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def generic_walk(root):
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(dp, fn))
    return paths


def encode_latents(vae, imgs_bgr_list, device):
    """批量 VAE encode -> 确定性 latent mean 指纹 (4,32,32) 量化"""
    tensors = []
    for img in imgs_bgr_list:
        r = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        tensors.append(t)
    x = torch.stack(tensors).to(device) * 2.0 - 1.0
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mean      # (B,4,32,32) 确定性
    latent = latent.float().mul_(LATENT_SCALE).cpu().numpy()  # (B,4,32,32)
    fps = []
    for i in range(latent.shape[0]):
        arr = (latent[i] * 1000).round().astype(np.int16)
        fps.append(hashlib.md5(arr.tobytes()).hexdigest())
    return fps, latent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--io-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--save-latent-dir", default=None,
                    help="若指定，额外把 latent(float16) 存成 npy，路径与图片相对 root 对应")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"VAE hashing on {device}")

    # 官方结构与通用结构统一用递归遍历，避免层级假设错配
    paths = generic_walk(args.root)
    paths.sort()
    print(f"total found: {len(paths)}")
    if args.limit > 0 and args.limit < len(paths):
        paths = paths[:args.limit]
        print(f"limited to {len(paths)}")

    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    read_pool = ThreadPoolExecutor(max_workers=args.io_workers)

    results = []
    n_fail = 0
    t0 = time.time()
    idx = 0
    total = len(paths)
    while idx < total:
        chunk = paths[idx: idx + args.batch]
        idx += len(chunk)
        futs = {p: read_pool.submit(read_image, p) for p in chunk}
        imgs = []
        keep = []
        for p, f in futs.items():
            img = f.result()
            if img is None:
                n_fail += 1
                continue
            imgs.append(img)
            keep.append(p)
        if imgs:
            lfps, latents = encode_latents(vae, imgs, device)
            for p, lfp, lat in zip(keep, lfps, latents):
                results.append({"path": p.replace("\\", "/"), "latent_fp": lfp})
                if args.save_latent_dir:
                    rel = os.path.relpath(p, args.root)
                    outp = os.path.join(args.save_latent_dir, os.path.splitext(rel)[0] + ".npy")
                    os.makedirs(os.path.dirname(outp), exist_ok=True)
                    np.save(outp, lat.astype(np.float16))
        el = time.time() - t0
        print(f"\r[{idx}/{total}] {idx/el:.1f} imgs/s fail={n_fail}", end="", flush=True)

    print(f"\nDone. total={len(results)} fail={n_fail} elapsed={time.time()-t0:.1f}s")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()

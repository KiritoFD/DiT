# -*- coding: utf-8 -*-
"""build_aux_latents_fame_e.py — 为 fame_e 训练集构建 aux 结构 latent (moyi 式辅助目标通道).

对 train_fame3_e_full.csv 每一行:
  1) 读图 final_imgs_fame_e/<img_id>.png -> 灰度 -> 墨迹二值 (dark=ink)
  2) skeleton : skeletonize 墨迹 -> 黑线白底图
  3) canny    : Canny 边缘 -> 黑线白底图
  4) VAE encode -> (4,32,32) latent (x0.18215)
输出两个 shard 目录 (shard_XXXXX.npz: latents fp16 + img_ids int64):
  aux_skel_latents_fame_e/ , aux_canny_latents_fame_e/
训练时 x 目标 = cat(image_latent, skel_latent, canny_latent) = 12ch (推理只用前 4).

用法:
  python tools/data/build_aux_latents_fame_e.py --csv 5script/train_fame3_e_full.csv \
      --img-root final_imgs_fame_e --out-skel aux_skel_latents_fame_e \
      --out-canny aux_canny_latents_fame_e --batch 32 --shard-size 2592
"""
import argparse
import csv
import os
import re
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CKPT_SF = 0.18215


def _process_one(task):
    """CPU: 读图 -> (skel_img, canny_img) 均为 uint8 黑线白底 256x256.

    task = (i, path, skel_dilate): skel_dilate>0 时对 1px 骨架做 N 次 3x3 膨胀
    (N=1 -> ~3px), 提升 VAE 256->32 下的可编码性/区分度。
    """
    i, path, skel_dilate = task
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L"), dtype=np.uint8)
    except Exception:
        z = np.full((256, 256), 255, np.uint8)
        return i, z, z
    ink = g < 128
    # skeleton
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(ink)
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        im2 = ink.copy()
        sk = np.zeros_like(ink)
        st = generate_binary_structure(2, 2)
        while im2.any():
            er = binary_erosion(im2, structure=st)
            sk |= im2 & ~er
            im2 = er
    if int(skel_dilate) > 0:
        from scipy.ndimage import binary_dilation, generate_binary_structure as _gbs
        sk = binary_dilation(sk, _gbs(2, 2), iterations=int(skel_dilate))
    skel_img = np.where(sk, 0, 255).astype(np.uint8)
    # canny
    try:
        import cv2
        ed = cv2.Canny(g, 50, 150)
    except Exception:
        from skimage.feature import canny as _canny
        ed = (_canny(g, sigma=1.5) * 255).astype(np.uint8)
    canny_img = np.where(ed > 0, 0, 255).astype(np.uint8)
    return i, skel_img, canny_img


def _encode(vae, imgs, dev, batch):
    """imgs: list of uint8 (256,256) -> (N,4,32,32) fp16 numpy."""
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), batch):
            chunk = imgs[s:s + batch]
            x = torch.from_numpy(np.stack(chunk).astype(np.float32) / 255.0 * 2 - 1)
            x = x[:, None].repeat(1, 3, 1, 1).to(dev)
            lat = (vae.encode(x).latent_dist.mode() * CKPT_SF).half().cpu().numpy()
            out.append(lat)
    return np.concatenate(out, 0)


def _encode_fast(fv, imgs, batch):
    """imgs: list of uint8 (256,256) -> (N,4,32,32) fp16 numpy (FastVAE, bf16 encode)."""
    x = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0 * 2 - 1)[:, None]
    z = fv.encode(x, chunk=int(batch))
    return z.half().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_fame3_e_full.csv")
    ap.add_argument("--img-root", default="final_imgs_fame_e")
    ap.add_argument("--out-skel", default="aux_skel_latents_fame_e")
    ap.add_argument("--out-canny", default="aux_canny_latents_fame_e")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=2592)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skel-dilate", type=int, default=0,
                    help="1px 骨架膨胀次数 (1 -> ~3px); 用于更易编码/更可分的 skel 监督")
    ap.add_argument("--skip-canny", action="store_true")
    ap.add_argument("--skip-skel", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    ids, paths = [], []
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if not m:
            continue
        iid = int(m.group(1))
        # 优先用 csv 的 image_path (相对仓库根); 否则回退 img_root/<id>.png
        cand = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(os.getcwd(), r["image_path"])
        p = cand if os.path.isfile(cand) else os.path.join(args.img_root, f"{iid}.png")
        ids.append(iid)
        paths.append(p)
    n = len(ids)
    print(f"[csv] {n} rows; missing images -> blank", flush=True)

    t0 = time.time()
    skels = [None] * n
    cannys = [None] * n
    with Pool(args.workers) as pool:
        for done, (i, sk, ca) in enumerate(pool.imap_unordered(
                _process_one, [(i, p, args.skel_dilate) for i, p in enumerate(paths)],
                chunksize=256), 1):
            skels[i] = sk
            cannys[i] = ca
            if done % 20000 == 0:
                print(f"  preprocess {done}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[preprocess] done {n} ({time.time()-t0:.0f}s)", flush=True)

    from diffusers import AutoencoderKL
    dev = args.device
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    jobs = []
    if not args.skip_skel:
        jobs.append(("skel", skels, args.out_skel))
    if not args.skip_canny:
        jobs.append(("canny", cannys, args.out_canny))
    for tag, imgs, outdir in jobs:
        os.makedirs(outdir, exist_ok=True)
        nsh = 0
        for s in range(0, n, args.shard_size):
            e = min(s + args.shard_size, n)
            lat = _encode(vae, imgs[s:e], dev, args.batch)
            np.savez(os.path.join(outdir, f"shard_{nsh:05d}.npz"),
                     latents=lat, img_ids=np.array(ids[s:e], dtype=np.int64))
            nsh += 1
            print(f"  [{tag}] shard {nsh} {s}:{e} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{tag}] {nsh} shards -> {outdir}/", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

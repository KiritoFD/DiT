# -*- coding: utf-8 -*-
"""build_eval_skel1_latents.py — 为 eval 集补全 1px skel PNG 并 encode 成 1px latent shards.

用途: 用当前已训练 ckpt 直接对比 3px vs 1px 骨架的控制效果, 无需重训.

两阶段 (断点续跑):
  Phase1: 对 eval csv 里缺失的 id, 从 GT 图 (白底黑字) 用 skimage skeletonize 提 1px,
          存为 白底黑线 PNG (与 data/skel/final_skel1 同极性). 已有则跳过.
  Phase2: 对全部 eval id 的 1px PNG 做 VAE encode -> latent shards (4ch/32x32).

用法 (远程后台, 注意 GPU 显存: 训练占 17.28G/24G, 用 batch=16 控制峰值):
  /opt/conda/bin/python _sync_work/build_eval_skel1_latents.py \
      --csv assets/eval_strict_midclean.csv \
      --img-root data/imgs/final_imgs_256 --skel1-dir data/skel/final_skel1 \
      --latent-out data/skel/final_skel_latents_eval_1px \
      --vae-path data/pretrained/sd-vae-ft-ema --batch 16
"""
import os, sys, csv, re, time, glob, argparse
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")


def _skel_impl():
    try:
        from skimage.morphology import skeletonize
        return skeletonize
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        def skeletonize(binary):
            skel = np.zeros_like(binary)
            img = binary.copy()
            struct = generate_binary_structure(2, 2)
            while img.any():
                eroded = binary_erosion(img, structure=struct)
                skel |= img & ~eroded
                img = eroded
            return skel
        return skeletonize


_SKEL = None


def _gen_one(args):
    """模块级: 供 multiprocessing pickling. 读 GT -> 1px 白底黑线 PNG."""
    global _SKEL
    if _SKEL is None:
        _SKEL = _skel_impl()
    iid, img_root, skel1_dir = args
    img = np.asarray(Image.open(os.path.join(img_root, f"{iid}.png")).convert("L"))
    binary = img < 127
    skel1 = _SKEL(binary)
    arr = np.where(skel1, 0, 255).astype(np.uint8)  # 白底黑线
    Image.fromarray(arr, mode="L").save(os.path.join(skel1_dir, f"{iid}.png"))
    return iid


def read_ids(csv_path):
    ids = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)\.png", row["image_path"])
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))


def phase1_pngs(ids, img_root, skel1_dir, workers):
    global _SKEL
    if _SKEL is None:
        _SKEL = _skel_impl()
    os.makedirs(skel1_dir, exist_ok=True)
    todo = []
    for iid in ids:
        p = os.path.join(skel1_dir, f"{iid}.png")
        if not os.path.exists(p):
            todo.append(iid)
    print(f"[p1] need 1px png: {len(todo)} / {len(ids)}", flush=True)

    args_list = [(iid, img_root, skel1_dir) for iid in todo]

    if workers <= 1:
        for iid in todo:
            _gen_one((iid, img_root, skel1_dir))
    else:
        import multiprocessing as mp
        t0 = time.time()
        done = 0
        with mp.Pool(workers) as pool:
            for _ in pool.imap_unordered(_gen_one, args_list, chunksize=64):
                done += 1
                if done % 200 == 0 or done == len(todo):
                    print(f"[p1] {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[p1] DONE", flush=True)


def phase2_latents(ids, skel1_dir, latent_out, vae_path, batch=16, scaling=0.18215):
    import torch
    from diffusers.models import AutoencoderKL
    os.makedirs(latent_out, exist_ok=True)
    done_ids = set()
    for sp in sorted(glob.glob(os.path.join(latent_out, "shard_*.npz"))):
        with np.load(sp) as d:
            done_ids.update(int(x) for x in d["img_ids"])
    todo = [i for i in ids if i not in done_ids]
    print(f"[p2] encode todo: {len(todo)} / {len(ids)} (skip {len(done_ids)})", flush=True)
    if not todo:
        print("[p2] nothing to encode", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    print(f"[p2] vae on {device}", flush=True)

    def flush(buf):
        if not buf:
            return
        t_ids = [b[0] for b in buf]
        imgs = np.stack([b[1] for b in buf])
        x = torch.from_numpy(imgs.astype(np.float32)) / 255.0 * 2.0 - 1.0
        x = x.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * scaling
        lat = lat.float().cpu().numpy().astype(np.float16)
        lo, hi = min(t_ids), max(t_ids)
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        np.savez(os.path.join(latent_out, f"shard_{lo:05d}_{hi:05d}.npz"),
                 latents=lat, img_ids=np.asarray(t_ids, dtype=np.int64))
        buf.clear()

    buf = []
    t0 = time.time()
    for i, iid in enumerate(todo):
        a = np.asarray(Image.open(os.path.join(skel1_dir, f"{iid}.png")).convert("L"))
        buf.append((iid, a))
        if len(buf) >= batch:
            flush(buf)
            if (i + 1) % 500 == 0 or (i + 1) == len(todo):
                print(f"[p2] {i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    flush(buf)
    print(f"[p2] DONE {len(todo)} -> {latent_out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", default="data/imgs/final_imgs_256")
    ap.add_argument("--skel1-dir", default="data/skel/final_skel1")
    ap.add_argument("--latent-out", required=True)
    ap.add_argument("--vae-path", default="data/pretrained/sd-vae-ft-ema")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-png", action="store_true", help="skip Phase1, only encode")
    args = ap.parse_args()

    ids = read_ids(args.csv)
    print(f"[main] eval ids = {len(ids)}", flush=True)
    if not args.skip_png:
        phase1_pngs(ids, args.img_root, args.skel1_dir, args.workers)
    phase2_latents(ids, args.skel1_dir, args.latent_out, args.vae_path, batch=args.batch)
    print("[main] DONE", flush=True)


if __name__ == "__main__":
    main()

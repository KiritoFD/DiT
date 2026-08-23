# -*- coding: utf-8 -*-
"""
encode_latents_klf4.py — 用 kl-f4 VAE 编码全量 MCCD 图片 → shard_XXXXX.npz

格式与现有 final_latents/ 兼容 (latents + img_ids), 但 latent shape = (N, 3, 64, 64).
- 图片来源: final_images/ (已确认全部含 256x256, 但 ~3% 非标尺寸 → 强制 resize 256x256)
- latent 存 fp16 (3*64*64*2 = 24576 bytes/img)
- scaling_factor = 0.102079 (1/std, 从 200 张样本估计)

用法 (远程):
  python tools/vae/encode_latents_klf4.py --csv 5script/train_top30.csv --img-root final_images \
    --vae pretrained_models/kl-f4 --out final_latents_f4 --shard-size 5000 --scaling-factor 0.102079
"""
import os, sys, csv, json, glob, time, re, argparse
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from PIL import Image


def load_image_ids(csv_path):
    """从 CSV 提取去重后的 img_id 列表."""
    ids = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("image_path", "")
            m = re.search(r"(\d+)\.png", rel)
            if m:
                ids.append(int(m.group(1)))
            else:
                iid = row.get("img_id") or row.get("id")
                if iid:
                    ids.append(int(iid))
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", default="final_images")
    ap.add_argument("--vae", default="pretrained_models/kl-f4")
    ap.add_argument("--out", default="final_latents_f4")
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--scaling-factor", type=float, default=0.102079)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load VAE
    from diffusers.models import AutoencoderKL
    print(f"Loading VAE: {args.vae}", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    in_ch = vae.config.in_channels
    latent_ch = vae.config.latent_channels
    print(f"  in_ch={in_ch}, latent_ch={latent_ch}, scaling={args.scaling_factor}", flush=True)

    # Load image IDs
    ids = load_image_ids(args.csv)
    if args.limit > 0:
        ids = ids[:args.limit]
    print(f"  {len(ids)} unique image IDs to encode", flush=True)

    os.makedirs(args.out, exist_ok=True)

    shard_idx = 0
    shard_ids = []
    shard_lats = []
    total = 0
    skipped = 0
    t0 = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_ids, shard_lats
        if not shard_ids:
            return
        path = os.path.join(args.out, f"shard_{shard_idx:05d}.npz")
        np.savez(path,
                 latents=np.stack(shard_lats).astype(np.float16),
                 img_ids=np.array(shard_ids, dtype=np.int64))
        print(f"  saved {path}: {len(shard_ids)} latents", flush=True)
        shard_idx += 1
        shard_ids = []
        shard_lats = []

    batch_imgs = []
    batch_ids = []

    def flush_batch():
        nonlocal batch_imgs, batch_ids, shard_ids, shard_lats
        if not batch_imgs:
            return
        t = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            z = vae.encode(t).latent_dist.sample().mul_(args.scaling_factor)
        lats = z.cpu().float().numpy().astype(np.float16)
        for i, lat in enumerate(lats):
            shard_lats.append(lat)
            shard_ids.append(batch_ids[i])
        if len(shard_lats) >= args.shard_size:
            flush_shard()
        batch_imgs = []
        batch_ids = []

    for iid in ids:
        path = os.path.join(args.img_root, f"{iid}.png")
        if not os.path.exists(path):
            skipped += 1
            continue
        try:
            img = Image.open(path).convert("RGB").resize((256, 256))
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
        except Exception as e:
            print(f"  [skip] {iid}: {e}", flush=True)
            skipped += 1
            continue
        batch_imgs.append(t)
        batch_ids.append(iid)
        total += 1
        if len(batch_imgs) >= args.batch:
            flush_batch()
            if total % 5000 == 0:
                elapsed = time.time() - t0
                eta = elapsed / total * (len(ids) - total) if total > 0 else 0
                print(f"  ... {total}/{len(ids)} ({total/len(ids)*100:.1f}%, "
                      f"{elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    flush_batch()
    flush_shard()
    print(f"\nDone: {total} encoded, {skipped} skipped, "
          f"{shard_idx} shards in {args.out}, "
          f"{time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()

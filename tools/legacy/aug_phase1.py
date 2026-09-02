# -*- coding: utf-8 -*-
"""Phase 1: generate augmented images → VAE encode → save temp shards.

Saves augmented latents to a temp directory first, then Phase 2 merges
with original latents efficiently.
"""
import os, sys, csv, re, time, argparse, glob, multiprocessing as mp
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from PIL import Image


def count_combos(csv_path):
    rows, combo = [], Counter()
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            combo[(r["script"], r["character"], r["calligrapher"])] += 1
            rows.append(r)
    return rows, combo


def detect_ink_bbox(arr):
    mask = arr < 0.5
    if mask.sum() < 16: return None
    ys, xs = np.where(mask)
    return (ys.min(), ys.max(), xs.min(), xs.max())


def augment_one(args):
    img_path, template, seed_delta = args
    rng = np.random.RandomState(seed_delta)
    img = Image.open(img_path).convert("L").resize((256, 256), Image.LANCZOS)
    W, H = 256, 256

    if template == 0:
        dx, dy = rng.uniform(-6, -2), rng.uniform(-6, -2)
        scale = rng.uniform(0.94, 0.97)
        angle = rng.uniform(-5, -2)
        contrast = rng.uniform(0.95, 1.05)
    elif template == 1:
        dx, dy = rng.uniform(2, 6), rng.uniform(2, 6)
        scale = rng.uniform(1.02, 1.06)
        angle = rng.uniform(2, 5)
        contrast = rng.uniform(0.95, 1.05)
    else:
        dx, dy = rng.uniform(-3, 3), rng.uniform(-3, 3)
        scale = rng.uniform(0.95, 0.98)
        angle = rng.uniform(-2, 2)
        contrast = rng.uniform(0.95, 1.05)

    if abs(angle) > 1.0:
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    if scale < 1.0:
        new_w, new_h = max(8, int(W * scale)), max(8, int(H * scale))
        small = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("L", (W, H), 255)
        ox = max(0, min((W - new_w) // 2 + int(dx), W - new_w))
        oy = max(0, min((H - new_h) // 2 + int(dy), H - new_h))
        canvas.paste(small, (ox, oy))
        img = canvas
    else:
        new_w, new_h = int(W * scale), int(H * scale)
        arr0 = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        big = img.resize((new_w, new_h), Image.LANCZOS)
        bbox = detect_ink_bbox(arr0)
        if bbox is not None:
            by0, by1, bx0, bx1 = [int(v * scale) for v in bbox]
            cx, cy = (bx0 + bx1) // 2 + int(dx), (by0 + by1) // 2 + int(dy)
        else:
            cx, cy = new_w // 2 + int(dx), new_h // 2 + int(dy)
        x0 = max(0, min(cx - W // 2, new_w - W))
        y0 = max(0, min(cy - H // 2, new_h - H))
        img = big.crop((x0, y0, x0 + W, y0 + H))

    if abs(contrast - 1.0) > 0.01:
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        arr = np.clip(arr * contrast, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    a = np.stack([a, a, a], axis=-1)
    return np.transpose(a, (2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="5script/train_3top30_nobeike.csv")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--vae", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--out-temp", default="/tmp/aug_tmp")
    ap.add_argument("--target", type=int, default=4)
    ap.add_argument("--scaling-factor", type=float, default=0.18215)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PHASE 1: Augment + VAE encode | Device: {device}, workers={args.workers}")
    os.makedirs(args.out_temp, exist_ok=True)

    # ── 1. Count combos ──
    print(f"Loading {args.train_csv}...")
    rows, combo = count_combos(args.train_csv)
    need_aug = {k: args.target - n for k, n in combo.items() if n < args.target}
    total_needed = sum(need_aug.values())
    print(f"  {len(rows)} images, {len(combo)} combos, need {total_needed} aug")

    # ── 2. Build task list ──
    combo2rows = defaultdict(list)
    for r in rows:
        combo2rows[(r["script"], r["character"], r["calligrapher"])].append(r)
    rng = np.random.RandomState(args.seed)
    tasks, aug_meta = [], []
    next_id = 1000000
    for key, n_needed in sorted(need_aug.items(), key=lambda x: -x[1]):
        script, char, calli = key
        src = rng.choice(combo2rows[key])
        iid = int(re.search(r"(\d+)\.png", src["image_path"]).group(1))
        img_path = os.path.join(args.img_root, f"{iid}.png")
        if not os.path.exists(img_path): continue
        for vi in range(n_needed):
            tasks.append((img_path, vi % 3, args.seed + vi))
            aug_meta.append((next_id, script, char, calli,
                             src["calligrapher_id"], src["character_id"], src["glyph_id"]))
            next_id += 1
    print(f"  {len(tasks)} augmentation tasks")

    # ── 3. CPU augment ──
    t0 = time.time()
    print("CPU augmentation...")
    with mp.Pool(args.workers) as pool:
        results = list(pool.imap(augment_one, tasks, chunksize=256))
    print(f"  {len(results)} images in {time.time()-t0:.0f}s ({len(results)/(time.time()-t0):.0f}/s)")

    # ── 4. Load VAE ──
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)

    # ── 5. Encode + save to temp shards (incremental) ──
    tt = time.time()
    vae_bs = 32
    meta_csv_lines = []
    shard_idx = 0
    lat_buf, id_buf = [], []

    for batch_start in range(0, len(results), vae_bs):
        batch_end = min(batch_start + vae_bs, len(results))
        xs = torch.from_numpy(np.stack(results[batch_start:batch_end], axis=0)).to(device)
        with torch.no_grad():
            lat = vae.encode(xs).latent_dist.sample() * args.scaling_factor
            if lat.shape[-1] != 32:
                lat = torch.nn.functional.interpolate(lat, size=32, mode="bilinear")
            lat = lat.float().cpu().numpy()

        for vi, l in enumerate(lat):
            idx = batch_start + vi
            lat_buf.append(l.astype(np.float16))
            id_buf.append(aug_meta[idx][0])
            if len(lat_buf) >= 5000:
                _save_shard(lat_buf, id_buf, args.out_temp, shard_idx)
                shard_idx += 1
                lat_buf, id_buf = [], []

        # Write meta CSV lines for this batch
        for vi in range(batch_end - batch_start):
            new_id, script, char, calli, caid, chid, gid = aug_meta[batch_start + vi]
            meta_csv_lines.append(f"{new_id},{script},{char},{calli},{caid},{chid},{gid}")

        if (batch_start // vae_bs) % 128 == 0 or batch_end == len(results):
            print(f"  ... {batch_end}/{len(results)} encoded, {(time.time()-tt)/60:.1f}min", flush=True)

    # Flush remaining
    if lat_buf:
        _save_shard(lat_buf, id_buf, args.out_temp, shard_idx)
        shard_idx += 1

    # Save meta CSV
    meta_path = os.path.join(args.out_temp, "aug_meta.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("new_id,script,char,calli,calli_id,char_id,glyph_id\n")
        f.writelines("\n".join(meta_csv_lines))
    print(f"  Saved {len(meta_csv_lines)} meta rows to {meta_path}")
    print(f"PHASE 1 done: {len(results)} aug images in {(time.time()-t0)/60:.1f}min")


def _save_shard(lat_buf, id_buf, out_dir, idx):
    path = os.path.join(out_dir, f"shard_{idx:05d}.npz")
    np.savez_compressed(path,
        latents=np.stack(lat_buf, axis=0),
        img_ids=np.array(id_buf, dtype=np.int64),
    )


if __name__ == "__main__":
    main()
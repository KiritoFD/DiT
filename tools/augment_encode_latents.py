# -*- coding: utf-8 -*-
"""Targeted augmentation: rare-combo oversampling + VAE re-encode + merged shard.

Phase A of the experiment plan.
Uses multiprocessing for CPU PIL augmentation, GPU batch VAE encode.
"""
import os, sys, csv, re, time, argparse, math, glob, multiprocessing as mp
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from PIL import Image
from functools import partial


def count_combos(csv_path):
    """Count (script, char, calli) samples per combo. Return list of rows + Counter."""
    rows = []
    combo = Counter()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["script"], r["character"], r["calligrapher"])
            combo[key] += 1
            rows.append(r)
    return rows, combo


def detect_ink_bbox(arr):
    """arr: (H,W) float [0,1]. Return ink bbox (y0,y1,x0,x1) or None."""
    mask = arr < 0.5
    if mask.sum() < 16:
        return None
    ys, xs = np.where(mask)
    return (ys.min(), ys.max(), xs.min(), xs.max())


def augment_one(args):
    """CPU-only: read source image → generate 1 variant → return numpy [-1,1] (3,256,256).
    Called by multiprocessing Pool. args = (img_path, template, seed_delta)"""
    img_path, template, seed_delta = args
    rng = np.random.RandomState(seed_delta)
    img = Image.open(img_path).convert("L").resize((256, 256), Image.LANCZOS)
    W, H = 256, 256

    # --- parameter ranges per template (ensures inter-variant distinguishability) ---
    if template == 0:
        dx = rng.uniform(-6, -2)
        dy = rng.uniform(-6, -2)
        scale = rng.uniform(0.94, 0.97)
        angle = rng.uniform(-5, -2)
        contrast = rng.uniform(0.95, 1.05)
    elif template == 1:
        dx = rng.uniform(2, 6)
        dy = rng.uniform(2, 6)
        scale = rng.uniform(1.02, 1.06)
        angle = rng.uniform(2, 5)
        contrast = rng.uniform(0.95, 1.05)
    else:
        dx = rng.uniform(-3, 3)
        dy = rng.uniform(-3, 3)
        scale = rng.uniform(0.95, 0.98)
        angle = rng.uniform(-2, 2)
        contrast = rng.uniform(0.95, 1.05)

    # --- rotation ---
    if abs(angle) > 1.0:
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

    # --- scale + shift ---
    if scale < 1.0:
        new_w = max(8, int(W * scale))
        new_h = max(8, int(H * scale))
        small = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("L", (W, H), 255)
        ox = (W - new_w) // 2 + int(dx)
        oy = (H - new_h) // 2 + int(dy)
        ox = max(0, min(ox, W - new_w))
        oy = max(0, min(oy, H - new_h))
        canvas.paste(small, (ox, oy))
        img = canvas
    else:
        new_w = int(W * scale)
        new_h = int(H * scale)
        arr0 = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        big = img.resize((new_w, new_h), Image.LANCZOS)
        bbox = detect_ink_bbox(arr0)
        if bbox is not None:
            by0, by1, bx0, bx1 = [int(v * scale) for v in bbox]
            cx = (bx0 + bx1) // 2 + int(dx)
            cy = (by0 + by1) // 2 + int(dy)
        else:
            cx, cy = new_w // 2 + int(dx), new_h // 2 + int(dy)
        half = W // 2
        x0 = max(0, min(cx - half, new_w - W))
        y0 = max(0, min(cy - half, new_h - W))
        img = big.crop((x0, y0, x0 + W, y0 + W))

    # --- contrast ---
    if abs(contrast - 1.0) > 0.01:
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        arr = np.clip(arr * contrast, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    # --- final numpy array ---
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1,1]
    a = np.stack([a, a, a], axis=-1)  # (256,256,3)
    return np.transpose(a, (2, 0, 1))  # (3,256,256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="5script/train_3top30_nobeike.csv")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--latent-dir", default="final_latents")
    ap.add_argument("--vae", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--out-latent", default="final_latents_aug")
    ap.add_argument("--out-csv", default="5script/train_3top30_aug.csv")
    ap.add_argument("--target", type=int, default=4, help="Target samples per combo")
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--scaling-factor", type=float, default=0.18215)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32,
                    help="Multiprocessing workers for CPU PIL augmentation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, CPU cores: {os.cpu_count()}, workers: {args.workers}")
    os.makedirs(args.out_latent, exist_ok=True)

    # ── 1. Load train CSV + count combos ──
    print(f"Loading train CSV: {args.train_csv}")
    rows, combo = count_combos(args.train_csv)
    N_orig = len(rows)
    print(f"  Total images: {N_orig}, unique combos: {len(combo)}")

    # ── 2. Determine which combos need augmentation ──
    need_aug = {}
    for key, n in combo.items():
        if n < args.target:
            need_aug[key] = args.target - n
    total_needed = sum(need_aug.values())
    print(f"  Combos needing augmentation: {len(need_aug)} "
          f"(total augment images: {total_needed})")

    if total_needed == 0:
        print("  Nothing to augment!")
        return

    # ── 3. Build augmentation task list ──
    combo2rows = defaultdict(list)
    for r in rows:
        key = (r["script"], r["character"], r["calligrapher"])
        combo2rows[key].append(r)

    rng = np.random.RandomState(args.seed)
    tasks = []  # (img_path, template, seed_delta)
    aug_meta = []  # (new_id, script, char, calli, calli_id, char_id, glyph_id)
    next_id = 1000000
    for key, n_needed in sorted(need_aug.items(), key=lambda x: -x[1]):
        script, char, calli = key
        src = rng.choice(combo2rows[key])
        iid = int(re.search(r"(\d+)\.png", src["image_path"]).group(1))
        img_path = os.path.join(args.img_root, f"{iid}.png")
        if not os.path.exists(img_path):
            print(f"  [skip] missing {img_path}")
            continue
        for vi in range(n_needed):
            tasks.append((img_path, vi % 3, args.seed + vi))
            aug_meta.append((next_id, script, char, calli,
                             src["calligrapher_id"], src["character_id"], src["glyph_id"]))
            next_id += 1

    print(f"  Total augmentation tasks: {len(tasks)}")

    # ── 4. CPU parallel augmentation ──
    t1 = time.time()
    print(f"CPU augmentation with {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        results = list(pool.imap(augment_one, tasks, chunksize=256))
    cpu_time = time.time() - t1
    print(f"  CPU augmentation done: {len(results)} images in {cpu_time:.0f}s "
          f"({len(results)/cpu_time:.0f} img/s)")

    # ── 5. VAE encode on GPU ──
    print(f"Loading VAE from {args.vae}...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    t2 = time.time()
    print(f"GPU VAE encode with batch=32...")
    vae_bs = 32
    aug_latents = [(None, None)] * len(results)  # pre-allocate

    for batch_start in range(0, len(results), vae_bs):
        batch_end = min(batch_start + vae_bs, len(results))
        xs = torch.from_numpy(np.stack(results[batch_start:batch_end], axis=0)).to(device)
        with torch.no_grad():
            lat = vae.encode(xs).latent_dist.sample()
            lat = lat * args.scaling_factor
            if lat.shape[-1] != 32:
                lat = torch.nn.functional.interpolate(lat, size=32, mode="bilinear")
            lat = lat.float().cpu().numpy()
        for vi, l in enumerate(lat):
            aug_latents[batch_start + vi] = (aug_meta[batch_start + vi][0], l)

        if (batch_start // vae_bs) % 128 == 0 or batch_end == len(results):
            print(f"  ... {batch_end}/{len(results)} encoded, "
                  f"{(time.time()-t2)/60:.1f}min", flush=True)

    gpu_time = time.time() - t2
    print(f"  VAE encode done in {gpu_time:.0f}s ({len(results)/gpu_time:.0f} img/s)")

    # ── 6. Read original latents ──
    t3 = time.time()
    print(f"Reading original latents from {args.latent_dir}...")
    id2lat = {}
    shards = sorted(glob.glob(os.path.join(args.latent_dir, "shard_*.npz")))
    for sp in shards:
        d = np.load(sp)
        for j, iid in enumerate(d["img_ids"]):
            id2lat[int(iid)] = d["latents"][j]
        d.close()
    print(f"  Loaded {len(id2lat)} latents in {time.time()-t3:.1f}s")

    # ── 7. Write merged shards ──
    N_total = N_orig + len(aug_latents)
    n_shards = (N_total + args.shard_size - 1) // args.shard_size
    print(f"Writing {N_total} samples to {args.out_latent} ({n_shards} shards)...")
    t4 = time.time()

    # Build merged list: original + augmented
    merged = []
    for r in rows:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        merged.append((id2lat[iid], iid))
    for new_id, lat in aug_latents:
        merged.append((lat, new_id))

    for shard_start in range(0, N_total, args.shard_size):
        shard_end = min(shard_start + args.shard_size, N_total)
        lat_cat = np.stack([merged[i][0] for i in range(shard_start, shard_end)], axis=0).astype(np.float16)
        ids_cat = np.array([merged[i][1] for i in range(shard_start, shard_end)], dtype=np.int64)
        shard_idx = shard_start // args.shard_size
        out_path = os.path.join(args.out_latent, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(out_path, latents=lat_cat, img_ids=ids_cat)
        if (shard_idx + 1) % 10 == 0 or (shard_idx + 1) == n_shards:
            print(f"  [{shard_idx+1}/{n_shards}] {out_path}: {lat_cat.shape} "
                  f"({(time.time()-t4)/60:.1f}min)", flush=True)

    print(f"  Shard writing done in {(time.time()-t4)/60:.1f}min")

    # ── 8. Write new CSV ──
    CSV_FIELDS = ["image_path", "calligrapher", "script", "character",
                  "calligrapher_id", "script_id", "character_id", "glyph_id"]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for new_id, script, char, calli, caid, chid, gid in aug_meta:
            w.writerow({
                "image_path": f"final_imgs_256/{new_id}.png",
                "calligrapher": calli, "script": script, "character": char,
                "calligrapher_id": caid, "script_id": "",
                "character_id": chid, "glyph_id": gid,
            })
    print(f"  CSV: {args.out_csv} ({N_total} rows)")

    # ── 9. Validation ──
    print("\n=== Validation ===")
    new_shards = sorted(glob.glob(os.path.join(args.out_latent, "shard_*.npz")))
    shard_ids = set()
    for sp in new_shards:
        d = np.load(sp)
        for iid in d["img_ids"]:
            shard_ids.add(int(iid))
        d.close()
    csv_ids = set()
    for r in rows:
        csv_ids.add(int(re.search(r"(\d+)\.png", r["image_path"]).group(1)))
    for new_id, *_ in aug_meta:
        csv_ids.add(new_id)
    missing = csv_ids - shard_ids
    if missing:
        print(f"  WARNING: {len(missing)} csv ids missing from shards")
    else:
        print("  ALL CSV img_ids present in shards ✓")
    print(f"  Total time: {(time.time()-t1)/60:.1f}min")


if __name__ == "__main__":
    main()
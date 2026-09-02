#!/usr/bin/env python3
"""Phase 2: merge original latents + augmented temp shards → final_latents_aug + CSV.

Reads original latents (final_latents/) and augmented temp shards (/tmp/aug_tmp/),
writes merged shards to final_latents_aug/ and merged CSV to 5script/train_3top30_aug.csv.

Single-pass: reads original shards once, interleaves aug latents, writes merged shards.
"""
import os, sys, csv, re, time, argparse, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

CSV_FIELDS = ["image_path", "calligrapher", "script", "character",
              "calligrapher_id", "script_id", "character_id", "glyph_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="5script/train_3top30_nobeike.csv")
    ap.add_argument("--latent-dir", default="final_latents")
    ap.add_argument("--aug-temp", default="/tmp/aug_tmp")
    ap.add_argument("--out-latent", default="final_latents_aug")
    ap.add_argument("--out-csv", default="5script/train_3top30_aug.csv")
    ap.add_argument("--shard-size", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_latent, exist_ok=True)
    t0 = time.time()

    # ── 1. Load original latents ──
    print(f"Reading original latents from {args.latent_dir}...")
    orig_shards = sorted(glob.glob(os.path.join(args.latent_dir, "shard_*.npz")))
    orig_latents = []
    orig_ids = []
    for sp in orig_shards:
        d = np.load(sp)
        orig_latents.append(d["latents"])
        orig_ids.append(d["img_ids"])
        d.close()
    orig_lat = np.concatenate(orig_latents, axis=0)  # (N, C, H, W) fp16
    orig_iid = np.concatenate(orig_ids, axis=0)       # (N,) int64
    del orig_latents, orig_ids
    N_orig = len(orig_iid)
    print(f"  {N_orig} original latents, shape={orig_lat.shape} ({time.time()-t0:.0f}s)")

    # ── 2. Load augmented latents ──
    t1 = time.time()
    aug_shards = sorted(glob.glob(os.path.join(args.aug_temp, "shard_*.npz")))
    if not aug_shards:
        print("  No augmented shards found, skipping merge")
        aug_lat = np.empty((0, orig_lat.shape[1], orig_lat.shape[2], orig_lat.shape[3]), dtype=np.float16)
        aug_iid = np.array([], dtype=np.int64)
    else:
        aug_latents = []
        aug_ids = []
        for sp in aug_shards:
            d = np.load(sp)
            aug_latents.append(d["latents"])
            aug_ids.append(d["img_ids"])
            d.close()
        aug_lat = np.concatenate(aug_latents, axis=0)  # (M, C, H, W) fp16
        aug_iid = np.concatenate(aug_ids, axis=0)       # (M,) int64
        del aug_latents, aug_ids
    N_aug = len(aug_iid)
    print(f"  {N_aug} augmented latents, shape={aug_lat.shape} ({time.time()-t1:.0f}s)")

    # ── 3. Load original CSV rows ──
    print(f"Loading original CSV: {args.train_csv}")
    orig_rows = []
    with open(args.train_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            orig_rows.append(r)
    print(f"  {len(orig_rows)} rows")

    # ── 4. Load augmented meta ──
    aug_meta_path = os.path.join(args.aug_temp, "aug_meta.csv")
    aug_rows = []
    if os.path.exists(aug_meta_path):
        with open(aug_meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                aug_rows.append(r)
    print(f"  {len(aug_rows)} aug meta rows")

    # ── 5. Write merged shards (single pass, concatenate then save) ──
    t2 = time.time()
    N_total = N_orig + N_aug
    n_shards = (N_total + args.shard_size - 1) // args.shard_size
    print(f"Writing {N_total} samples to {args.out_latent} ({n_shards} shards)...")

    # Concatenate once
    all_lat = np.concatenate([orig_lat, aug_lat], axis=0)
    all_iid = np.concatenate([orig_iid, aug_iid], axis=0)
    del orig_lat, aug_lat, orig_iid, aug_iid

    for shard_start in range(0, N_total, args.shard_size):
        shard_end = min(shard_start + args.shard_size, N_total)
        si = shard_start // args.shard_size
        out_path = os.path.join(args.out_latent, f"shard_{si:05d}.npz")
        np.savez_compressed(out_path,
            latents=all_lat[shard_start:shard_end],
            img_ids=all_iid[shard_start:shard_end],
        )
        if (si + 1) % 10 == 0 or (si + 1) == n_shards:
            print(f"  [{si+1}/{n_shards}] {out_path} ({(time.time()-t2)/60:.1f}min)", flush=True)

    del all_lat, all_iid
    print(f"  Shard writing done in {(time.time()-t2)/60:.1f}min")

    # ── 6. Write merged CSV ──
    t3 = time.time()
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in orig_rows:
            w.writerow(r)
        for r in aug_rows:
            w.writerow({
                "image_path": f"final_imgs_256/{r['new_id']}.png",
                "calligrapher": r["calli"],
                "script": r["script"],
                "character": r["char"],
                "calligrapher_id": r["calli_id"],
                "script_id": "",
                "character_id": r["char_id"],
                "glyph_id": r["glyph_id"],
            })
    print(f"  CSV: {args.out_csv} ({len(orig_rows) + len(aug_rows)} rows, {time.time()-t3:.0f}s)")

    # ── 7. Validation ──
    print("\n=== Validation ===")
    new_shards = sorted(glob.glob(os.path.join(args.out_latent, "shard_*.npz")))
    shard_ids = set()
    for sp in new_shards:
        d = np.load(sp)
        for iid in d["img_ids"]:
            shard_ids.add(int(iid))
        d.close()
    csv_ids = set()
    for r in orig_rows:
        csv_ids.add(int(re.search(r"(\d+)\.png", r["image_path"]).group(1)))
    for r in aug_rows:
        csv_ids.add(int(r["new_id"]))
    missing = csv_ids - shard_ids
    if missing:
        print(f"  WARNING: {len(missing)} csv ids missing from shards: {sorted(missing)[:5]}")
    else:
        print("  ALL CSV img_ids present in shards ✓")
    print(f"  Total time: {(time.time()-t0)/60:.1f}min")
    print("Done!")


if __name__ == "__main__":
    main()
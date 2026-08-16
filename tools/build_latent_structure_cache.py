"""Build a compact, row-aligned cache for training a latent structure probe."""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latent_dataset import MCCDLatentDataset


def _reduce_map(task):
    index, path, size = task
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8) > 127
    if array.shape != (size * 8, size * 8):
        raise ValueError(f"expected {size * 8}x{size * 8}, got {array.shape}: {path}")
    reduced = array.reshape(size, 8, size, 8).max(axis=(1, 3))
    return index, np.packbits(reduced.reshape(-1))


def _image_id(row):
    match = re.search(r"(\d+)\.png", row["image_path"])
    if not match:
        raise ValueError(f"cannot parse image id from {row['image_path']!r}")
    return int(match.group(1))


def _build_maps(ids, root, size, workers, name):
    packed_width = (size * size + 7) // 8
    output = np.empty((len(ids), packed_width), dtype=np.uint8)
    tasks = [(i, os.path.join(root, f"{image_id}.png"), size)
             for i, image_id in enumerate(ids)]
    started = time.time()
    with mp.Pool(workers) as pool:
        for completed, (index, packed) in enumerate(
                pool.imap_unordered(_reduce_map, tasks, chunksize=256), 1):
            output[index] = packed
            if completed % 50000 == 0:
                print(f"[{name}] {completed:,}/{len(ids):,}")
    print(f"[{name}] complete in {time.time() - started:.1f}s")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="5script/train.csv")
    parser.add_argument("--latent-shards-dir", default="final_latents")
    parser.add_argument("--canny-root", default="final_canny")
    parser.add_argument("--skel-root", default="final_skeleton")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.size != 32:
        raise ValueError("current cache builder expects 256 -> 32 exact max pooling")

    dataset = MCCDLatentDataset(
        args.csv, args.latent_shards_dir, img_root=None, preload=True,
        load_image=False, load_canny=False, load_skel=False)
    ids = np.asarray([_image_id(row) for row in dataset.samples], dtype=np.int64)
    canny = _build_maps(ids, args.canny_root, args.size, args.workers, "canny")
    skeleton = _build_maps(ids, args.skel_root, args.size, args.workers, "skeleton")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Uncompressed NPZ is intentional: fp16 latents dominate size and load much
    # faster locally; SSH compression can be enabled during transfer if useful.
    np.savez(
        args.out,
        latents=dataset._latents.astype(np.float16),
        canny_packed=canny,
        skeleton_packed=skeleton,
        image_ids=ids,
        size=np.asarray(args.size, dtype=np.int32),
    )
    metadata = {
        "rows": len(dataset),
        "latent_dtype": "float16",
        "latent_shape": list(dataset._latents.shape[1:]),
        "structure_size": args.size,
        "bit_packed": True,
        "source_csv": args.csv,
    }
    with open(args.out + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024 ** 3:.2f} GiB)")


if __name__ == "__main__":
    main()

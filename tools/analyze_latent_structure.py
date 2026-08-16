"""Measure whether raw VAE-latent gradients are usable as stroke supervision.

This is a read-only diagnostic.  It compares spatial gradient energy in cached
32x32 VAE latents with max-pooled Canny and skeleton targets.  Low correlation
or top-density IoU means that a learned, frozen latent-to-structure probe is
preferable to applying an edge loss directly to the four latent channels.
"""

import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latent_dataset import MCCDLatentDataset


def _image_id(row):
    match = re.search(r"(\d+)\.png", row["image_path"])
    if not match:
        raise ValueError(f"cannot parse image id from {row['image_path']!r}")
    return int(match.group(1))


def _load_binary(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array > 0.5).float()


def _gradient_energy(latents):
    dx = F.pad(latents[:, :, :, 1:] - latents[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(latents[:, :, 1:, :] - latents[:, :, :-1, :], (0, 0, 0, 1))
    return (dx.square() + dy.square()).mean(dim=1).sqrt()


def _pearson(a, b, dim=None):
    if dim is None:
        a, b = a.flatten(), b.flatten()
        dim = 0
    a = a - a.mean(dim=dim, keepdim=True)
    b = b - b.mean(dim=dim, keepdim=True)
    numerator = (a * b).sum(dim=dim)
    denominator = (a.square().sum(dim=dim) * b.square().sum(dim=dim)).sqrt()
    return numerator / denominator.clamp_min(1e-12)


def _summarize(energy, target):
    flat_e = energy.flatten(1)
    flat_t = target.flatten(1)
    per_image_corr = _pearson(flat_e, flat_t, dim=1)
    ious = []
    for e, t in zip(flat_e, flat_t):
        positives = max(int(t.sum().item()), 1)
        selected = torch.zeros_like(t, dtype=torch.bool)
        selected[e.topk(min(positives, e.numel())).indices] = True
        truth = t.bool()
        union = (selected | truth).sum().clamp_min(1)
        ious.append(float((selected & truth).sum() / union))
    return {
        "pooled_pearson": float(_pearson(flat_e, flat_t)),
        "mean_image_pearson": float(per_image_corr.mean()),
        "median_image_pearson": float(per_image_corr.median()),
        "mean_top_density_iou": float(np.mean(ious)),
        "target_occupancy": float(flat_t.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="5script/train.csv")
    parser.add_argument("--latent-shards-dir", default="final_latents")
    parser.add_argument("--canny-root", default="final_canny")
    parser.add_argument("--skel-root", default="final_skeleton")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dataset = MCCDLatentDataset(
        args.csv, args.latent_shards_dir, img_root=None, preload=True,
        load_image=False, load_canny=False, load_skel=False)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.n, len(dataset)))
    latents = torch.from_numpy(dataset._latents[indices]).float()

    cannys, skeletons = [], []
    for index in indices:
        image_id = _image_id(dataset.samples[index])
        cannys.append(_load_binary(os.path.join(args.canny_root, f"{image_id}.png")))
        skeletons.append(_load_binary(os.path.join(args.skel_root, f"{image_id}.png")))
    canny = torch.stack(cannys).unsqueeze(1)
    skeleton = torch.stack(skeletons).unsqueeze(1)
    # Preserve any thin positive line when reducing 256x256 maps to latent size.
    canny_32 = F.adaptive_max_pool2d(canny, (32, 32)).squeeze(1)
    skeleton_32 = F.adaptive_max_pool2d(skeleton, (32, 32)).squeeze(1)
    energy = _gradient_energy(latents)

    result = {
        "n": len(indices),
        "seed": args.seed,
        "latent_shape": list(latents.shape[1:]),
        "canny": _summarize(energy, canny_32),
        "skeleton": _summarize(energy, skeleton_32),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()

"""Train the small latent-to-Canny/skeleton probe on a local CUDA GPU."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import LatentStructureProbe


def _loss(logits, target):
    positives = target.sum(dim=(0, 2, 3))
    negatives = target.shape[0] * target.shape[2] * target.shape[3] - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(1, 10)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight.view(1, -1, 1, 1))
    probs = logits.sigmoid()
    intersection = (probs * target).sum(dim=(0, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6)
                / (probs.sum(dim=(0, 2, 3)) + positives + 1e-6))
    return bce + dice.mean()


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    intersections = torch.zeros(2, device=device)
    unions = torch.zeros(2, device=device)
    correct = torch.zeros(2, device=device)
    total = 0
    for latents, targets in loader:
        latents = latents.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True)
        predictions = model(latents).sigmoid() >= 0.5
        truth = targets.bool()
        intersections += (predictions & truth).sum(dim=(0, 2, 3))
        unions += (predictions | truth).sum(dim=(0, 2, 3))
        correct += (predictions == truth).sum(dim=(0, 2, 3))
        total += targets.shape[0] * targets.shape[2] * targets.shape[3]
    return {
        "canny_iou": float(intersections[0] / unions[0].clamp_min(1)),
        "skeleton_iou": float(intersections[1] / unions[1].clamp_min(1)),
        "canny_accuracy": float(correct[0] / total),
        "skeleton_accuracy": float(correct[1] / total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for probe training")
    device = torch.device("cuda")
    cache = np.load(args.cache)
    latents = torch.from_numpy(cache["latents"])
    size = int(cache["size"])
    count = latents.shape[0]
    canny = np.unpackbits(cache["canny_packed"], axis=1)[:, :size * size]
    skeleton = np.unpackbits(cache["skeleton_packed"], axis=1)[:, :size * size]
    targets = torch.from_numpy(
        np.stack([canny, skeleton], axis=1).reshape(count, 2, size, size).astype(np.float32))

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(count, generator=generator)
    val_count = max(int(count * args.val_fraction), 1)
    val_indices, train_indices = permutation[:val_count], permutation[val_count:]
    train_ds = TensorDataset(latents[train_indices], targets[train_indices])
    val_ds = TensorDataset(latents[val_indices], targets[val_indices])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = LatentStructureProbe(width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    history = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"device={torch.cuda.get_device_name(0)} rows={count:,} "
          f"train={len(train_ds):,} val={len(val_ds):,} params="
          f"{sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        started = time.time()
        for latents_batch, targets_batch in train_loader:
            latents_batch = latents_batch.to(device, non_blocking=True).float()
            targets_batch = targets_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = _loss(model(latents_batch), targets_batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * latents_batch.shape[0]
            seen += latents_batch.shape[0]
        metrics = _evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, train_loss=loss_sum / seen,
                       seconds=time.time() - started,
                       peak_vram_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
        history.append(metrics)
        score = metrics["skeleton_iou"]
        print(json.dumps(metrics))
        if score > best:
            best = score
            torch.save({
                "model": {key: value.detach().cpu()
                          for key, value in model.state_dict().items()},
                "args": vars(args),
                "metrics": metrics,
                "history": history,
            }, args.out)
    with open(args.out + ".json", "w", encoding="utf-8") as handle:
        json.dump({"best_skeleton_iou": best, "history": history}, handle, indent=2)
    print(f"best skeleton IoU={best:.4f}; wrote {args.out}")


if __name__ == "__main__":
    main()

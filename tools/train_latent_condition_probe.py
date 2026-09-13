"""Train a held-out evaluator for character/calligrapher/script adherence."""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latent_condition_probe import LatentConditionProbe


def _topk(logits, targets, k):
    return logits.topk(k, dim=1).indices.eq(targets[:, None]).any(dim=1).sum()


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = torch.zeros(5, dtype=torch.int64, device=device)
    total = 0
    for latents, chars, calligs, scripts in loader:
        latents = latents.to(device, non_blocking=True).float()
        chars = chars.to(device, non_blocking=True)
        calligs = calligs.to(device, non_blocking=True)
        scripts = scripts.to(device, non_blocking=True)
        char_logits, callig_logits, script_logits = model(latents)
        correct[0] += _topk(char_logits, chars, 1)
        correct[1] += _topk(char_logits, chars, 5)
        correct[2] += _topk(callig_logits, calligs, 1)
        correct[3] += _topk(callig_logits, calligs, 5)
        correct[4] += _topk(script_logits, scripts, 1)
        total += latents.shape[0]
    values = correct.float() / total
    return {
        "char_top1": float(values[0]),
        "char_top5": float(values[1]),
        "callig_top1": float(values[2]),
        "callig_top5": float(values[3]),
        "script_top1": float(values[4]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-characters", type=int, default=7026)
    parser.add_argument("--num-calligraphers", type=int, default=1011)
    parser.add_argument("--num-scripts", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=.05)
    parser.add_argument("--char-alpha", type=float, default=.35)
    parser.add_argument("--callig-alpha", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    cache = np.load(args.cache)
    latents = torch.from_numpy(cache["latents"])
    with open(args.csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(latents):
        raise ValueError(f"CSV/cache row mismatch: {len(rows)} != {len(latents)}")
    chars = torch.tensor([int(row["character_id"]) for row in rows])
    calligs = torch.tensor([int(row["calligrapher_id"]) for row in rows])
    scripts = torch.tensor([int(row["script_id"]) for row in rows])

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(rows), generator=generator)
    val_count = max(int(len(rows) * args.val_fraction), 1)
    val_indices, train_indices = permutation[:val_count], permutation[val_count:]
    train_ds = TensorDataset(latents[train_indices], chars[train_indices],
                             calligs[train_indices], scripts[train_indices])
    val_ds = TensorDataset(latents[val_indices], chars[val_indices],
                           calligs[val_indices], scripts[val_indices])

    char_counts = torch.bincount(chars[train_indices], minlength=args.num_characters).float()
    callig_counts = torch.bincount(calligs[train_indices], minlength=args.num_calligraphers).float()
    sample_weights = (char_counts[chars[train_indices]].clamp_min(1).pow(-args.char_alpha)
                      * callig_counts[calligs[train_indices]].clamp_min(1).pow(-args.callig_alpha))
    sampler = WeightedRandomSampler(sample_weights.double(), len(train_ds),
                                    replacement=True, generator=generator)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = LatentConditionProbe(
        args.num_characters, args.num_calligraphers, args.num_scripts,
        width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * .1)
    best = -1.0
    history = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"device={torch.cuda.get_device_name(0)} train={len(train_ds):,} "
          f"val={len(val_ds):,} params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        started = time.time()
        for latents_batch, chars_batch, calligs_batch, scripts_batch in train_loader:
            latents_batch = latents_batch.to(device, non_blocking=True).float()
            chars_batch = chars_batch.to(device, non_blocking=True)
            calligs_batch = calligs_batch.to(device, non_blocking=True)
            scripts_batch = scripts_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                char_logits, callig_logits, script_logits = model(latents_batch)
                loss = (F.cross_entropy(char_logits, chars_batch, label_smoothing=.02)
                        + .5 * F.cross_entropy(callig_logits, calligs_batch, label_smoothing=.02)
                        + .2 * F.cross_entropy(script_logits, scripts_batch, label_smoothing=.02))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * latents_batch.shape[0]
            seen += latents_batch.shape[0]
        scheduler.step()
        metrics = _evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, train_loss=loss_sum / seen,
                       seconds=time.time() - started,
                       peak_vram_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
        history.append(metrics)
        print(json.dumps(metrics))
        score = metrics["char_top1"]
        if score > best:
            best = score
            torch.save({
                "model": {key: value.detach().cpu()
                          for key, value in model.state_dict().items()},
                "args": vars(args), "metrics": metrics, "history": history,
            }, args.out)
    with open(args.out + ".json", "w", encoding="utf-8") as handle:
        json.dump({"best_char_top1": best, "history": history}, handle, indent=2)
    print(f"best char top1={best:.4f}; wrote {args.out}")


if __name__ == "__main__":
    main()

"""Train glyph classifier on 3-top30 VAE latents.

Input: pretrained_models/3top30_latents.npz (41029, 4, 32, 32) + 5script/train_3top30_nobeike.csv
Output: glyph_classifier_ckpts/best.pt

Usage:
  python train_glyph_classifier.py
  python train_glyph_classifier.py --epochs 50 --batch 256 --lr 3e-4
"""
import os, sys, csv, json, argparse, time
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from glyph_classifier import GlyphLatentClassifier

# ── Config ──────────────────────────────────────────────────────────────────
LATENTS_PATH = "pretrained_models/3top30_latents.npz"
TRAIN_CSV = "5script/train_3top30_nobeike.csv"
EVAL_CSV = "5script/eval100_3top30.csv"
CKPT_DIR = "glyph_classifier_ckpts"
NUM_CLASSES = 9401
LABEL_SMOOTHING = 0.1


class LatentGlyphDataset(Dataset):
    """Load (latent, label) pairs from a single npz + csv.

    Supports two modes:
      - label_mode='glyph': classify by glyph_id (9401 classes, script+char)
      - label_mode='char': classify by character_id (4578 classes, script-agnostic)

    Labels are remapped to contiguous 0..N-1 for cross_entropy.
    """
    def __init__(self, npz_path, csv_path, glyph_remap=None, label_mode='glyph'):
        data = np.load(npz_path)
        self.latents = data['latents']  # (N, 4, 32, 32) float32
        self.img_ids = data['img_ids']  # (N,) int64
        id2idx = {int(iid): idx for idx, iid in enumerate(self.img_ids)}
        self.label_mode = label_mode

        label_col = 'character_id' if label_mode == 'char' else 'glyph_id'

        # Build label remap if not provided
        raw_labels = set()
        rows_raw = []
        with open(csv_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                iid = int(os.path.basename(r['image_path']).replace('.png', ''))
                if iid in id2idx:
                    raw_labels.add(int(r[label_col]))
                    rows_raw.append((iid, int(r[label_col])))
        if glyph_remap is None:
            self.glyph_remap = {g: i for i, g in enumerate(sorted(raw_labels))}
        else:
            self.glyph_remap = glyph_remap
        self.num_classes = len(self.glyph_remap)

        self.samples = []
        missing = 0
        for iid, gid in rows_raw:
            if iid in id2idx:
                self.samples.append((id2idx[iid], self.glyph_remap[gid]))
            else:
                missing += 1
        if missing:
            print(f"  [warn] {missing} csv rows have no matching latent (skipped)")
        print(f"  dataset: {len(self.samples)} samples, {self.num_classes} classes ({label_mode}) from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lat_idx, glyph_id = self.samples[idx]
        return self.latents[lat_idx], glyph_id


def evaluate(model, loader, device, max_batches=None, topk=5):
    model.eval()
    correct, total = 0, 0
    topk_correct = 0
    loss_sum = 0.0
    nb = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x, return_embed=False)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            # top-k accuracy
            if topk > 1:
                _, topk_pred = logits.topk(topk, dim=1)
                topk_correct += (topk_pred == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
            loss_sum += loss.item()
            nb += 1
    acc = correct / max(total, 1)
    topk_acc = topk_correct / max(total, 1) if topk > 1 else acc
    return acc, topk_acc, loss_sum / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--latent-noise", type=float, default=0.1,
                    help="Std of additive Gaussian noise injected on latents during training (0=off)")
    ap.add_argument("--eval-split", type=float, default=0.1,
                    help="Fraction of train set held out for validation (eval.csv is too small & overlaps glyphs)")
    ap.add_argument("--class-balanced", action="store_true",
                    help="Use class-balanced sampling (WeightedRandomSampler)")
    ap.add_argument("--label-mode", type=str, default="glyph", choices=["glyph", "char"],
                    help="Classify by 'glyph' (9401 cls, script+char) or 'char' (4578 cls, char only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading data...")
    full_ds = LatentGlyphDataset(LATENTS_PATH, TRAIN_CSV, glyph_remap=None,
                                 label_mode=args.label_mode)
    actual_num_classes = full_ds.num_classes
    if actual_num_classes != NUM_CLASSES:
        print(f"  [info] actual classes = {actual_num_classes} (config default {NUM_CLASSES}), using actual")

    # Split: use train_3top30_nobeike for both train and val (eval100 overlaps)
    n_total = len(full_ds)
    n_val = int(n_total * args.eval_split)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], g)
    print(f"  train: {n_train}, val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = GlyphLatentClassifier(actual_num_classes, latent_channels=4,
                                  embed_dim=args.embed_dim,
                                  dropout=args.dropout,
                                  latent_noise_std=args.latent_noise).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model: {nparams/1e6:.2f}M params, {actual_num_classes} classes, "
          f"dropout={args.dropout}, latent_noise={args.latent_noise}")

    # save remap for later use (eval, loss integration)
    with open(os.path.join(CKPT_DIR, "glyph_remap.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in full_ds.glyph_remap.items()}, f)

    # ── Class-balanced sampler ──────────────────────────────────────────────
    from collections import Counter
    train_targets = [full_ds.samples[i][1] for i in train_ds.indices]
    class_counts = Counter(train_targets)
    if args.class_balanced:
        sample_weights = [1.0 / max(class_counts[t], 1) for t in train_targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_targets), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=0, pin_memory=True, drop_last=True)
        print(f"  class-balanced sampling: ON")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Train ───────────────────────────────────────────────────────────────
    best_acc = 0.0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        nb = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            nb += 1

        sched.step()
        val_acc, val_topk, val_loss = evaluate(model, val_loader, device, topk=5)
        train_avg = train_loss / nb

        elapsed = time.time() - t0
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_topk': val_topk,
                'val_loss': val_loss,
                'config': vars(args),
                'num_classes': actual_num_classes,
            }, os.path.join(CKPT_DIR, "best.pt"))
            mark = " *"
        print(f"epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss {train_avg:.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"val_acc {val_acc:.4f} top5 {val_topk:.4f} | "
              f"lr {sched.get_last_lr()[0]:.2e} | "
              f"{elapsed:.0f}s{mark}")

    print(f"\nDone. best val_acc={best_acc:.4f} @ epoch {best_epoch+1}")
    print(f"saved: {os.path.join(CKPT_DIR, 'best.pt')}")

    # also save final with topk info
    torch.save({
        'model': model.state_dict(),
        'epoch': args.epochs - 1,
        'val_acc': val_acc,
        'val_topk': val_topk,
        'val_loss': val_loss,
        'config': vars(args),
        'num_classes': actual_num_classes,
    }, os.path.join(CKPT_DIR, "final.pt"))


if __name__ == "__main__":
    main()

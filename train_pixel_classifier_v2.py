"""Improved pixel classifier v2: data augmentation + multi-scale + mixup.

Key improvements over v1:
  - Random flip (H) + small rotation (±10°) augmentation
  - Mixup (alpha=0.2) for inter-class regularization
  - Cosine warmup (5 epochs) before cosine decay
  - Longer training (100 epochs) with early stopping patience
  - Label smoothing 0.1

Usage:
  python train_pixel_classifier_v2.py --label-mode char --epochs 100
"""
import os, sys, csv, json, argparse, time, math
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.stdout.reconfigure(encoding='utf-8')

PIXEL_NPZ = "_classifier_pixel64_data.npz"
TRAIN_CSV = "5script/train_3top30_nobeike.csv"
CKPT_DIR = "glyph_classifier_ckpts"
LABEL_SMOOTHING = 0.1

from train_pixel_classifier import PixelClassifier, ConvBlock


class PixelGlyphDataset(Dataset):
    """Load (image, label) from pixel npz + csv, with augmentation."""
    def __init__(self, npz_path, csv_path, label_mode='glyph', augment=False):
        data = np.load(npz_path)
        self.images = data['images'].astype(np.float32)  # (N, 1, 64, 64)
        self.img_ids = data['img_ids']
        id2idx = {int(iid): i for i, iid in enumerate(self.img_ids)}

        label_col = 'character_id' if label_mode == 'char' else 'glyph_id'
        raw_labels = set()
        rows_raw = []
        with open(csv_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                iid = int(os.path.basename(r['image_path']).replace('.png', ''))
                if iid in id2idx:
                    lid = int(r[label_col])
                    raw_labels.add(lid)
                    rows_raw.append((iid, lid))
        self.label_remap = {g: i for i, g in enumerate(sorted(raw_labels))}
        self.num_classes = len(self.label_remap)
        self.samples = [(id2idx[iid], self.label_remap[lid]) for iid, lid in rows_raw]
        self.augment = augment
        print(f"  pixel dataset: {len(self.samples)} samples, {self.num_classes} classes ({label_mode}), augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, label = self.samples[idx]
        img = self.images[img_idx]  # (1, 64, 64) float32

        if self.augment:
            # Random horizontal flip (calligraphy can be flipped for aug)
            if np.random.random() < 0.5:
                img = img[:, :, ::-1].copy()
            # Small rotation ±10° via affine
            angle = np.random.uniform(-10, 10)
            if abs(angle) > 0.5:
                img_t = torch.from_numpy(img).unsqueeze(0)
                angle_rad = angle * math.pi / 180
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                theta = torch.tensor([[cos_a, -sin_a, 0],
                                       [sin_a, cos_a, 0]], dtype=torch.float32)
                grid = F.affine_grid(theta.unsqueeze(0), img_t.shape, align_corners=False)
                img_t = F.grid_sample(img_t, grid, align_corners=False, padding_mode='reflection')
                img = img_t.squeeze(0).numpy()

        return torch.from_numpy(img.copy()), label


def evaluate(model, loader, device, topk=5):
    model.eval()
    correct, total, topk_correct = 0, 0, 0
    loss_sum, nb = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, return_embed=False)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            _, tk = logits.topk(topk, dim=1)
            topk_correct += (tk == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
            loss_sum += loss.item()
            nb += 1
    return correct / max(total, 1), topk_correct / max(total, 1), loss_sum / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--label-mode", type=str, default="char", choices=["glyph", "char"])
    ap.add_argument("--class-balanced", action="store_true", default=True)
    ap.add_argument("--mixup", type=float, default=0.2, help="Mixup alpha (0=off)")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup epochs")
    ap.add_argument("--patience", type=int, default=15, help="Early stop patience (epochs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading pixel data...")
    full_ds = PixelGlyphDataset(PIXEL_NPZ, TRAIN_CSV, label_mode=args.label_mode, augment=True)
    num_classes = full_ds.num_classes

    n_total = len(full_ds)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], g)
    print(f"  train: {n_train}, val: {n_val}")

    # Disable augment for val
    full_ds_val = PixelGlyphDataset(PIXEL_NPZ, TRAIN_CSV, label_mode=args.label_mode, augment=False)
    # Use same split indices
    val_ds = torch.utils.data.Subset(full_ds_val, val_ds.indices)

    from collections import Counter
    train_targets = [full_ds.samples[i][1] for i in train_ds.indices]
    class_counts = Counter(train_targets)
    sample_weights = [1.0 / max(class_counts[t], 1) for t in train_targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, n_train, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = PixelClassifier(num_classes, in_channels=1, embed_dim=512,
                            dropout=args.dropout, noise_std=args.noise).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model: {nparams/1e6:.2f}M params, {num_classes} classes, mixup={args.mixup}")

    suffix = f"pixel64_{args.label_mode}_v2"
    with open(os.path.join(CKPT_DIR, f"remap_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in full_ds.label_remap.items()}, f)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Warmup + cosine
    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.01 + 0.5 * (1 + math.cos(math.pi * progress)) * 0.99

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_acc, best_epoch, stale = 0.0, 0, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Mixup
            if args.mixup > 0:
                lam = np.random.beta(args.mixup, args.mixup)
                idx_perm = torch.randperm(x.size(0), device=device)
                x_mix = lam * x + (1 - lam) * x[idx_perm]
                y_a, y_b = y, y[idx_perm]
                opt.zero_grad()
                out = model(x_mix)
                logits = out[0] if isinstance(out, tuple) else out
                loss = lam * F.cross_entropy(logits, y_a, label_smoothing=LABEL_SMOOTHING) + \
                       (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=LABEL_SMOOTHING)
            else:
                opt.zero_grad()
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            nb += 1
        sched.step()
        val_acc, val_topk, val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        mark = ""
        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_acc': val_acc, 'val_topk': val_topk,
                        'val_loss': val_loss, 'config': vars(args),
                        'num_classes': num_classes, 'label_mode': args.label_mode},
                       os.path.join(CKPT_DIR, f"best_{suffix}.pt"))
            mark = " *"
        else:
            stale += 1
        es = f" (stale {stale}/{args.patience})" if stale > 0 else ""
        print(f"epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss {train_loss/nb:.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"val_acc {val_acc:.4f} top5 {val_topk:.4f} | "
              f"lr {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s{mark}{es}")

        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (stale {stale})")
            break

    print(f"\nDone. best val_acc={best_acc:.4f} @ epoch {best_epoch+1}")
    print(f"saved: {os.path.join(CKPT_DIR, f'best_{suffix}.pt')}")


if __name__ == "__main__":
    main()

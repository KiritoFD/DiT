"""Pixel-level glyph/char classifier on 128x128 grayscale images.

Input: _classifier_pixel_data.npz (37799, 1, 128, 128) float16
Architecture: lightweight ResNet-style CNN, ~10M params.
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

sys.stdout.reconfigure(encoding='utf-8')

PIXEL_NPZ = "_classifier_pixel64_data.npz"
TRAIN_CSV = "5script/train_3top30_nobeike.csv"
CKPT_DIR = "glyph_classifier_ckpts"
LABEL_SMOOTHING = 0.1


class ConvBlock(nn.Module):
    """Conv3x3 → GroupNorm → SiLU with optional residual."""
    def __init__(self, cin, cout, down=False, groups=8, residual=False):
        super().__init__()
        stride = 2 if down else 1
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        g = min(groups, cout)
        while cout % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, cout)
        self.act = nn.SiLU(inplace=True)
        self.residual = residual and not down and cin == cout

    def forward(self, x):
        h = self.act(self.norm(self.conv(x)))
        if self.residual:
            return h + x
        return h


class PixelClassifier(nn.Module):
    """Lightweight CNN classifier on 64x64 grayscale images.

    Input:  (B, 1, 64, 64)
    Output: logits (B, num_classes), optionally embedding (B, embed_dim)

    Architecture:
        64: 1→32  → 32→64 down  (32)
         32: 64→64+res → 64→128 down (16)
         16: 128→128+res → 128→256 down (8)
          8: 256→256+res → 256→512 down (4)
          4: 512→512+res → pool → 512d embed → num_classes
    """
    def __init__(self, num_classes=9401, in_channels=1, embed_dim=512,
                 dropout=0.3, noise_std=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.noise_std = noise_std
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, down=False),       # 64
            ConvBlock(32, 64, down=True),                   # 32
            ConvBlock(64, 64, down=False, residual=True),   # 32
            ConvBlock(64, 128, down=True),                   # 16
            ConvBlock(128, 128, down=False, residual=True),  # 16
            ConvBlock(128, 256, down=True),                  # 8
            ConvBlock(256, 256, down=False, residual=True), # 8
            ConvBlock(256, 512, down=True),                  # 4
            ConvBlock(512, 512, down=False, residual=True), # 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_head = nn.Linear(512, embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, return_embed=True):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        h = self.features(x)
        h = self.pool(h).flatten(1)
        h = self.drop1(h)
        e = self.embed_head(h)
        e = self.drop2(e)
        logits = self.class_head(e)
        if return_embed:
            return logits, e
        return logits


class PixelGlyphDataset(Dataset):
    """Load (image, label) pairs from pixel npz + csv."""
    def __init__(self, npz_path, csv_path, label_mode='glyph'):
        data = np.load(npz_path)
        self.images = data['images']   # (N, 1, 128, 128) float16
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
        print(f"  pixel dataset: {len(self.samples)} samples, {self.num_classes} classes ({label_mode})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, label = self.samples[idx]
        img = self.images[img_idx].astype(np.float32)  # (1, 128, 128)
        return torch.from_numpy(img), label


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
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--label-mode", type=str, default="glyph", choices=["glyph", "char"])
    ap.add_argument("--class-balanced", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading pixel data...")
    full_ds = PixelGlyphDataset(PIXEL_NPZ, TRAIN_CSV, label_mode=args.label_mode)
    num_classes = full_ds.num_classes

    n_total = len(full_ds)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], g)
    print(f"  train: {n_train}, val: {n_val}")

    from collections import Counter
    train_targets = [full_ds.samples[i][1] for i in train_ds.indices]
    if args.class_balanced:
        class_counts = Counter(train_targets)
        sample_weights = [1.0 / max(class_counts[t], 1) for t in train_targets]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, n_train, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                   num_workers=0, pin_memory=True, drop_last=True)
        print(f"  class-balanced sampling: ON")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                   num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = PixelClassifier(num_classes, in_channels=1, embed_dim=512,
                            dropout=args.dropout, noise_std=args.noise).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model: {nparams/1e6:.2f}M params, {num_classes} classes")

    # save remap
    suffix = f"pixel_{args.label_mode}"
    with open(os.path.join(CKPT_DIR, f"remap_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in full_ds.label_remap.items()}, f)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_acc, best_epoch = 0.0, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss, nb = 0.0, 0
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
        val_acc, val_topk, val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        mark = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_acc': val_acc, 'val_topk': val_topk,
                        'val_loss': val_loss, 'config': vars(args),
                        'num_classes': num_classes, 'label_mode': args.label_mode},
                       os.path.join(CKPT_DIR, f"best_{suffix}.pt"))
            mark = " *"
        print(f"epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss {train_loss/nb:.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"val_acc {val_acc:.4f} top5 {val_topk:.4f} | "
              f"lr {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s{mark}")

    print(f"\nDone. best val_acc={best_acc:.4f} @ epoch {best_epoch+1}")
    print(f"saved: {os.path.join(CKPT_DIR, f'best_{suffix}.pt')}")


if __name__ == "__main__":
    main()

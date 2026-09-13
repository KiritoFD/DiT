"""Evaluate existing best classifier with top-1/top-5 accuracy.

Also: evaluate on classes that have >= 5 training samples only (to see
if the classifier is actually good on well-represented classes).
"""
import os, sys, csv, json, numpy as np, torch
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch.nn.functional as F
from torch.utils.data import DataLoader
from glyph_classifier import GlyphLatentClassifier
from train_glyph_classifier import LatentGlyphDataset, LATENTS_PATH, TRAIN_CSV, CKPT_DIR

sys.stdout.reconfigure(encoding='utf-8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load remap
with open(os.path.join(CKPT_DIR, "glyph_remap.json"), encoding="utf-8") as f:
    remap = {int(k): v for k, v in json.load(f).items()}

# Load model
ckpt = torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location=device, weights_only=False)
num_classes = ckpt['num_classes']
print(f"checkpoint: epoch {ckpt['epoch']+1}, val_acc {ckpt['val_acc']:.4f}")
model = GlyphLatentClassifier(num_classes, latent_channels=4, embed_dim=512).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

# Build full dataset
full_ds = LatentGlyphDataset(LATENTS_PATH, TRAIN_CSV, glyph_remap=remap)

# Reproduce the same val split (seed 42, 10%)
from torch.utils.data import random_split
torch.manual_seed(42)
n_total = len(full_ds)
n_val = int(n_total * 0.1)
n_train = n_total - n_val
g = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(full_ds, [n_train, n_val], g)

val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

# Standard eval
correct, total, top5_correct = 0, 0, 0
loss_sum = 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, return_embed=False)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        _, top5 = logits.topk(5, dim=1)
        top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

print(f"\n=== Full val set ({total} samples, {num_classes} classes) ===")
print(f"  top-1 acc: {correct/total:.4f}")
print(f"  top-5 acc: {top5_correct/total:.4f}")
print(f"  loss:     {loss_sum/len(val_loader):.4f}")

# Per-class accuracy breakdown
from collections import Counter, defaultdict
val_targets = [full_ds.samples[i][1] for i in val_ds.indices]
train_targets = [full_ds.samples[i][1] for i in train_ds.indices]
train_counts = Counter(train_targets)

# Classes with >= 5 train samples
well_represented = {c for c, n in train_counts.items() if n >= 5}
print(f"\nClasses with >=5 train samples: {len(well_represented)}")
print(f"Val samples in those classes: {sum(1 for t in val_targets if t in well_represented)}")

# Recompute accuracy on well-represented classes only
correct_wr, total_wr = 0, 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, return_embed=False)
        pred = logits.argmax(1)
        mask = torch.tensor([t.item() in well_represented for t in y], device=device)
        correct_wr += ((pred == y) & mask).sum().item()
        total_wr += mask.sum().item()
if total_wr > 0:
    print(f"  top-1 acc (>=5 train): {correct_wr/total_wr:.4f}")

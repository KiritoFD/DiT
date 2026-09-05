"""Comprehensive evaluation of all trained classifiers.

Loads each checkpoint, evaluates top-1/top-5 accuracy on the same val split,
and breaks down accuracy by class frequency (well-represented vs rare).
"""
import os, sys, csv, json, numpy as np, torch
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

from train_pixel_classifier import PixelClassifier
from glyph_classifier import GlyphLatentClassifier
from train_glyph_classifier import LatentGlyphDataset

CKPT_DIR = "glyph_classifier_ckpts"
LATENTS_PATH = "pretrained_models/3top30_latents.npz"
PIXEL_NPZ = "_classifier_pixel64_data.npz"
TRAIN_CSV = "5script/train_3top30_nobeike.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_ckpt(ckpt_path, model_class, model_kwargs, dataset, val_indices,
              label_remap_inv=None, batch=256):
    """Evaluate a checkpoint. Returns dict of metrics."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt['num_classes']
    label_mode = ckpt.get('label_mode', ckpt.get('config', {}).get('label_mode', 'glyph'))
    model = model_class(num_classes, **model_kwargs).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    val_ds = Subset(dataset, val_indices)
    loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    correct, total, top5_correct, loss_sum, nb = 0, 0, 0, 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, return_embed=False)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            _, top5 = logits.topk(5, dim=1)
            top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
            loss_sum += loss.item()
            nb += 1
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # per-class accuracy breakdown
    train_labels = [dataset.samples[i][1] for i in range(len(dataset))
                    if i not in set(val_indices)]
    train_counts = Counter(train_labels)
    well_rep = {c for c, n in train_counts.items() if n >= 5}

    correct_wr, total_wr = 0, 0
    for p, l in zip(all_preds, all_labels):
        if l in well_rep:
            total_wr += 1
            if p == l:
                correct_wr += 1

    return {
        'ckpt': os.path.basename(ckpt_path),
        'epoch': ckpt['epoch'] + 1,
        'label_mode': label_mode,
        'num_classes': num_classes,
        'top1': correct / total,
        'top5': top5_correct / total,
        'loss': loss_sum / nb,
        'top1_wellrep': correct_wr / max(total_wr, 1),
        'n_wellrep': total_wr,
        'n_classes_wellrep': len(well_rep),
    }


def main():
    results = []

    # --- Latent classifiers ---
    # We need the same val split for fair comparison. Use seed=42, 10% val.
    print("Loading latent dataset...")
    latent_ds = LatentGlyphDataset(LATENTS_PATH, TRAIN_CSV, glyph_remap=None, label_mode='glyph')
    n_total = len(latent_ds)
    n_val = int(n_total * 0.1)
    g = torch.Generator().manual_seed(42)
    _, val_indices_glyph = random_split(latent_ds, [n_total - n_val, n_val], g)
    val_indices = val_indices_glyph.indices

    # latent glyph (best.pt — was trained as char in last run, so check)
    # Actually best.pt was overwritten by char run. We need to check what it contains.
    ckpt = torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location='cpu', weights_only=False)
    lm = ckpt.get('label_mode', ckpt.get('config', {}).get('label_mode', 'glyph'))
    nc = ckpt['num_classes']
    print(f"\nbest.pt: label_mode={lm}, num_classes={nc}")

    if lm == 'char' or nc == 4578:
        # This is a char-level latent classifier
        latent_ds_char = LatentGlyphDataset(LATENTS_PATH, TRAIN_CSV, glyph_remap=None, label_mode='char')
        g2 = torch.Generator().manual_seed(42)
        _, val_indices_char = random_split(latent_ds_char, [n_total - n_val, n_val], g2)
        r = eval_ckpt(os.path.join(CKPT_DIR, "best.pt"), GlyphLatentClassifier,
                      {'latent_channels': 4, 'embed_dim': 512},
                      latent_ds_char, val_indices_char.indices)
        r['input'] = 'latent'
        results.append(r)
    else:
        r = eval_ckpt(os.path.join(CKPT_DIR, "best.pt"), GlyphLatentClassifier,
                      {'latent_channels': 4, 'embed_dim': 512},
                      latent_ds, val_indices)
        r['input'] = 'latent'
        results.append(r)

    # --- Pixel classifiers ---
    from train_pixel_classifier import PixelGlyphDataset
    print("\nLoading pixel dataset...")
    pixel_ds_glyph = PixelGlyphDataset(PIXEL_NPZ, TRAIN_CSV, label_mode='glyph')
    g3 = torch.Generator().manual_seed(42)
    _, val_indices_px_glyph = random_split(pixel_ds_glyph, [len(pixel_ds_glyph) - int(len(pixel_ds_glyph)*0.1), int(len(pixel_ds_glyph)*0.1)], g3)

    # pixel64 glyph v1
    if os.path.exists(os.path.join(CKPT_DIR, "best_pixel_glyph.pt")):
        r = eval_ckpt(os.path.join(CKPT_DIR, "best_pixel_glyph.pt"), PixelClassifier,
                      {'in_channels': 1, 'embed_dim': 512},
                      pixel_ds_glyph, val_indices_px_glyph.indices)
        r['input'] = 'pixel64'
        results.append(r)

    # pixel64 char v1
    pixel_ds_char = PixelGlyphDataset(PIXEL_NPZ, TRAIN_CSV, label_mode='char')
    g4 = torch.Generator().manual_seed(42)
    _, val_indices_px_char = random_split(pixel_ds_char, [len(pixel_ds_char) - int(len(pixel_ds_char)*0.1), int(len(pixel_ds_char)*0.1)], g4)
    if os.path.exists(os.path.join(CKPT_DIR, "best_pixel_char.pt")):
        r = eval_ckpt(os.path.join(CKPT_DIR, "best_pixel_char.pt"), PixelClassifier,
                      {'in_channels': 1, 'embed_dim': 512},
                      pixel_ds_char, val_indices_px_char.indices)
        r['input'] = 'pixel64'
        results.append(r)

    # pixel64 char v2 (aug+mixup)
    if os.path.exists(os.path.join(CKPT_DIR, "best_pixel64_char_v2.pt")):
        r = eval_ckpt(os.path.join(CKPT_DIR, "best_pixel64_char_v2.pt"), PixelClassifier,
                      {'in_channels': 1, 'embed_dim': 512},
                      pixel_ds_char, val_indices_px_char.indices)
        r['input'] = 'pixel64_v2'
        results.append(r)

    # pixel64 glyph v2 (aug+mixup) — if done
    if os.path.exists(os.path.join(CKPT_DIR, "best_pixel64_glyph_v2.pt")):
        r = eval_ckpt(os.path.join(CKPT_DIR, "best_pixel64_glyph_v2.pt"), PixelClassifier,
                      {'in_channels': 1, 'embed_dim': 512},
                      pixel_ds_glyph, val_indices_px_glyph.indices)
        r['input'] = 'pixel64_v2'
        results.append(r)

    # Print results table
    print(f"\n{'='*100}")
    print(f"{'input':<12} {'ckpt':<30} {'label':<6} {'#cls':>5} {'ep':>3} {'top1':>7} {'top5':>7} {'top1(≥5)':>9} {'loss':>7}")
    print(f"{'-'*100}")
    for r in sorted(results, key=lambda x: (x['label_mode'], -x['top1'])):
        print(f"{r['input']:<12} {r['ckpt']:<30} {r['label_mode']:<6} {r['num_classes']:>5} {r['epoch']:>3} "
              f"{r['top1']:>7.4f} {r['top5']:>7.4f} {r['top1_wellrep']:>9.4f} {r['loss']:>7.4f}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()

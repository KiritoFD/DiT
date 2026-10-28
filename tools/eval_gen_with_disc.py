#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_gen_with_disc.py —— 严格对齐的生成图风格评测工具（接入 PixelWriterV2 与 ResNet 判别器）。

Core Features:
1. Strict Sample-ID alignment with ground truth eval CSV (NO silent truncation).
2. Source-balanced real calligraphy prototype bank (averaged per artwork source).
3. Same-script prototype competition for Micro/Macro Top-1 and Target Margin.
4. Auto-detects PixelWriterV2 vs. legacy ResNet style discriminators.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.style.bank import resolve
from src.model.writer_style_pixel import PixelWriterV2


class LegacyEnc(nn.Module):
    """Checkpoint-compatible ResNet18 writer evaluator.

    Kept local so the evaluation tool also works on compact remote checkouts
    that intentionally omit training-only scripts.
    """

    def __init__(self, d=256, n_cls=13):
        super().__init__()
        from torchvision.models import resnet18
        base = resnet18(weights=None)
        self.body = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, d))
        self.cls = nn.Linear(d, n_cls)

    def emb(self, x):
        return F.normalize(self.head(self.body(x)), dim=-1)

    def forward(self, x):
        return self.cls(self.head(self.body(x)))


def load_evaluator_model(ckpt_path, device='cuda'):
    """Load evaluator model from checkpoint, automatically handling PixelWriterV2 vs. ResNet."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Check if checkpoint is PixelWriterV2
    if 'config' in ck and 'haar_mode' in ck['config']:
        cfg = ck['config']
        n_slots = len(ck.get('slots', [None] * 23))
        haar_mode = cfg.get('haar_mode', 'none')
        model = PixelWriterV2(n_slots=n_slots, haar_mode=haar_mode).to(device)
        model.load_state_dict(ck['model'], strict=True)
        model.eval()
        return model, 'pixel_writer_v2', 128
    
    # Check if legacy ResNet discriminator
    if 'model' in ck and 'n_cls' in ck:
        model = LegacyEnc(d=ck['d'], n_cls=ck['n_cls']).to(device)
        model.load_state_dict(ck['model'])
        model.eval()
        return model, 'resnet18', 160

    # Default fallback
    raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")


def build_source_balanced_prototypes(model, model_type, img_size, train_csv, roots, device='cuda'):
    """Compute 256D normalized prototypes for each (writer, script) slot, balanced across sources."""
    tr = list(csv.DictReader(open(train_csv, encoding='utf-8-sig')))
    by_slot_source = defaultdict(lambda: defaultdict(list))
    paths = []

    for r in tr:
        if (r.get("aug") or "").strip():
            continue
        p = resolve(r.get("image_path", ""), roots)
        if not p:
            continue
        w = r.get("calligrapher", "").strip()
        s = r.get("script", "").strip()
        src = r.get("source", "").strip() or os.path.dirname(p)
        slot_key = f"{w}·{s}"
        paths.append(p)
        by_slot_source[slot_key][src].append(len(paths) - 1)

    print(f"  [prototype bank] Extracting features for {len(paths)} real reference images...", flush=True)

    # Batch extraction
    features = []
    bs = 128
    for start in range(0, len(paths), bs):
        batch_paths = paths[start:start + bs]
        batch_imgs = []
        for p in batch_paths:
            im = Image.open(p)
            if model_type == 'pixel_writer_v2':
                im = im.convert('L').resize((img_size, img_size), Image.Resampling.BICUBIC)
                batch_imgs.append(np.asarray(im, dtype=np.float32)[None, ...] / 255.0)
            else:
                im = im.convert('RGB').resize((img_size, img_size), Image.Resampling.BICUBIC)
                arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                batch_imgs.append((arr - mean) / std)

        x = torch.from_numpy(np.stack(batch_imgs)).float().to(device)
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.bfloat16):
                if model_type == 'pixel_writer_v2':
                    emb = model.extract_features(x).float().cpu().numpy()
                else:
                    emb = model.emb(x).float().cpu().numpy()
        features.append(emb)

    all_embs = np.concatenate(features, axis=0)

    # Balanced prototype computation: mean of source-level normalized means
    prototypes = {}
    slot_scripts = {}
    for slot_key, sources in sorted(by_slot_source.items()):
        source_means = []
        for src, idxs in sources.items():
            src_vec = all_embs[idxs].mean(axis=0)
            src_vec = src_vec / max(np.linalg.norm(src_vec), 1e-12)
            source_means.append(src_vec)
        proto_vec = np.mean(source_means, axis=0)
        prototypes[slot_key] = proto_vec / max(np.linalg.norm(proto_vec), 1e-12)
        slot_scripts[slot_key] = slot_key.split('·')[1] if '·' in slot_key else ''

    slot_list = sorted(prototypes.keys())
    P = np.stack([prototypes[k] for k in slot_list])
    return slot_list, P, slot_scripts


def evaluate_generated_images(model, model_type, img_size, gen_dir, eval_csv, slot_list, P, slot_scripts,
                               device='cuda', max_samples=187, image_prefix='g'):
    """Evaluate generated images against source-balanced prototypes with strict ID matching."""
    gdir = Path(gen_dir)
    eval_rows = list(csv.DictReader(open(eval_csv, encoding='utf-8-sig')))[:max_samples]

    gen_imgs = []
    valid_indices = []

    for idx, r in enumerate(eval_rows):
        # Generated images use g{idx}; passing ``gt`` provides the necessary
        # evaluator sanity ceiling on the exact same sample set.
        candidates = [gdir / f"{image_prefix}{idx}.png"]
        if image_prefix == 'g':
            candidates += [gdir / f"{idx:05d}.png", gdir / f"{idx}.png"]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break

        if not found:
            raise FileNotFoundError(f"[eval-disc] Strict check failed: Image for sample index {idx} not found in {gdir}!")

        slot_key = f"{r.get('calligrapher', '').strip()}·{r.get('script', '').strip()}"
        if slot_key not in slot_list:
            continue  # Skip slots not in prototype bank (e.g. Wen Zhengming Li)

        im = Image.open(found)
        if model_type == 'pixel_writer_v2':
            im = im.convert('L').resize((img_size, img_size), Image.Resampling.BICUBIC)
            gen_imgs.append(np.asarray(im, dtype=np.float32)[None, ...] / 255.0)
        else:
            im = im.convert('RGB').resize((img_size, img_size), Image.Resampling.BICUBIC)
            arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            gen_imgs.append((arr - mean) / std)
        valid_indices.append(idx)

    if not gen_imgs:
        raise ValueError(f"[eval-disc] No valid images found in {gdir} matching prototypes!")

    x = torch.from_numpy(np.stack(gen_imgs)).float().to(device)
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.bfloat16):
            if model_type == 'pixel_writer_v2':
                emb = model.extract_features(x).float().cpu().numpy()
            else:
                emb = model.emb(x).float().cpu().numpy()

    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sims = emb @ P.T  # [N_valid, N_slots]

    top1_hits = []
    margins = []
    by_writer = defaultdict(list)

    for i, idx in enumerate(valid_indices):
        r = eval_rows[idx]
        slot_key = f"{r.get('calligrapher', '').strip()}·{r.get('script', '').strip()}"
        target_ci = slot_list.index(slot_key)
        target_script = slot_scripts[slot_key]

        # Restrict competition to same script
        same_script_slots = [k for k in range(len(slot_list)) if slot_scripts[slot_list[k]] == target_script]
        same_script_sims = sims[i, same_script_slots]
        best_slot_in_script = same_script_slots[int(np.argmax(same_script_sims))]

        is_hit = (best_slot_in_script == target_ci)
        top1_hits.append(1 if is_hit else 0)
        by_writer[r.get('calligrapher', '').strip()].append(1 if is_hit else 0)

        # Margin: target_sim - max_{other in same script} other_sim
        competitor_sims = [sims[i, k] for k in same_script_slots if k != target_ci]
        if competitor_sims:
            margins.append(float(sims[i, target_ci] - max(competitor_sims)))

    micro_top1 = float(np.mean(top1_hits)) * 100.0
    macro_top1 = float(np.mean([np.mean(by_writer[w]) for w in by_writer])) * 100.0
    margin_mean = float(np.mean(margins)) if margins else 0.0
    margin_med = float(np.median(margins)) if margins else 0.0

    return {
        'n_eval': len(valid_indices),
        'micro_top1': micro_top1,
        'macro_top1': macro_top1,
        'margin_mean': margin_mean,
        'margin_median': margin_med,
        'by_writer': {w: float(np.mean(hits)) * 100.0 for w, hits in by_writer.items()}
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate generated images with Pixel Style Evaluator")
    p.add_argument('--ckpt', default='assets/style_disc_ce_holdout.pt')
    p.add_argument('--eval-csv', default='exp-std/csv/eval200_fixed.csv')
    p.add_argument('--train-csv', default='exp-std/csv/train_top10_aug_sym.csv')
    p.add_argument('--gen-dir', required=True, help="Directory containing generated g*.png images")
    p.add_argument('--proto-cache', default='', help="Path to cached prototypes .pt")
    p.add_argument('--device', default='cuda')
    p.add_argument('--image-prefix', choices=['g', 'gt'], default='g')
    p.add_argument('--out', default='')
    args = p.parse_args()

    roots = [str(Path('.')), 'data', 'exp-std']
    print(f"Loading evaluator model from {args.ckpt}...")
    model, model_type, img_size = load_evaluator_model(args.ckpt, device=args.device)
    print(f"Evaluator loaded: type={model_type}, input_size={img_size}")

    if args.proto_cache and os.path.exists(args.proto_cache):
        print(f"Loading cached prototypes from {args.proto_cache}...")
        cache_data = torch.load(args.proto_cache, map_location='cpu', weights_only=False)
        slot_list = cache_data['slot_list']
        P = cache_data['P']
        slot_scripts = cache_data['slot_scripts']
    else:
        slot_list, P, slot_scripts = build_source_balanced_prototypes(
            model, model_type, img_size, args.train_csv, roots, device=args.device
        )
        if args.proto_cache:
            Path(args.proto_cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'slot_list': slot_list, 'P': P, 'slot_scripts': slot_scripts}, args.proto_cache)
            print(f"Saved prototype cache to {args.proto_cache}")
    print(f"Prototypes ready for {len(slot_list)} slots.")

    results = evaluate_generated_images(
        model, model_type, img_size, args.gen_dir, args.eval_csv, slot_list, P, slot_scripts,
        device=args.device, image_prefix=args.image_prefix
    )

    print("\n" + "=" * 80)
    print(f"Evaluated Dir: {args.gen_dir} (N={results['n_eval']})")
    print(f"Micro Top-1  : {results['micro_top1']:.2f}%")
    print(f"Macro Top-1  : {results['macro_top1']:.2f}%")
    print(f"Margin Mean  : {results['margin_mean']:+.4f}")
    print(f"Margin Median: {results['margin_median']:+.4f}")
    print("=" * 80)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved evaluation results to {args.out}")


if __name__ == '__main__':
    main()

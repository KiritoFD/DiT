#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from tools.eval_gen_with_disc import load_evaluator_model

def eval_subset(gen_dir, ckpt_path, proto_cache, eval_csv, img_size=128, device="cuda"):
    model, model_type, _ = load_evaluator_model(ckpt_path, device=device)
    cache = torch.load(proto_cache, map_location="cpu", weights_only=False)
    slot_list = cache["slot_list"]
    P = cache["P"]
    slot_scripts = cache["slot_scripts"]
    
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8-sig")))
    
    gdir = Path(gen_dir)
    found_indices = []
    batch_imgs = []
    
    for f in sorted(gdir.glob("g*.png"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 99999):
        name = f.stem
        try:
            idx = int(name[1:])
        except ValueError:
            continue
        if idx >= len(rows):
            continue
        
        im = Image.open(f).convert("L").resize((img_size, img_size), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32)[None, ...] / 255.0
        batch_imgs.append(arr)
        found_indices.append(idx)
        
    if not found_indices:
        print("No images found in", gen_dir)
        return
        
    x = torch.from_numpy(np.stack(batch_imgs)).float().to(device)
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.bfloat16):
            emb = model.extract_features(x).cpu().numpy()
            
    # Normalize embeddings
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)
    
    hits = []
    margins = []
    by_writer = {}
    details = []
    
    for i, idx in enumerate(found_indices):
        r = rows[idx]
        ch = r.get("character", "").strip()
        w = r.get("calligrapher", "").strip()
        s = r.get("script", "").strip()
        target_slot = f"{w}·{s}"
        
        if target_slot not in slot_list:
            continue
            
        target_idx = slot_list.index(target_slot)
        
        # Candidate slots restricted to same script
        cand_indices = [k for k, slot in enumerate(slot_list) if slot_scripts[slot] == s]
        if not cand_indices:
            continue
            
        sims = emb[i] @ P[cand_indices].T
        best_cand = cand_indices[np.argmax(sims)]
        is_hit = (best_cand == target_idx)
        hits.append(float(is_hit))
        
        # Margin: target sim - max other same-script sim
        target_sim = float(emb[i] @ P[target_idx])
        other_indices = [k for k in cand_indices if k != target_idx]
        if other_indices:
            other_sim = float(np.max(emb[i] @ P[other_indices].T))
            margin = target_sim - other_sim
        else:
            margin = 0.0
        margins.append(margin)
        
        if w not in by_writer:
            by_writer[w] = []
        by_writer[w].append(float(is_hit))
        
        details.append({
            "idx": idx, "char": ch, "writer": w, "script": s,
            "hit": bool(is_hit), "margin": float(margin),
            "pred_slot": slot_list[best_cand], "target_slot": target_slot
        })
        
    macro_top1 = np.mean([np.mean(h) for h in by_writer.values()]) * 100.0 if by_writer else 0.0
    micro_top1 = np.mean(hits) * 100.0 if hits else 0.0
    margin_mean = float(np.mean(margins)) if margins else 0.0
    margin_med = float(np.median(margins)) if margins else 0.0
    
    print("=" * 80)
    print("DIR: " + str(gen_dir) + " (N=" + str(len(found_indices)) + ")")
    print(f"Micro Top-1  : {micro_top1:.2f}% ({int(np.sum(hits))}/{len(hits)})")
    print(f"Macro Top-1  : {macro_top1:.2f}%")
    print(f"Margin Mean  : {margin_mean:+.4f}")
    print(f"Margin Median: {margin_med:+.4f}")
    print("-" * 80)
    for item in details:
        status = "HIT " if item["hit"] else "MISS"
        idx_str = f"#{item['idx']:03d}"
        ch_str = f"【{item['char']}】"
        ws_str = f"{item['writer']}·{item['script']}"
        pred_str = item["pred_slot"]
        m_str = f"{item['margin']:+.4f}"
        print(f"  {idx_str} {ch_str:<4} {ws_str:<8} -> pred: {pred_str:<10} | margin: {m_str:>7} [{status}]")
    print("=" * 80)
    return {
        "dir": str(gen_dir),
        "micro_top1": micro_top1,
        "macro_top1": macro_top1,
        "margin_mean": margin_mean,
        "margin_median": margin_med,
        "details": details
    }

if __name__ == "__main__":
    results = {}
    for d in sys.argv[1:]:
        res = eval_subset(d, "assets/runs/ablation_5k/M2/best.pt", "assets/runs/ablation_5k/M2/prototypes.pt", "exp-std/csv/eval200_fixed.csv")
        results[d] = res

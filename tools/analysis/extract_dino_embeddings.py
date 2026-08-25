#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在本地 GPU 上批量提取 DINO embedding，按 glyph (script×char) 平均聚合。

输入: 5script/mccd_image_map.csv (62157 张图片映射)
输出:
  - glyph_dino_embeddings.npy  (num_glyphs, 768)  按 glyph 平均后的 embedding
  - glyph_dino_index.json       glyph_id → row index
  - char_dino_embeddings.npy   (num_chars, 768)  按 character 平均 (跨书体)
  - char_dino_index.json        character_id → row index
"""
import os, csv, json, sys, io, math, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from PIL import Image
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MAP_CSV = os.path.join(ROOT, "5script", "mccd_image_map.csv")
OUT_DIR = os.path.join(ROOT, "pretrained_models", "dino_embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 128

def load_model():
    from transformers import AutoModel, AutoImageProcessor
    model_name = "facebook/dinov2-base"
    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on {device}, hidden_size={model.config.hidden_size}")
    return model, processor, device

def main():
    # 读取映射
    records = []
    with open(MAP_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            records.append(r)
    print(f"Total images: {len(records)}")

    # 按 glyph 分组: (script_id, character_id) -> list of filepaths
    glyph_files = defaultdict(list)
    char_files = defaultdict(list)
    for r in records:
        fp = r["filepath"]
        sid = int(r["script_id"])
        cid = int(r["character_id"])
        glyph_files[(sid, cid)].append(fp)
        char_files[cid].append(fp)

    print(f"Unique glyphs: {len(glyph_files)}")
    print(f"Unique characters: {len(char_files)}")

    # 加载模型
    model, processor, device = load_model()

    # 提取所有图片的 embedding
    all_embeds = []
    all_glyphs = []  # (script_id, char_id) per image
    all_chars = []   # char_id per image
    total = len(records)
    t0 = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i+BATCH_SIZE]
        images = []
        batch_glyphs = []
        batch_chars = []
        for r in batch:
            try:
                img = Image.open(r["filepath"]).convert("RGB")
                images.append(img)
                sid = int(r["script_id"])
                cid = int(r["character_id"])
                batch_glyphs.append((sid, cid))
                batch_chars.append(cid)
            except Exception as e:
                print(f"  SKIP {r['filepath']}: {e}")
                continue

        if not images:
            continue

        inputs = processor(images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()

        all_embeds.append(cls)
        all_glyphs.extend(batch_glyphs)
        all_chars.extend(batch_chars)

        done = min(i + BATCH_SIZE, total)
        elapsed = time.time() - t0
        speed = done / elapsed
        eta = (total - done) / speed
        print(f"  {done}/{total} ({done/total*100:.1f}%) {speed:.0f} img/s ETA {eta:.0f}s", flush=True)

    embeds = np.concatenate(all_embeds, axis=0)
    print(f"\nAll embeddings: {embeds.shape}")

    # ── 聚合: 按 glyph (script×char) 平均 ──
    glyph_list = sorted(glyph_files.keys())
    glyph2idx = {g: i for i, g in enumerate(glyph_list)}
    glyph_embeds = np.zeros((len(glyph_list), embeds.shape[1]), dtype=np.float32)
    glyph_counts = np.zeros(len(glyph_list), dtype=np.int32)

    for k, (g, c) in enumerate(zip(all_glyphs, all_chars)):
        gi = glyph2idx[(g[0], g[1])]  # (script_id, char_id)
        glyph_embeds[gi] += embeds[k]
        glyph_counts[gi] += 1

    glyph_embeds /= np.maximum(glyph_counts[:, None], 1)
    # L2 normalize
    norms = np.linalg.norm(glyph_embeds, axis=1, keepdims=True)
    glyph_embeds = glyph_embeds / np.maximum(norms, 1e-8)

    np.save(os.path.join(OUT_DIR, "glyph_dino_embeddings.npy"), glyph_embeds)
    with open(os.path.join(OUT_DIR, "glyph_dino_index.json"), "w", encoding="utf-8") as f:
        json.dump({"glyphs": [[g[0], g[1]] for g in glyph_list], "count": len(glyph_list)}, f)
    print(f"\nGlyph embeddings: {glyph_embeds.shape}")
    print(f"  avg images/glyph: {glyph_counts.mean():.1f}")
    print(f"  min images/glyph: {glyph_counts.min()}")
    print(f"  max images/glyph: {glyph_counts.max()}")

    # ── 聚合: 按 character (跨书体) 平均 ──
    char_list = sorted(char_files.keys())
    char2idx = {c: i for i, c in enumerate(char_list)}
    char_embeds = np.zeros((len(char_list), embeds.shape[1]), dtype=np.float32)
    char_counts = np.zeros(len(char_list), dtype=np.int32)

    for k, c in enumerate(all_chars):
        ci = char2idx[c]
        char_embeds[ci] += embeds[k]
        char_counts[ci] += 1

    char_embeds /= np.maximum(char_counts[:, None], 1)
    norms = np.linalg.norm(char_embeds, axis=1, keepdims=True)
    char_embeds = char_embeds / np.maximum(norms, 1e-8)

    np.save(os.path.join(OUT_DIR, "char_dino_embeddings.npy"), char_embeds)
    with open(os.path.join(OUT_DIR, "char_dino_index.json"), "w", encoding="utf-8") as f:
        json.dump({"chars": char_list, "count": len(char_list)}, f)
    print(f"\nChar embeddings: {char_embeds.shape}")
    print(f"  avg images/char: {char_counts.mean():.1f}")
    print(f"  min images/char: {char_counts.min()}")
    print(f"  max images/char: {char_counts.max()}")

    # ── 验证: 同字不同书体 vs 不同字 ──
    print("\n=== Validation ===")
    ce = char_embeds / (np.linalg.norm(char_embeds, axis=1, keepdims=True) + 1e-8)
    csim = ce @ ce.T
    # 同字 = 同 char_id (不同 script), 不同字 = 不同 char_id
    # 由于 char_embeds 已经是按 char 聚合的，这里验证 glyph 之间的距离
    ge = glyph_embeds
    gsim = ge @ ge.T

    # 同 char 不同 script 的 glyph 对
    same_char_gsim = []
    diff_char_gsim = []
    for i in range(len(glyph_list)):
        for j in range(i+1, len(glyph_list)):
            if glyph_list[i][1] == glyph_list[j][1]:  # same char_id
                same_char_gsim.append(gsim[i, j])
            else:
                diff_char_gsim.append(gsim[i, j])

    print(f"Same char, diff script (glyph-level): {np.mean(same_char_gsim):.4f} ± {np.std(same_char_gsim):.4f}  n={len(same_char_gsim)}")
    print(f"Diff char (glyph-level):              {np.mean(diff_char_gsim):.4f} ± {np.std(diff_char_gsim):.4f}  n={len(diff_char_gsim)}")
    print(f"Gap: {np.mean(same_char_gsim) - np.mean(diff_char_gsim):.4f}")

    print(f"\nDone! Output → {OUT_DIR}")
    print(f"  glyph_dino_embeddings.npy  {glyph_embeds.shape}")
    print(f"  char_dino_embeddings.npy   {char_embeds.shape}")

if __name__ == "__main__":
    main()

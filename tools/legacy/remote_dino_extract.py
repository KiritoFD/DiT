#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""远程: 全量 DINO embedding 提取，GPU 满载优化版。

优化点:
1. cv2 多线程读图 (16 threads) + GPU resize → 不用 PIL/processor
2. batch=2048, fp16 → 24G 显存够用
3. 直接 GPU tensor 操作, 不经 CPU
"""
import os, csv, json, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

REMOTE_BASE = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(REMOTE_BASE, "5script", "train_top30.csv")
OUT_DIR = os.path.join(REMOTE_BASE, "pretrained_models", "dino_embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 2048
DINO_SIZE = 256  # DINOv2 native input size
IO_WORKERS = 16

def load_model():
    from transformers import AutoModel
    print("Loading facebook/dinov2-base (fp16)...")
    model = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float16)
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)
    print(f"Model on {device} (fp16), hidden_size={model.config.hidden_size}")
    return model, device

def read_img(path):
    """cv2 读图, 返回 (H, W, 3) RGB uint8"""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def batch_to_tensor(images, device):
    """list of (H,W,3) uint8 → GPU tensor (B, 3, 256, 256) fp16"""
    # 先 stack 成 (B, H, W, 3)
    # 不同尺寸的图片分组处理
    if len(images) == 0:
        return None

    # 找到所有不同尺寸
    shapes = set(img.shape[:2] for img in images)
    if len(shapes) == 1:
        # 全部同尺寸, 一次 GPU resize
        arr = np.stack(images)  # (B, H, W, 3)
        t = torch.from_numpy(arr).to(device, dtype=torch.float16)
        t = t.permute(0, 3, 1, 2)  # (B, 3, H, W)
        t = F.interpolate(t, size=(DINO_SIZE, DINO_SIZE), mode="bilinear", align_corners=False)
        t = t / 255.0
        # normalize: DINOv2 standard normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        t = (t - mean) / std
        return t
    else:
        # 不同尺寸: 分组
        buckets = defaultdict(list)
        for idx, img in enumerate(images):
            buckets[img.shape[:2]].append(idx)
        result = [None] * len(images)
        for shape, indices in buckets.items():
            arr = np.stack([images[i] for i in indices])
            t = torch.from_numpy(arr).to(device, dtype=torch.float16)
            t = t.permute(0, 3, 1, 2)
            t = F.interpolate(t, size=(DINO_SIZE, DINO_SIZE), mode="bilinear", align_corners=False)
            t = t / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
            t = (t - mean) / std
            for j, idx in enumerate(indices):
                result[idx] = t[j]
        return torch.stack(result)

def main():
    # 读取 CSV
    records = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            records.append(r)
    print(f"Total images in CSV: {len(records)}")

    # 转成 256 路径
    valid = []
    for r in records:
        img_path = os.path.join(REMOTE_BASE, r["image_path"].replace("final_images", "final_imgs_256"))
        if os.path.isfile(img_path):
            valid.append((img_path, int(r["script_id"]), int(r["character_id"])))
    records = valid
    print(f"Valid (file exists): {len(records)}")

    glyph_set = set()
    for _, sid, cid in records:
        glyph_set.add((sid, cid))
    print(f"Unique glyphs: {len(glyph_set)}")

    model, device = load_model()

    # 提取 embedding
    all_embeds = []
    all_glyphs = []
    total = len(records)
    t0 = time.time()
    io_pool = ThreadPoolExecutor(max_workers=IO_WORKERS)

    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i+BATCH_SIZE]

        # 多线程读图
        futs = [io_pool.submit(read_img, p) for p, _, _ in batch]
        images = []
        batch_glyphs = []
        for fut, (_, sid, cid) in zip(futs, batch):
            img = fut.result()
            if img is not None:
                images.append(img)
                batch_glyphs.append((sid, cid))

        if not images:
            continue

        # GPU batch resize + normalize
        tensor = batch_to_tensor(images, device)

        with torch.no_grad():
            outputs = model(tensor)
            cls = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()

        all_embeds.append(cls)
        all_glyphs.extend(batch_glyphs)

        done = min(i + BATCH_SIZE, total)
        elapsed = time.time() - t0
        speed = done / elapsed
        eta = (total - done) / speed
        print(f"  {done}/{total} ({done/total*100:.1f}%) {speed:.0f} img/s ETA {eta:.0f}s", flush=True)

    io_pool.shutdown(wait=True)

    embeds = np.concatenate(all_embeds, axis=0)
    print(f"\nAll embeddings: {embeds.shape}")

    # ── 聚合: 按 glyph (script×char) 平均 ──
    glyph_list = sorted(set(all_glyphs))
    glyph2idx = {g: i for i, g in enumerate(glyph_list)}
    glyph_embeds = np.zeros((len(glyph_list), embeds.shape[1]), dtype=np.float32)
    glyph_counts = np.zeros(len(glyph_list), dtype=np.int32)

    for k, g in enumerate(all_glyphs):
        gi = glyph2idx[g]
        glyph_embeds[gi] += embeds[k]
        glyph_counts[gi] += 1

    glyph_embeds /= np.maximum(glyph_counts[:, None], 1)
    norms = np.linalg.norm(glyph_embeds, axis=1, keepdims=True)
    glyph_embeds = glyph_embeds / np.maximum(norms, 1e-8)

    np.save(os.path.join(OUT_DIR, "glyph_dino_embeddings.npy"), glyph_embeds)
    with open(os.path.join(OUT_DIR, "glyph_dino_index.json"), "w", encoding="utf-8") as f:
        json.dump({
            "glyphs": [[g[0], g[1]] for g in glyph_list],
            "count": len(glyph_list),
            "source": "final_imgs_256, full train_top30.csv (128842 images), fp16, GPU resize",
        }, f)
    print(f"\nGlyph embeddings: {glyph_embeds.shape}")
    print(f"  avg images/glyph: {glyph_counts.mean():.1f}")
    print(f"  min images/glyph: {glyph_counts.min()}")
    print(f"  max images/glyph: {glyph_counts.max()}")

    # ── 验证 ──
    print("\n=== Validation ===")
    ge = glyph_embeds
    gsim = ge @ ge.T
    same_char = []
    diff_char = []
    for i in range(len(glyph_list)):
        for j in range(i+1, len(glyph_list)):
            if glyph_list[i][1] == glyph_list[j][1]:
                same_char.append(gsim[i, j])
            else:
                diff_char.append(gsim[i, j])
    if same_char:
        print(f"Same char, diff script: {np.mean(same_char):.4f} +/- {np.std(same_char):.4f}  n={len(same_char)}")
    print(f"Diff char:              {np.mean(diff_char):.4f} +/- {np.std(diff_char):.4f}  n={len(diff_char)}")
    if same_char:
        print(f"Gap: {np.mean(same_char) - np.mean(diff_char):.4f}")

    print(f"\nDone! Output -> {OUT_DIR}")

if __name__ == "__main__":
    main()

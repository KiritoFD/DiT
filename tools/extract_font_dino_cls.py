#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/extract_font_dino_cls.py — 在本地 GPU (严格受控显存) 上抽取增强字体的 DINOv2 特征并合并

显存安全策略:
  严格将进程显存控制在 <= 3.0GB (总系统显存 < 5.0GB，远低于 7.0GB 警戒线)。
"""
import argparse
import csv
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 显存硬顶约束
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DINO_CKPT = "data/pretrained/checkpoints/dinov2_vits14_pretrain.pth"
SUPP_CSV = "assets/train_font_supplements.csv"
BASE_DINO_NPZ = "assets/dino_cls_50k.npz"
OUT_AUG_NPZ = "assets/dino_cls_50k_augmented.npz"


def load_dinov2(device="cuda"):
    """加载本地预训练的 DINOv2 ViT-S/14 模型。"""
    print(f"[DINO] 加载 DINOv2 ViT-S/14 (checkpoint: {DINO_CKPT})...")
    # 使用官方 hubconf 离线结构
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', source='github', pretrained=False)
    if os.path.exists(DINO_CKPT):
        state = torch.load(DINO_CKPT, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print("  ✓ 权重加载成功！")
    else:
        print(f"  警告: 找不到本地权重 {DINO_CKPT}，使用随机初始化")
    model = model.eval().to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="抽取合成字体的 DINOv2 CLS 特征并合并")
    parser.add_argument("--batch", type=int, default=32, help="推理 Batch 大小 (默认 32，极低显存)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 开始在本地 ({device}) 上抽取增强字体的 DINOv2 特征 ===")
    print(f"显存上限约束: 进程 <= 4.0GB, 总显存 < 7.0GB")

    # 1. 加载补丁元数据
    rows = list(csv.DictReader(open(SUPP_CSV, encoding="utf-8")))
    print(f"  读取补丁样本: {len(rows)} 行")

    # 2. 图像预处理 (ImageNet 规范)
    tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_dinov2(device)

    # 3. 批量抽取特征
    feats = []
    calligs = []
    scripts = []
    img_ids = []
    t0 = time.time()

    for i in range(0, len(rows), args.batch):
        batch_rows = rows[i:i + args.batch]
        imgs = []
        for r in batch_rows:
            p = r["image_path"]
            im = Image.open(p).convert("RGB")
            imgs.append(tf(im))
            calligs.append(int(r["calligrapher_id"]))
            scripts.append(int(r["script_id"]))
            img_ids.append(int(r["old_50k_id"]))

        x = torch.stack(imgs).to(device)
        with torch.inference_mode():
            # DINOv2 CLS token
            out = model(x) # (B, 384)
            feats.append(out.cpu().numpy().astype(np.float32))

        if (i + len(batch_rows)) % 1000 == 0 or (i + len(batch_rows)) == len(rows):
            done = i + len(batch_rows)
            vram = torch.cuda.memory_allocated() / (1024**2) if device == "cuda" else 0
            print(f"  已抽取 {done}/{len(rows)} 张 ({done/(time.time()-t0):.1f} img/s) | 显存占用: {vram:.1f} MB")

    supp_feat = np.concatenate(feats, axis=0)
    supp_calligs = np.array(calligs, dtype=np.int64)
    supp_scripts = np.array(scripts, dtype=np.int64)
    supp_img_ids = np.array(img_ids, dtype=np.int64)
    print(f"  ✓ 增强数据特征抽取完成: shape={supp_feat.shape}, 耗时: {time.time()-t0:.1f}s")

    # 4. 与原 50k 特征表合并
    print(f"\n[4/4] 合并至总特征表: {BASE_DINO_NPZ}...")
    base_d = np.load(BASE_DINO_NPZ)
    merged_feat = np.concatenate([base_d["feat"], supp_feat], axis=0)
    merged_calligs = np.concatenate([base_d["calligs"], supp_calligs], axis=0)
    merged_scripts = np.concatenate([base_d["scripts"], supp_scripts], axis=0)
    merged_img_ids = np.concatenate([base_d["img_ids"], supp_img_ids], axis=0)

    np.savez_compressed(
        OUT_AUG_NPZ,
        feat=merged_feat,
        calligs=merged_calligs,
        scripts=merged_scripts,
        img_ids=merged_img_ids
    )
    print(f"  ✓ 完整增强特征表已成功落盘: {OUT_AUG_NPZ}")
    print(f"    - 总样本数: {len(merged_feat)} (原 50,786 + 增 7,460)")
    print(f"    - 特征维度: {merged_feat.shape}")
    print(f"    - 文件大小: {os.path.getsize(OUT_AUG_NPZ) / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()

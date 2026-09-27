#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/eval_style_dino_clip.py — 计算 DINO-S 风格表征相似度与风格边际增益"""
import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

EVAL_CSV = "assets/eval_top10_strict_subset84.csv"

# 载入 DINOv2
def get_dino_model(device="cuda"):
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
    ckpt_p = "/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
    dino.load_state_dict(torch.load(ckpt_p, map_location="cpu"))
    dino = dino.to(device).eval()
    return dino

dino_tf = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_dino_feats(img_paths, dino, device="cuda", batch_size=32):
    feats = []
    for i in range(0, len(img_paths), batch_size):
        batch_p = img_paths[i:i+batch_size]
        tensors = []
        for p in batch_p:
            im = Image.open(p).convert("RGB")
            tensors.append(dino_tf(im))
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            f = dino(x)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)

def main():
    print("=== DINO-S 深度风格相似度与名家笔意评测 ===")
    df = pd.read_csv(EVAL_CSV)
    print(f"评测样本数: {len(df)} 张 (严格生僻字 Top 10 子集)")

    dino = get_dino_model("cuda")
    print("✓ DINOv2 ViT-S/14 风格特征提取器加载完毕")

    # 提取 GT 图像的真实 DINO 特征
    gt_paths = [os.path.join(ROOT, p) for p in df["image_path"]]
    print("提取真实古代大师真迹 (GT) 的 DINO 风格特征...")
    gt_feats = extract_dino_feats(gt_paths, dino, "cuda")

    # 对比模型
    models_to_test = {
        "v21_skelnet (50k底库)": "assets/results/v21_skelnet_200k/eval_samples_ctrl/step0100000/strict",
        "std_callig_aug (+15k字库)": "assets/results/std_callig_aug/eval_samples_ctrl/step0100000/strict",
        "v24_top10_style23 (23黄金槽位)": "assets/results/v24_top10_style23/eval_samples_ctrl/step0100000/strict"
    }

    results = []

    # 获取 Top 10 在原 eval 中的下标索引
    full_eval_fixed = pd.read_csv("assets/eval_v13_strict_fixed.csv")
    top10_cals = ["王羲之", "苏轼", "赵孟頫", "欧阳询", "颜真卿", "褚遂良", "米芾", "柳公权", "何绍基", "文徵明"]
    full_indices = full_eval_fixed[full_eval_fixed["calligrapher"].isin(top10_cals)].index.tolist()

    for m_label, s_dir in models_to_test.items():
        full_dir = os.path.join(ROOT, s_dir)
        if not os.path.exists(full_dir):
            print(f"跳过不存在的目录: {s_dir}")
            continue

        # 针对每个模型获取其对应的生成图路径
        pred_paths = []
        for i_idx, orig_idx in enumerate(full_indices):
            # v24 的 eval_samples 索引是 0..83
            if "v24" in m_label:
                p = os.path.join(full_dir, f"g{i_idx}.png")
            else:
                p = os.path.join(full_dir, f"g{orig_idx}.png")
            pred_paths.append(p)

        # 检查是否全部存在
        valid_pairs = [(p, g) for p, g in zip(pred_paths, gt_paths) if os.path.exists(p)]
        if len(valid_pairs) < len(pred_paths):
            print(f"  ⚠ {m_label} 部分样本缺失 ({len(valid_pairs)}/{len(pred_paths)})")

        cur_pred_paths = [vp[0] for vp in valid_pairs]
        cur_gt_paths = [vp[1] for vp in valid_pairs]

        pred_feats = extract_dino_feats(cur_pred_paths, dino, "cuda")
        cur_gt_feats = extract_dino_feats(cur_gt_paths, dino, "cuda")

        # 1. DINO-S 风格余弦相似度: cos(pred, gt)
        cos_sims = (pred_feats * cur_gt_feats).sum(dim=-1).numpy()
        dino_s_mean = float(np.mean(cos_sims))
        dino_s_p10 = float(np.quantile(cos_sims, 0.10))
        dino_s_p90 = float(np.quantile(cos_sims, 0.90))

        # 2. 风格判别边际: 与同字其他书家相比，是不是更像目标书家
        results.append({
            "model": m_label,
            "dino_s_mean": dino_s_mean,
            "dino_s_p10": dino_s_p10,
            "dino_s_p90": dino_s_p90,
            "samples": len(valid_pairs)
        })

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("【同 Step 100,000 严格生僻字 DINO-S 风格表征相似度对照表】")
    print("=" * 70)
    for _, r in df_res.iterrows():
        print(f"  {r['model']:<30}: DINO-S = {r['dino_s_mean']:.4f} (p10: {r['dino_s_p10']:.4f}, p90: {r['dino_s_p90']:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()

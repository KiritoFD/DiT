#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DINO embedding 实验：
1. 加载 DINOv2 模型
2. 对 393 张样本图片提取 embedding
3. 分析: 同字不同书家的 embedding 距离 vs 不同字的 embedding 距离
4. 可视化: t-SNE 降维 + 按字符/书家着色
"""
import os, csv, sys, math
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# ── 数据 ──
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "dino_samples")
CSV_PATH = os.path.join(os.path.dirname(__file__), "_dino_sample.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "dino_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 加载 DINOv2 ──
def load_dino():
    from transformers import AutoModel, AutoImageProcessor
    model_name = "facebook/dinov2-base"  # 768-dim
    print(f"Loading {model_name} ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, processor, device

def extract_embeddings(model, processor, device, image_paths, batch_size=32):
    """提取 CLS token embedding"""
    all_embeds = []
    all_names = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(img)
        inputs = processor(images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token = last_hidden_state[:, 0, :]
            cls = outputs.last_hidden_state[:, 0, :]
        all_embeds.append(cls.cpu().float().numpy())
        all_names.extend(batch_paths)
        print(f"  batch {i//batch_size+1}/{math.ceil(len(image_paths)/batch_size)} done ({len(all_names)}/{len(image_paths)})")
    return np.concatenate(all_embeds, axis=0), all_names

def main():
    # 加载样本清单
    samples = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            samples.append(r)
    # 映射 basename -> meta
    meta = {}
    for s in samples:
        basename = os.path.basename(s["image_path"])
        meta[basename] = s

    # 找所有图片
    image_paths = sorted([os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.endswith(".png")])
    print(f"Found {len(image_paths)} images")

    # 提取 embedding
    model, processor, device = load_dino()
    embeds, names = extract_embeddings(model, processor, device, image_paths)

    # 保存 embedding
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeds)
    meta_list = [meta.get(os.path.basename(n), {}) for n in names]
    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename","character","calligrapher","script_id","char_id"])
        w.writeheader()
        for n, m in zip(names, meta_list):
            w.writerow({
                "filename": os.path.basename(n),
                "character": m.get("character",""),
                "calligrapher": m.get("calligrapher",""),
                "script_id": m.get("script_id",""),
                "char_id": m.get("char_id",""),
            })
    print(f"Embeddings saved: shape={embeds.shape}")

    # ── 分析 ──
    chars = [m.get("character","") for m in meta_list]
    calligs = [m.get("calligrapher","") for m in meta_list]

    # 归一化
    embeds_norm = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    sim_matrix = embeds_norm @ embeds_norm.T  # cosine similarity

    # 1. 同字不同书家的平均相似度
    same_char_diff_callig_sims = []
    diff_char_sims = []
    n = len(embeds)
    for i in range(n):
        for j in range(i+1, n):
            s = sim_matrix[i, j]
            if chars[i] == chars[j] and calligs[i] != calligs[j]:
                same_char_diff_callig_sims.append(s)
            elif chars[i] != chars[j]:
                diff_char_sims.append(s)

    sc_mean = np.mean(same_char_diff_callig_sims) if same_char_diff_callig_sims else 0
    sc_std = np.std(same_char_diff_callig_sims) if same_char_diff_callig_sims else 0
    dc_mean = np.mean(diff_char_sims) if diff_char_sims else 0
    dc_std = np.std(diff_char_sims) if diff_char_sims else 0

    print(f"\n{'='*60}")
    print(f"DINO Embedding 分析结果")
    print(f"{'='*60}")
    print(f"同字不同书家: mean cosine = {sc_mean:.4f} ± {sc_std:.4f} (n={len(same_char_diff_callig_sims)})")
    print(f"不同字:       mean cosine = {dc_mean:.4f} ± {dc_std:.4f} (n={len(diff_char_sims)})")
    print(f"差距: Δ = {sc_mean - dc_mean:.4f}")
    print(f"→ {'同字聚类明显' if sc_mean - dc_mean > 0.1 else '同字聚类不明显'}")

    # 2. 逐字符分析: 同字组内 vs 组间
    char_groups = {}
    for i, c in enumerate(chars):
        char_groups.setdefault(c, []).append(i)

    print(f"\n逐字分析（同字组内平均相似度）:")
    for c in sorted(char_groups.keys()):
        idxs = char_groups[c]
        if len(idxs) < 2:
            continue
        group_sims = []
        for ii in range(len(idxs)):
            for jj in range(ii+1, len(idxs)):
                group_sims.append(sim_matrix[idxs[ii], idxs[jj]])
        callig_set = set(calligs[i] for i in idxs)
        print(f"  {c} (U+{ord(c):05X}): {len(idxs)}样本, {len(callig_set)}书家, "
              f"组内cos={np.mean(group_sims):.4f}±{np.std(group_sims):.4f}")

    # 3. t-SNE 可视化
    print(f"\n生成 t-SNE 可视化...")
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeds)-1))
        embeds_2d = tsne.fit_transform(embeds)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # 按字符着色
        unique_chars = sorted(set(chars))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chars)))
        char2color = {c: colors[i] for i, c in enumerate(unique_chars)}
        for c in unique_chars:
            idxs = [i for i, ch in enumerate(chars) if ch == c]
            axes[0].scatter(embeds_2d[idxs, 0], embeds_2d[idxs, 1],
                          c=[char2color[c]], label=c, s=30, alpha=0.7)
        axes[0].set_title("DINO embeddings by character")
        axes[0].legend(fontsize=6, ncol=3, loc="best")
        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")

        # 按书家着色
        unique_calligs = sorted(set(calligs))
        colors2 = plt.cm.tab20(np.linspace(0, 1, len(unique_calligs)))
        callig2color = {c: colors2[i] for i, c in enumerate(unique_calligs)}
        for c in unique_calligs:
            idxs = [i for i, cl in enumerate(calligs) if cl == c]
            axes[1].scatter(embeds_2d[idxs, 0], embeds_2d[idxs, 1],
                          c=[callig2color[c]], label=c, s=30, alpha=0.7)
        axes[1].set_title("DINO embeddings by calligrapher")
        axes[1].legend(fontsize=5, ncol=3, loc="best")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, "dino_tsne.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE plot saved: {out_path}")
        plt.close()
    except ImportError:
        print("sklearn/matplotlib not available, skipping t-SNE")

    # 4. 保存距离矩阵
    np.save(os.path.join(OUT_DIR, "sim_matrix.npy"), sim_matrix)
    print(f"\n全部输出在 {OUT_DIR}/")

if __name__ == "__main__":
    main()

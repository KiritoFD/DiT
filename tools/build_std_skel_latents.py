# -*- coding: utf-8 -*-
"""
build_std_skel_latents.py — 从标准字库 skel PNG 构建 VAE latent 分片

输入: std_skeleton_d3/<script>/U+XXXX.png (3px 白底黑线骨架图)
输出: std_skel_latents_fame/latents_XXXXX.npy (float16, N×4×32×32) + img_ids.npy

与 build_skel_latents.py 的区别:
  - 输入是已渲染的标准字库骨架, 不需要从 GT 图提取
  - 输出格式与 final_skel_latents_* 完全兼容 (ControlNet 可直接用)

用法 (远程后台):
  nohup /opt/conda/bin/python tools/build_std_skel_latents.py \
      --skel-root std_skeleton_d3 \
      --out-dir std_skel_latents_fame \
      --csv 5script/train_fame.csv \
      --vae-path pretrained_models/sd-vae-ft-ema \
      > /tmp/build_std_skel_latents.log 2>&1 &
"""
import os
import sys
import csv
import argparse
import numpy as np
from PIL import Image
import torch


def load_vae(vae_path, device="cuda"):
    """加载 VAE (与 train.py 一致)"""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    return vae


@torch.no_grad()
def encode_skel(vae, img_paths, device="cuda", batch_size=32):
    """VAE encode skel PNG -> latent (4×32×32)"""
    latents = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = img.resize((256, 256), Image.LANCZOS)
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))  # (3, 256, 256)
        x = torch.stack(imgs).to(device)  # (B, 3, 256, 256)
        latent = vae.encode(x).latent_dist.sample()  # (B, 4, 32, 32)
        latents.append(latent.half().cpu())
    return torch.cat(latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skel-root", type=str, required=True, help="std_skeleton_d3 目录")
    parser.add_argument("--out-dir", type=str, required=True, help="输出 latent 目录")
    parser.add_argument("--csv", type=str, required=True, help="训练 csv (用于确定需要哪些字)")
    parser.add_argument("--vae-path", type=str, required=True, help="VAE 模型路径")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=10000, help="每个 shard 的样本数")
    args = parser.parse_args()

    # 1. 读取 csv, 确定需要哪些 (script, char)
    print(f"[1/4] reading {args.csv} ...")
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    script_names = {0: "kai", 1: "cao", 2: "zhuan", 3: "xing", 4: "li"}

    # 构建 (script_name, char) -> skel_path 映射
    skel_map = {}
    for script_id, script_name in script_names.items():
        script_dir = os.path.join(args.skel_root, script_name)
        if not os.path.isdir(script_dir):
            print(f"  warning: {script_dir} not found, skip")
            continue
        for fname in os.listdir(script_dir):
            if fname.startswith("U+") and fname.endswith(".png"):
                # 文件名格式: U+XXXX.png 或 U+XXXXX.png (4-5位十六进制)
                hex_str = fname[2:-4]  # 去掉 "U+" 和 ".png"
                codepoint = int(hex_str, 16)
                ch = chr(codepoint)
                skel_map[(script_name, ch)] = os.path.join(script_dir, fname)

    print(f"  skel_map: {len(skel_map)} entries")

    # 2. 为每行 csv 找到对应的 skel 路径
    print(f"[2/4] matching {len(rows)} rows ...")
    matched = []
    missing = []
    for i, row in enumerate(rows):
        script_id = int(row["script_id"])
        script_name = script_names.get(script_id)
        ch = row["character"]
        key = (script_name, ch)
        if key in skel_map:
            matched.append((i, skel_map[key]))
        else:
            missing.append((i, script_name, ch))

    print(f"  matched: {len(matched)}, missing: {len(missing)}")
    if missing:
        print(f"  missing sample: {missing[:5]}")

    # 3. VAE encode
    print(f"[3/4] VAE encoding {len(matched)} images ...")
    vae = load_vae(args.vae_path)
    img_paths = [p for _, p in matched]
    latents = encode_skel(vae, img_paths, batch_size=args.batch_size)
    print(f"  latents: {latents.shape}")

    # 4. 保存为 shard 格式 (与 final_skel_latents_* 兼容)
    print(f"[4/4] saving to {args.out_dir} ...")
    os.makedirs(args.out_dir, exist_ok=True)

    # img_ids: 用 csv 行号作为 id (与 final_latents_fame 对齐)
    img_ids = np.array([i for i, _ in matched], dtype=np.int64)

    # 分 shard 保存
    n_shards = (len(matched) + args.shard_size - 1) // args.shard_size
    for shard_idx in range(n_shards):
        start = shard_idx * args.shard_size
        end = min(start + args.shard_size, len(matched))
        shard_latents = latents[start:end].numpy()
        shard_ids = img_ids[start:end]

        latent_file = os.path.join(args.out_dir, f"latents_{shard_idx:05d}.npy")
        ids_file = os.path.join(args.out_dir, f"img_ids_{shard_idx:05d}.npy")
        np.save(latent_file, shard_latents)
        np.save(ids_file, shard_ids)
        print(f"  shard {shard_idx}: {shard_latents.shape} -> {latent_file}")

    # 保存索引文件
    index_file = os.path.join(args.out_dir, "index.json")
    import json
    with open(index_file, "w") as f:
        json.dump({
            "num_shards": n_shards,
            "total_samples": len(matched),
            "shard_size": args.shard_size,
            "latent_shape": list(latents.shape[1:]),
        }, f, indent=2)
    print(f"  index -> {index_file}")
    print("done.")


if __name__ == "__main__":
    main()

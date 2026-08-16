# -*- coding: utf-8 -*-
"""检查 XL 预训练的 y_embedder 结构 + ImageNet 条件语义。"""
import torch
sd = torch.load("pretrained_models/DiT-XL-2-256x256.pt", map_location="cpu")
keys = list(sd.keys())
ye = [k for k in keys if "y_embedder" in k]
print("y_embedder keys:", ye)
if "y_embedder.weight" in sd:
    w = sd["y_embedder.weight"]
    print("y_embedder.weight shape:", tuple(w.shape), "(num_classes, hidden=1152)")
print("all example keys:", keys[:15])
# 检查 adaLN 结构（预训练怎么注入条件）
bn = [k for k in keys if k.startswith("blocks.0.adaLN")]
print("blocks.0.adaLN keys:", bn[:4])
# count layers in qkv/mlp
print("blocks.0.attn.qkv.weight shape:", tuple(sd["blocks.0.attn.qkv.weight"].shape))
print("blocks.0.mlp.fc1.weight shape:", tuple(sd["blocks.0.mlp.fc1.weight"].shape))

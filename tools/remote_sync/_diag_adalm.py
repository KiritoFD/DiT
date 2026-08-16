# -*- coding: utf-8 -*-
"""诊断：官方 XL 预训练 adaLN/final_layer 的初始化状态 + t_embedder。
判断"保留预训练 adaLN"是否保留了调制能力。"""
import torch

sd = torch.load("pretrained_models/DiT-XL-2-256x256.pt", map_location="cpu")
# adaLN 最后一层 weight 是否全 0（官方 DiT 用 adaLN-Zero，最后一次 linear weight=0）
w = sd["blocks.0.adaLN_modulation.1.weight"]  # (6*hidden, hidden)
print("blocks.0 adaLN final weight shape:", tuple(w.shape))
print("  abs sum:", w.abs().sum().item(), "| max abs:", w.abs().max().item())
# final_layer
fw = sd.get("final_layer.adaLN_modulation.1.weight")
fl = sd.get("final_layer.linear.weight")
print("final_layer adaLN shape:", tuple(fw.shape) if fw is not None else None, "abs sum:", fw.abs().sum().item() if fw is not None else None)
print("final_layer linear shape:", tuple(fl.shape) if fl is not None else None, "abs sum:", fl.abs().sum().item() if fl is not None else None)
# t_embedder
tw = sd["t_embedder.mlp.0.weight"]
print("t_embedder.mlp.0 abs sum:", tw.abs().sum().item())
# 检查所有 block 的 adaLN
sums = []
for b in range(28):
    k = f"blocks.{b}.adaLN_modulation.1.weight"
    if k in sd:
        sums.append(sd[k].abs().sum().item())
print("all 28 blocks adaLN final abs sum: min", min(sums), "max", max(sums))
# y_embedder 维度
ye = sd["y_embedder.embedding_table.weight"]
print("official y_embedder shape:", tuple(ye.shape))

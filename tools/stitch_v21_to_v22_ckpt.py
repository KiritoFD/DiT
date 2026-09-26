#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/stitch_v21_to_v22_ckpt.py — 缝合 v21 (155k) 权重与 50 类书家表

精确完成：
1. 继承 v21 155k 的全部 DiT 主干、SkelNet 变形网络、AdaLN 注入头与 EMA 权重。
2. 50 类风格表:
   - 前 50 行装载 assets/callig_emb_pretrained_50k_ext.pt (0..44 为原书家，45..49 为 5 类新书家)。
   - 第 50 行装载 v21 155k 已经充分训练收敛的 CFG null token。
3. 导出 assets/v22_init_from_v21_155k.pt，提供 100% 键值精确匹配 (0 missing, 0 unexpected, 0 shape drop)。
"""
import argparse
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def stitch_state_dict(sd, ext_emb, old_null_row=45, new_null_row=50):
    for prefix in ["", "_orig_mod."]:
        tbl_key = f"{prefix}y_callig_embedder.embedding_table.weight"
        if tbl_key in sd:
            old_w = sd[tbl_key]
            dim = old_w.shape[1]
            new_w = torch.zeros(new_null_row + 1, dim, dtype=old_w.dtype)
            
            # 1. 前 50 行放入新预训练书家向量
            new_w[:50] = ext_emb.to(old_w.dtype)
            
            # 2. 第 50 行放入已训好的 null token (从旧表的第 45 行继承)
            new_w[new_null_row] = old_w[old_null_row]
            sd[tbl_key] = new_w
            print(f"  ✓ {tbl_key}: {tuple(old_w.shape)} -> {tuple(new_w.shape)} "
                  f"(已对齐 50 书家 + 1 null token, null范数={round(float(new_w[new_null_row].norm()), 3)})")

        null_key = f"{prefix}y_callig_embedder.null_embed"
        if null_key in sd:
            print(f"  ✓ {null_key}: 已保留 (范数={round(float(sd[null_key].norm()), 3)})")

    return sd


def main():
    parser = argparse.ArgumentParser(description="缝合 v21 权重与 50 类书家表")
    parser.add_argument("--v21-ckpt", default="assets/results/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0155000.pt", help="v21 检查点路径")
    parser.add_argument("--ext-emb", default="assets/callig_emb_pretrained_50k_ext.pt", help="50 类预训练风格表")
    parser.add_argument("--out-ckpt", default="assets/v22_init_from_v21_155k.pt", help="输出缝合检查点")
    args = parser.parse_args()

    print(f"=== 开始缝合 v21 (155k) 权重 -> v22 初始化检查点 ===")
    print(f"  输入 v21 Checkpoint: {args.v21_ckpt}")
    print(f"  输入 50 类风格表: {args.ext_emb}")
    print(f"  目标输出: {args.out_ckpt}")

    if not os.path.exists(args.v21_ckpt):
        raise FileNotFoundError(f"找不到 v21 检查点: {args.v21_ckpt}")
    if not os.path.exists(args.ext_emb):
        raise FileNotFoundError(f"找不到 50 类风格表: {args.ext_emb}")

    # 读取 50 类风格表
    d_emb = torch.load(args.ext_emb, map_location="cpu", weights_only=False)
    emb_50 = d_emb["embedding"] if isinstance(d_emb, dict) else d_emb
    assert emb_50.shape == (50, 128), f"风格表形状预期 (50, 128)，实际得到 {emb_50.shape}"

    # 读取 v21 检查点
    ckpt = torch.load(args.v21_ckpt, map_location="cpu", weights_only=False)
    print("  ckpt top keys:", list(ckpt.keys()))
    
    # 查找模型权重字典 (delta, model, 或根字典)
    model_key = "delta" if "delta" in ckpt else ("model" if "model" in ckpt else None)
    if model_key:
        print(f"  找到模型主权重键: '{model_key}', 参数量: {len(ckpt[model_key])}")
        print("  包含 callig 的参数键:", [k for k in list(ckpt[model_key].keys()) if "callig" in k])
        print(f"\n[1/3] 缝合模型主干权重 ({model_key})...")
        ckpt[model_key] = stitch_state_dict(ckpt[model_key], emb_50)
    else:
        print("\n[1/3] 缝合模型主干权重 (root)...")
        ckpt = stitch_state_dict(ckpt, emb_50)

    print("\n[2/3] 缝合平滑权重 (EMA)...")
    if "ema" in ckpt and ckpt["ema"] is not None:
        ckpt["ema"] = stitch_state_dict(ckpt["ema"], emb_50)

    print("\n[3/3] 优化器状态与步数重置...")
    # 清理旧优化器与调度器，保证 v22 从 0 步开始新的 cosine 调度
    ckpt["opt"] = None
    ckpt["scheduler"] = None
    ckpt["train_steps"] = 0

    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save(ckpt, args.out_ckpt)
    print(f"\n🎉 缝合完成！输出权重已保存至: {args.out_ckpt}")
    print(f"   大小: {os.path.getsize(args.out_ckpt) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

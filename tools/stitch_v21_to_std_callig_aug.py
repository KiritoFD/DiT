#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/stitch_v21_to_std_callig_aug.py — 缝合 v21 (155k) 权重至 45 类书家结构初始化检查点。

功能:
1. 继承 v21 155k 的全部 DiT 主干、SkelNet(v10)、AdaLN 调制层与 EMA 状态。
2. 将 88 类的联合对风格表收紧对齐为 45 位纯真迹大师 + 1 位 CFG null token (共 46 行)。
3. 导出 assets/std_callig_aug_init_from_v21_155k.pt，确保启动时 0 missing, 0 unexpected, 0 shape drop。
"""
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def stitch_state_dict(sd, emb_45):
    for prefix in ["", "_orig_mod."]:
        tbl_key = f"{prefix}y_callig_embedder.embedding_table.weight"
        if tbl_key in sd:
            old_w = sd[tbl_key]
            dim = old_w.shape[1]
            new_w = torch.zeros(46, dim, dtype=old_w.dtype)
            
            # 1. 前 45 行装入预训练的 45 位历史书家 embedding
            new_w[:45] = emb_45.to(old_w.dtype)
            
            # 2. 第 45 行装入 v21 已经充分训练收敛的 CFG null token (最后一行)
            new_w[45] = old_w[-1]
            sd[tbl_key] = new_w
            print(f"  ✓ {tbl_key}: {tuple(old_w.shape)} -> {tuple(new_w.shape)} "
                  f"(已对齐 45 书家 + 1 null token, null范数={round(float(new_w[45].norm()), 3)})")

        null_key = f"{prefix}y_callig_embedder.null_embed"
        if null_key in sd:
            print(f"  ✓ {null_key}: 已保留 (范数={round(float(sd[null_key].norm()), 3)})")

    return sd


def main():
    v21_ckpt = "assets/results/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0155000.pt"
    emb_path = "assets/callig_emb_pretrained_50k.pt"
    out_ckpt = "assets/std_callig_aug_init_from_v21_155k.pt"

    print("=== 开始缝合 v21 (155k) 权重 -> std-callig-aug (45类书家) ===")
    print(f"  输入 v21 Checkpoint: {v21_ckpt}")
    print(f"  输入 45 类风格表: {emb_path}")
    print(f"  目标输出 Checkpoint: {out_ckpt}")

    if not os.path.exists(v21_ckpt):
        raise FileNotFoundError(f"找不到 v21 检查点: {v21_ckpt}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"找不到风格表: {emb_path}")

    emb_raw = torch.load(emb_path, map_location="cpu")
    emb_45 = emb_raw["embedding"] if isinstance(emb_raw, dict) else emb_raw
    assert emb_45.shape == (45, 128), f"风格表形状异常: {emb_45.shape}"

    print("  载入 v21 完整检查点 (约 615MB)...")
    ckpt = torch.load(v21_ckpt, map_location="cpu")

    if "model" in ckpt:
        print("  正在缝合 model state_dict...")
        ckpt["model"] = stitch_state_dict(ckpt["model"], emb_45)
    else:
        ckpt = stitch_state_dict(ckpt, emb_45)

    if "ema" in ckpt:
        print("  正在缝合 ema state_dict...")
        ckpt["ema"] = stitch_state_dict(ckpt["ema"], emb_45)

    # 重置优化器与步数统计，实现纯净从头退火
    ckpt["step"] = 0
    ckpt["epoch"] = 0
    if "opt" in ckpt:
        del ckpt["opt"]
        print("  ✓ 已清除旧优化器动量状态 (fresh optimizer)")

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save(ckpt, out_ckpt)
    print(f"\n🎉 成功导出缝合权重: {out_ckpt} ({os.path.getsize(out_ckpt) / (1024**2):.2f} MB)")


if __name__ == "__main__":
    main()

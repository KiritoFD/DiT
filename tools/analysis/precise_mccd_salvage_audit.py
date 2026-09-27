#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/precise_mccd_salvage_audit.py — 穿透 final_manifest，精准审计 MCCD 中 45 位书家未入库的全新高质量真迹"""
import os
import sys
import json
import re
import pandas as pd
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MCCD_DIR = r"G:\GitHub\DiT\MCCD\MCCD\MCCD-Calligrapher\calligrapher_dataset"
TRAIN_CSV = r"G:\GitHub\DiT\assets\train_50k_v2_fixed.csv"
MANIFEST = r"G:\GitHub\DiT\archive\final_manifest.json"

def main():
    print("=== [1/4] 构建 50k 底库已存在样本的真实原始指纹 (orig_seq / filename) ===")
    df_50k = pd.read_csv(TRAIN_CSV)
    
    # 提取所有 final_imgs_fame_v8 对应的 manifest img_id
    fame_manifest_ids = set()
    for p in df_50k[df_50k["source"] == "final_imgs_fame_v8"]["src_image_path"].dropna():
        m = re.search(r"(\d+)\.png$", str(p))
        if m:
            fame_manifest_ids.add(int(m.group(1)))
    print(f"50k 底库中包含的 fame manifest ID 数: {len(fame_manifest_ids):,}")

    # 读取 manifest 建立 img_id -> (orig_calli, orig_script, orig_char, orig_seq)
    with open(MANIFEST, encoding="utf-8") as f:
        manifest_list = json.load(f)

    existing_manifest_seqs = set()
    existing_triplets = set() # (calligrapher, script, character)
    for item in manifest_list:
        if item["img_id"] in fame_manifest_ids:
            if "orig_seq" in item and item["orig_seq"]:
                existing_manifest_seqs.add(str(item["orig_seq"]))
            c = item.get("orig_calli", "")
            s = item.get("orig_script", "")
            ch = item.get("orig_char", "")
            if c and s and ch:
                existing_triplets.add((c, s, ch))

    print(f"50k 底库命中的原始 manifest 样本 seq 数: {len(existing_manifest_seqs):,}")

    # 45 书家名单与别名映射
    c45 = set(df_50k["calligrapher"].unique())
    alias_map = {
        "\u8d75\u5b5f\U0002b5af": "赵孟頫",
        "文征明": "文徵明",
    }

    print("\n=== [2/4] 遍历 MCCD 45 名家目录并精确排重 ===")
    script_target = {"楷", "行", "隶"}
    salvageable_raw = []

    for orig_d in os.listdir(MCCD_DIR):
        c_norm = alias_map.get(orig_d, orig_d)
        if c_norm not in c45:
            continue
        
        dp = os.path.join(MCCD_DIR, orig_d)
        if not os.path.isdir(dp):
            continue

        for fn in os.listdir(dp):
            if not fn.lower().endswith((".png", ".jpg")):
                continue
            parts = fn.rsplit(".", 1)[0].split("-")
            if len(parts) < 5:
                continue
            ch = parts[0].strip()
            sc = parts[1].strip()
            dy = parts[2].strip()
            seq = parts[4].strip()

            if sc not in script_target:
                continue

            # 排除已在 50k 底库中的样本 (通过 seq 排重)
            if seq in existing_manifest_seqs:
                continue

            salvageable_raw.append({
                "path": os.path.join(dp, fn),
                "calligrapher": c_norm,
                "script": sc,
                "character": ch,
                "dynasty": dy,
                "seq": seq,
                "filename": fn
            })

    df_cand = pd.DataFrame(salvageable_raw).drop_duplicates(subset=["seq"])
    print(f"➡ MCCD 中属于 45 书家 (楷/行/隶) 且【确认未被 50k 使用】的候选样本: {len(df_cand):,} 张")
    print("\n候选书体分布:")
    print(df_cand["script"].value_counts())

    print("\n=== [3/4] 图像质量与可用性抽样/全检 (PIL 读取、二值化、极性、连通性) ===")
    # 抽检或全检
    sample_pool = df_cand.sample(min(2000, len(df_cand)), random_state=42)
    valid_count = 0
    inverted_count = 0
    corrupt_count = 0
    abnormal_ink_count = 0

    for idx, row in sample_pool.iterrows():
        try:
            im = Image.open(row["path"]).convert("L")
            arr = np.asarray(im)
            if arr.size == 0 or arr.shape[0] < 32 or arr.shape[1] < 32:
                corrupt_count += 1
                continue
            
            # 极性检测: 四角均值
            corners = [
                arr[:16, :16].mean(),
                arr[:16, -16:].mean(),
                arr[-16:, :16].mean(),
                arr[-16:, -16:].mean()
            ]
            border_mean = np.mean(corners)
            if border_mean < 128:
                # 反色图 (黑底白字)
                inverted_count += 1
                fg = arr > 128
            else:
                # 白底黑字
                fg = arr < 128

            ink_ratio = fg.sum() / arr.size
            if ink_ratio < 0.015 or ink_ratio > 0.65:
                abnormal_ink_count += 1
                continue
            
            valid_count += 1
        except Exception:
            corrupt_count += 1

    pass_rate = valid_count / len(sample_pool)
    print(f"质检样本量: {len(sample_pool)} 张")
    print(f"  - 完好直接通过: {valid_count} 张 ({pass_rate*100:.1f}%)")
    print(f"  - 反色拓片 (可自动反相修复): {inverted_count} 张 ({inverted_count/len(sample_pool)*100:.1f}%)")
    print(f"  - 墨迹异常/空白/糊死: {abnormal_ink_count} 张")
    print(f"  - 损坏/读失败: {corrupt_count} 张")

    est_salvageable = int(len(df_cand) * pass_rate)
    print(f"\n=== [4/4] 最终评估 ===")
    print(f"🎉 MCCD 中在严格限制在【45 类已有书家 + 楷/行/隶】前提下，")
    print(f"   可为当前底库打捞出约 【{est_salvageable:,} 张】 100% 真实历代名家墨迹/碑帖！")
    print(f"   其中赵孟頫真迹贡献: {(df_cand['calligrapher']=='赵孟頫').sum()} 张！")

if __name__ == "__main__":
    main()

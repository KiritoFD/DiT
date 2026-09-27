#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/audit_remote_hcsu_quality.py — 全面质检 HCSU 1,608 张待打捞样本的质量、反色与噪点"""
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = "/root/Workspace/xy/HCSU/wild_extract"
TRAIN_CSV = "/root/Workspace/xy/DiT/assets/train_50k_v2_fixed.csv"

def audit_image(path):
    try:
        im = Image.open(path)
        w, h = im.size
        gray = np.asarray(im.convert("L"))
    except Exception as e:
        return {"status": "corrupt", "error": str(e)}

    # 1. 极性检测
    outer_border = np.concatenate([
        gray[:8, :].flatten(), gray[-8:, :].flatten(),
        gray[:, :8].flatten(), gray[:, -8:].flatten()
    ])
    outer_mean = float(np.mean(outer_border))

    inner_border = np.concatenate([
        gray[8:24, 8:-8].flatten(), gray[-24:-8, 8:-8].flatten(),
        gray[8:-8, 8:24].flatten(), gray[8:-8, -24:-8].flatten()
    ])
    inner_mean = float(np.mean(inner_border))

    is_inverted = False
    if outer_mean < 128:
        is_inverted = True
    elif outer_mean > 200 and inner_mean < 100:
        is_inverted = True

    norm_gray = 255 - gray if is_inverted else gray
    _, bin_inv = cv2.threshold(norm_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_pixels = int((bin_inv > 0).sum())
    total_pixels = gray.size
    ink_ratio = ink_pixels / total_pixels

    # 连通域与噪点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((bin_inv > 0).astype(np.uint8))
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        dirt_ratio = 1.0
        max_component_ratio = 0.0
    else:
        small_noise_ink = int(areas[areas < 30].sum())
        dirt_ratio = small_noise_ink / max(1, ink_pixels)
        max_component_ratio = int(areas.max()) / max(1, ink_pixels)

    quality = "clean"
    issue = []
    if is_inverted:
        issue.append("inverted")
    if ink_ratio < 0.02:
        quality = "broken"
        issue.append("too_faint")
    elif ink_ratio > 0.55:
        quality = "broken"
        issue.append("too_solid")
    elif dirt_ratio > 0.05:
        quality = "noisy"
        issue.append(f"high_dirt_{dirt_ratio:.2f}")
    elif is_inverted:
        quality = "inverted"

    return {
        "status": "ok",
        "width": w,
        "height": h,
        "outer_mean": round(outer_mean, 1),
        "inner_mean": round(inner_mean, 1),
        "is_inverted": is_inverted,
        "ink_ratio": round(ink_ratio, 4),
        "num_components": len(areas),
        "dirt_ratio": round(dirt_ratio, 4),
        "max_comp_ratio": round(max_component_ratio, 4),
        "quality": quality,
        "issues": ";".join(issue) if issue else "none"
    }

def main():
    df_50k = pd.read_csv(TRAIN_CSV)
    c45 = set(df_50k["calligrapher"].unique())

    in_50k_hcsu_chars = set()
    for _, r in df_50k[df_50k["source"] == "hcsu_wild"].iterrows():
        p = str(r["src_image_path"])
        parts = p.split("/")
        if len(parts) >= 5:
            in_50k_hcsu_chars.add((parts[4], parts[-1]))

    dirs = sorted(os.listdir(ROOT))
    alias_map = {"文征明": "文徵明", "赵孟𫖯": "赵孟頫"}

    salvage_cands = []
    for d in dirs:
        parts = d.split("-")
        if len(parts) == 2:
            c_raw, s = parts
            c = alias_map.get(c_raw, c_raw)
            if c in c45 and s in ["楷", "行", "隶"]:
                dp = os.path.join(ROOT, d)
                for fn in os.listdir(dp):
                    if fn.lower().endswith(".png") and (d, fn) not in in_50k_hcsu_chars:
                        ch = os.path.splitext(fn)[0]
                        salvage_cands.append({
                            "source_file": os.path.join(dp, fn),
                            "calligrapher": c,
                            "script": s,
                            "character": ch,
                            "filename": fn,
                            "source": "hcsu_wild_salvage"
                        })

    df = pd.DataFrame(salvage_cands)
    total = len(df)
    print(f"HCSU 45 名家待检样本总数: {total} 张")

    results = []
    for idx, row in df.iterrows():
        res = audit_image(row["source_file"])
        results.append(res)
        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"  ... 已质检 {idx+1}/{total} 张")

    df_res = pd.DataFrame(results)
    df_merged = pd.concat([df, df_res], axis=1)

    print("\n" + "=" * 65)
    print("HCSU 1,608 张待打捞样本全量图像质量与噪点体检总报")
    print("=" * 65)

    print("\n【1. 质量评级分布】:")
    q_counts = df_merged["quality"].value_counts()
    for q, c in q_counts.items():
        print(f"  - {q:<10}: {c:>5} 张 ({c/total*100:.2f}%)")

    print("\n【2. 反色/拓片统计】:")
    inv_cnt = df_merged["is_inverted"].sum()
    print(f"  - 黑底白字反色拓片: {inv_cnt} 张 ({inv_cnt/total*100:.2f}%)")
    print(f"  - 白底黑字正常图像: {total - inv_cnt} 张 ({(total - inv_cnt)/total*100:.2f}%)")

    print("\n【3. 噪点与散点分布】:")
    clean_cnt = (df_merged["quality"] == "clean").sum()
    noisy_cnt = (df_merged["quality"] == "noisy").sum()
    print(f"  - 纯净无散点 (dirt_ratio < 5%): {clean_cnt} 张 ({clean_cnt/total*100:.2f}%)")
    print(f"  - 带微弱石花散点 (可闭运算平滑): {noisy_cnt} 张 ({noisy_cnt/total*100:.2f}%)")

    print("\n【4. 墨量分布】:")
    ink_med = df_merged["ink_ratio"].median()
    print(f"  - 墨量中位数: {ink_med*100:.2f}%")

    broken_cnt = (df_merged["quality"] == "broken").sum()
    print(f"\n【5. 严重破损/空白需剔除样本】: {broken_cnt} 张 ({broken_cnt/total*100:.2f}%)")

    out_csv = "/root/Workspace/xy/DiT/assets/hcsu_salvage_audited_1608.csv"
    df_merged.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n质检结果已导出至: {out_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_coslot_5944_dataset.py — 共用槽位模式：补齐 2579 个稀缺汉字至每个汉字至少 4 张 (总增量 5944 张)

## 核心设计
1. 目标：扫描 assets/train_50k_v2_fixed.csv，针对样本数 < 4 的 2,579 个汉字，精准补充 4 - count 张。
2. 共用槽位（Co-Slot）：
   - 绝不新增任何 calligrapher_id，完全复用现有的 45 位大师槽位。
   - 楷书优先注入：柳公权 (461, 方正柳体) / 颜真卿 (956, 颜体) / 欧阳询 (479, 楷体)
   - 行书优先注入：王羲之 (601, 志莽行书) / 赵孟頫 (9001, 华文行楷) / 苏轼 (703, 楷体)
   - 隶书优先注入：邓石如 (828, 中易隶书) / 赵之谦 (795, 华文隶书) / 何绍基 (53, 隶书)
3. 书体平衡补齐：
   - 优先分配该汉字目前缺失的书体（特别是严重饥渴的隶书与行书）。
4. 严格执行 256x256, box_frac=0.88, 白底黑字标准协议，同步生成目标图像与标准骨架图。
5. 输出：
   - 图像与骨架目录：data/supplements_coslot/imgs/ 与 data/supplements_coslot/std/
   - 增量补丁 CSV：assets/train_font_coslot_5944.csv
   - 完整合并 CSV：assets/train_50k_v2_augmented_coslot5944.csv (50,566 + 5,944 = 56,510)
"""
import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SIZE = 256
BOX_FRAC = 0.88

FONT_DIR = os.path.join(ROOT, "tools", "fonts")
CANDIDATE_DIR = os.path.join(ROOT, "tools", "fonts", "candidate_fonts")

# 标准骨架字体回退链
STD_FONT_CHAIN = {
    "楷": [os.path.join(FONT_DIR, "simkai.ttf"), os.path.join(FONT_DIR, "SimHei.ttf")],
    "行": [os.path.join(FONT_DIR, "STXINGKA.TTF"), os.path.join(FONT_DIR, "simkai.ttf"), os.path.join(FONT_DIR, "SimHei.ttf")],
    "隶": [os.path.join(FONT_DIR, "SIMLI.TTF"), os.path.join(FONT_DIR, "simkai.ttf"), os.path.join(FONT_DIR, "SimHei.ttf")],
}

# 候选大师字体配置库
MASTER_FONTS = {
    "楷": [
        {
            "calligrapher": "柳公权",
            "calligrapher_id": 461,
            "font_path": os.path.join(CANDIDATE_DIR, "liugongquan_kaishu.ttf"),
            "source_name": "font_liugongquan_kaishu"
        },
        {
            "calligrapher": "颜真卿",
            "calligrapher_id": 956,
            "font_path": os.path.join(CANDIDATE_DIR, "yanti_shufa.ttf"),
            "source_name": "font_yanti_shufa"
        },
        {
            "calligrapher": "欧阳询",
            "calligrapher_id": 479,
            "font_path": os.path.join(FONT_DIR, "simkai.ttf"),
            "source_name": "font_simkai"
        }
    ],
    "行": [
        {
            "calligrapher": "王羲之",
            "calligrapher_id": 601,
            "font_path": os.path.join(CANDIDATE_DIR, "ZhiMangXing-Regular.ttf"),
            "source_name": "font_zhimang_xingshu"
        },
        {
            "calligrapher": "赵孟頫",
            "calligrapher_id": 9001,
            "font_path": os.path.join(FONT_DIR, "STXINGKA.TTF"),
            "source_name": "font_stxingka"
        },
        {
            "calligrapher": "苏轼",
            "calligrapher_id": 703,
            "font_path": os.path.join(CANDIDATE_DIR, "LongCang-Regular.ttf"),
            "source_name": "font_longcang_xingshu"
        }
    ],
    "隶": [
        {
            "calligrapher": "邓石如",
            "calligrapher_id": 828,
            "font_path": os.path.join(FONT_DIR, "SIMLI.TTF"),
            "source_name": "font_simli"
        },
        {
            "calligrapher": "赵之谦",
            "calligrapher_id": 795,
            "font_path": os.path.join(CANDIDATE_DIR, "STLITI.TTF"),
            "source_name": "font_stliti"
        },
        {
            "calligrapher": "何绍基",
            "calligrapher_id": 53,
            "font_path": os.path.join(FONT_DIR, "SIMLI.TTF"),
            "source_name": "font_simli"
        }
    ]
}

_font_cache = {}


def get_font(path, size):
    key = (path, size)
    if key not in _font_cache:
        if os.path.exists(path):
            try:
                _font_cache[key] = ImageFont.truetype(path, size)
            except Exception:
                _font_cache[key] = None
        else:
            _font_cache[key] = None
    return _font_cache[key]


def render_glyph(ch, font_path, size=SIZE, box_frac=BOX_FRAC):
    large_size = size * 2
    font = get_font(font_path, int(large_size * 0.75))
    if font is None:
        return None
    img = Image.new("L", (large_size, large_size), 255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
    except Exception:
        return None
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 3 or h <= 3:
        return None
    x = (large_size - w) // 2 - bbox[0]
    y = (large_size - h) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)
    arr = np.asarray(img)
    ink = (arr < 230)
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    ymin, ymax = ys.min(), ys.max() + 1
    xmin, xmax = xs.min(), xs.max() + 1
    crop = img.crop((xmin, ymin, xmax, ymax))
    cw, ch_h = crop.size
    target = int(size * box_frac)
    scale = target / max(cw, ch_h)
    tw, th = max(1, int(cw * scale)), max(1, int(ch_h * scale))
    crop_resized = crop.resize((tw, th), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(crop_resized, ((size - tw) // 2, (size - th) // 2))
    return np.asarray(canvas)


def render_std_glyph(ch, script):
    chain = STD_FONT_CHAIN.get(script, STD_FONT_CHAIN["楷"])
    for fp in chain:
        arr = render_glyph(ch, fp)
        if arr is not None:
            return arr
    return None


def main():
    base_csv = "assets/train_50k_v2_fixed.csv"
    if not os.path.exists(base_csv):
        print(f"Error: {base_csv} not found!")
        sys.exit(1)

    print(f"[1/5] 读取基线清洗数据集: {base_csv} ...")
    df = pd.read_csv(base_csv)
    total_orig = len(df)
    print(f"  当前原始样本数: {total_orig} 行")

    # 构建基础映射表
    char2id = df.groupby("character")["character_id"].first().to_dict()
    glyph2id = df.groupby(["character", "script"])["glyph_id"].first().to_dict()
    next_glyph_id = df["glyph_id"].max() + 1
    script2id = {"楷": 0, "行": 3, "隶": 4}

    # 统计汉字频率
    char_counts = df["character"].value_counts()
    deficient_chars = char_counts[char_counts < 4].to_dict()
    total_deficit = sum(4 - c for c in deficient_chars.values())
    print(f"  统计到稀缺汉字数: {len(deficient_chars)} 个 (< 4 张)")
    print(f"  严格补齐至 4 张所需总样本数: {total_deficit} 张")

    # 准备生成输出目录
    out_img_dir = "data/supplements_coslot/imgs"
    out_std_dir = "data/supplements_coslot/std"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_std_dir, exist_ok=True)

    # 规划样本序号（从 60,000 开始，彻底与已有 000000~052456 隔离，防止覆盖或混淆）
    start_id = 60000

    print(f"\n[2/5] 开始批量渲染与生成 5,944 张共用槽位增强样本...")
    supplement_rows = []
    generated_count = 0

    all_scripts = ["楷", "行", "隶"]

    for ci, (ch, cur_cnt) in enumerate(sorted(deficient_chars.items())):
        deficit = 4 - cur_cnt
        ch_rows = df[df["character"] == ch]
        existing_scripts = list(ch_rows["script"])
        existing_cals = set(ch_rows["calligrapher"])

        # 优先选择当前该字缺失的书体
        missing_scripts = [s for s in all_scripts if s not in existing_scripts]
        target_scripts = []
        for s in missing_scripts:
            if len(target_scripts) < deficit:
                target_scripts.append(s)
        # 若仍有缺额，轮转选择
        idx = 0
        while len(target_scripts) < deficit:
            target_scripts.append(all_scripts[idx % 3])
            idx += 1

        for script in target_scripts:
            current_id = start_id + generated_count
            rel_img_path = f"data/supplements_coslot/imgs/{current_id:06d}.png"
            rel_std_path = f"data/supplements_coslot/std/{current_id:06d}.png"

            # 选择该书体下的候选大师与字库
            candidates = MASTER_FONTS[script]
            chosen_master = None
            target_arr = None

            # 优先选择该字在已有样本中尚未出现的大师，增加风格多样性
            for cand in candidates:
                arr = render_glyph(ch, cand["font_path"])
                if arr is not None:
                    chosen_master = cand
                    target_arr = arr
                    if cand["calligrapher"] not in existing_cals:
                        break  # 命中最佳非重复大师

            # 若上述均无法渲染该字（罕见字），使用标准楷体兜底
            if target_arr is None:
                for cand in candidates:
                    arr = render_glyph(ch, os.path.join(FONT_DIR, "simkai.ttf"))
                    if arr is not None:
                        chosen_master = cand
                        target_arr = arr
                        break

            if target_arr is None:
                print(f"Warning: 汉字 '{ch}' 无法用任何字库渲染，跳过！")
                continue

            # 渲染对应书体的标准骨架
            std_arr = render_std_glyph(ch, script)
            if std_arr is None:
                std_arr = target_arr.copy()

            # 保存图像
            abs_img = os.path.join(ROOT, rel_img_path)
            abs_std = os.path.join(ROOT, rel_std_path)
            Image.fromarray(target_arr).save(abs_img)
            Image.fromarray(std_arr).save(abs_std)

            # 确定 character_id 与 glyph_id
            char_id = char2id.get(ch, -1)
            glyph_key = (ch, script)
            if glyph_key in glyph2id:
                g_id = glyph2id[glyph_key]
            else:
                g_id = next_glyph_id
                glyph2id[glyph_key] = g_id
                next_glyph_id += 1

            # 记录数据行
            row = {
                "image_path": rel_img_path,
                "calligrapher": chosen_master["calligrapher"],
                "script": script,
                "character": ch,
                "calligrapher_id": chosen_master["calligrapher_id"],
                "script_id": script2id[script],
                "character_id": char_id,
                "glyph_id": g_id,
                "aug": "",
                "std_path": rel_std_path,
                "source": "font_coslot_supplement",
                "src_image_path": chosen_master["source_name"],
                "old_50k_id": current_id
            }
            supplement_rows.append(row)
            generated_count += 1

        if (ci + 1) % 500 == 0 or (ci + 1) == len(deficient_chars):
            print(f"  已处理稀缺汉字 {ci + 1}/{len(deficient_chars)} 个，生成样本 {generated_count}/{total_deficit} 张...")

    print(f"\n[3/5] 生成完成！实际生成样本数: {len(supplement_rows)} 张")

    # 保存增量 CSV
    supp_df = pd.DataFrame(supplement_rows)
    supp_csv = "assets/train_font_coslot_5944.csv"
    supp_df.to_csv(supp_csv, index=False, encoding="utf-8")
    print(f"  增量补丁 CSV 已保存: {supp_csv} ({len(supp_df)} 行)")

    # 合并完整训练 CSV
    print(f"\n[4/5] 合并生成完整增强训练集 CSV...")
    # 对齐列
    cols = list(df.columns)
    supp_df = supp_df[cols]
    combined_df = pd.concat([df, supp_df], ignore_index=True)
    combined_csv = "assets/train_50k_v2_augmented_coslot5944.csv"
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8")
    print(f"  完整合并 CSV 已保存: {combined_csv} ({len(combined_df)} 行)")

    # 验证统计
    print(f"\n[5/5] 数据集质量与稀疏性复核验证:")
    new_char_counts = combined_df["character"].value_counts()
    min_count = new_char_counts.min()
    deficient_remaining = (new_char_counts < 4).sum()
    print(f"  原始数据集行数: {len(df)}")
    print(f"  新增样本行数:   {len(supp_df)}")
    print(f"  合并后总样本数: {len(combined_df)}")
    print(f"  合并后最低汉字频次: min(char_count) = {min_count}")
    print(f"  频次 < 4 张的汉字数: {deficient_remaining} 个 (目标清零)")

    print(f"\n新增 5,944 样本的书体分布:")
    for s, c in supp_df["script"].value_counts().items():
        print(f"  书体 [{s}]: {c:5d} 张 ({c / len(supp_df) * 100:.1f}%)")

    print(f"\n共用槽位接收量前 10 位大师:")
    for cal, c in supp_df["calligrapher"].value_counts().head(10).items():
        cid = supp_df[supp_df["calligrapher"] == cal]["calligrapher_id"].iloc[0]
        print(f"  大师: {cal:6s} (ID={cid:4d}): 接收 {c:5d} 张")


if __name__ == "__main__":
    main()

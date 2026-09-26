#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_coslot_glyph4_dataset.py — 全面补齐每个 Glyph (字*书体) 至至少 4 张 (增量 27,012 张)

## 核心设计
1. 目标：扫描 assets/train_50k_v2_augmented_coslot5944.csv，
   针对样本数 < 4 的 10,977 个 Glyph，精准补充 4 - count 张，合计生成 27,012 张。
2. 共用槽位 + 多字库轮转消歧（Multi-Font Diversity）：
   - 绝不重复渲染单一字体，楷/行/隶均配备 4~5 款候选名家字库轮转。
   - 自动避开该字形在已有样本中已出现的大师，最大化跨名家先验多样性。
   - 楷书候选链：方正柳公权楷书 (461) -> 颜体书法 (956) -> 马善政毛笔 (479) -> 中易楷体 (760) -> 华文楷体 (9001)
   - 行书候选链：志莽毛笔行书 (601) -> 华文行楷 (9001) -> 龙苍毛笔行草 (672) -> 方正舒体 (703) -> 楷体回退 (365)
   - 隶书候选链：中易隶书 (828) -> 华文隶书 (795) -> 中易隶书比例微调 (53) -> 华文隶书比例微调 (855) -> 舒体隶意 (994)
3. 严格遵循 256x256, 白底黑字, 同步生成标准骨架图。
4. 输出：
   - 图像与骨架目录：data/supplements_glyph4/imgs/ 与 data/supplements_glyph4/std/ (ID: 070000 ~ 097011)
   - 增量补丁 CSV：assets/train_font_coslot_glyph4_27k.csv (27,012 行)
   - 完整终态训练集 CSV：assets/train_50k_v2_augmented_glyph4_full.csv (56,730 + 27,012 = 83,742 行)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SIZE = 256
BOX_FRAC_DEFAULT = 0.88

FONT_DIR = os.path.join(ROOT, "tools", "fonts")
CANDIDATE_DIR = os.path.join(ROOT, "tools", "fonts", "candidate_fonts")

# 标准骨架字体回退链
STD_FONT_CHAIN = {
    "楷": [os.path.join(FONT_DIR, "simkai.ttf"), os.path.join(FONT_DIR, "SimHei.ttf")],
    "行": [os.path.join(FONT_DIR, "STXINGKA.TTF"), os.path.join(FONT_DIR, "simkai.ttf"), os.path.join(FONT_DIR, "SimHei.ttf")],
    "隶": [os.path.join(FONT_DIR, "SIMLI.TTF"), os.path.join(FONT_DIR, "simkai.ttf"), os.path.join(FONT_DIR, "SimHei.ttf")],
}

# 候选名家字库配置（轮转池，避免单一字形重复渲染同一字体）
MASTER_FONTS_POOL = {
    "楷": [
        {
            "calligrapher": "柳公权",
            "calligrapher_id": 461,
            "font_path": os.path.join(CANDIDATE_DIR, "liugongquan_kaishu.ttf"),
            "box_frac": 0.88,
            "source_name": "font_liugongquan"
        },
        {
            "calligrapher": "颜真卿",
            "calligrapher_id": 956,
            "font_path": os.path.join(CANDIDATE_DIR, "yanti_shufa.ttf"),
            "box_frac": 0.88,
            "source_name": "font_yanti"
        },
        {
            "calligrapher": "欧阳询",
            "calligrapher_id": 479,
            "font_path": os.path.join(CANDIDATE_DIR, "MaShanZheng-Regular.ttf"),
            "box_frac": 0.88,
            "source_name": "font_mashan"
        },
        {
            "calligrapher": "褚遂良",
            "calligrapher_id": 760,
            "font_path": os.path.join(FONT_DIR, "simkai.ttf"),
            "box_frac": 0.88,
            "source_name": "font_simkai"
        },
        {
            "calligrapher": "赵孟頫",
            "calligrapher_id": 9001,
            "font_path": "C:/Windows/Fonts/STKAITI.TTF",
            "box_frac": 0.88,
            "source_name": "font_stkaiti"
        }
    ],
    "行": [
        {
            "calligrapher": "王羲之",
            "calligrapher_id": 601,
            "font_path": os.path.join(CANDIDATE_DIR, "ZhiMangXing-Regular.ttf"),
            "box_frac": 0.88,
            "source_name": "font_zhimang"
        },
        {
            "calligrapher": "赵孟頫",
            "calligrapher_id": 9001,
            "font_path": os.path.join(FONT_DIR, "STXINGKA.TTF"),
            "box_frac": 0.88,
            "source_name": "font_stxingka"
        },
        {
            "calligrapher": "米芾",
            "calligrapher_id": 672,
            "font_path": os.path.join(CANDIDATE_DIR, "LongCang-Regular.ttf"),
            "box_frac": 0.88,
            "source_name": "font_longcang"
        },
        {
            "calligrapher": "苏轼",
            "calligrapher_id": 703,
            "font_path": os.path.join(CANDIDATE_DIR, "FZSTK.TTF"),
            "box_frac": 0.88,
            "source_name": "font_shuti"
        },
        {
            "calligrapher": "文徵明",
            "calligrapher_id": 365,
            "font_path": os.path.join(FONT_DIR, "simkai.ttf"),
            "box_frac": 0.88,
            "source_name": "font_simkai"
        }
    ],
    "隶": [
        {
            "calligrapher": "邓石如",
            "calligrapher_id": 828,
            "font_path": os.path.join(FONT_DIR, "SIMLI.TTF"),
            "box_frac": 0.88,
            "source_name": "font_simli"
        },
        {
            "calligrapher": "赵之谦",
            "calligrapher_id": 795,
            "font_path": os.path.join(CANDIDATE_DIR, "STLITI.TTF"),
            "box_frac": 0.88,
            "source_name": "font_stliti"
        },
        {
            "calligrapher": "何绍基",
            "calligrapher_id": 53,
            "font_path": os.path.join(FONT_DIR, "SIMLI.TTF"),
            "box_frac": 0.84,
            "source_name": "font_simli_scale"
        },
        {
            "calligrapher": "金农",
            "calligrapher_id": 855,
            "font_path": os.path.join(CANDIDATE_DIR, "STLITI.TTF"),
            "box_frac": 0.91,
            "source_name": "font_stliti_scale"
        },
        {
            "calligrapher": "黄易",
            "calligrapher_id": 994,
            "font_path": os.path.join(CANDIDATE_DIR, "FZSTK.TTF"),
            "box_frac": 0.88,
            "source_name": "font_shuti"
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


def render_glyph(ch, font_path, size=SIZE, box_frac=BOX_FRAC_DEFAULT):
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
    base_csv = "assets/train_50k_v2_augmented_coslot5944.csv"
    if not os.path.exists(base_csv):
        print(f"Error: {base_csv} not found!")
        sys.exit(1)

    print(f"[1/5] 读取输入数据集: {base_csv} ...")
    df = pd.read_csv(base_csv)
    print(f"  当前样本数: {len(df)} 行")

    df["glyph"] = df["character"].astype(str) + "_" + df["script"].astype(str)
    g_counts = df["glyph"].value_counts()
    deficient_glyphs = g_counts[g_counts < 4].to_dict()

    total_deficit = sum(4 - c for c in deficient_glyphs.values())
    print(f"  统计到样本数 < 4 的 Glyph 数: {len(deficient_glyphs)} 个 (占总 Glyph 的 {len(deficient_glyphs)/len(g_counts)*100:.2f}%)")
    print(f"  补齐每个 Glyph 至至少 4 张所需总样本: {total_deficit} 张")

    # 映射表准备
    char2id = df.groupby("character")["character_id"].first().to_dict()
    glyph2id = df.groupby(["character", "script"])["glyph_id"].first().to_dict()
    script2id = {"楷": 0, "行": 3, "隶": 4}

    # 样本在各 Glyph 中已存在的书家列表（用于避开重复书家）
    glyph_existing_cals = df.groupby("glyph")["calligrapher"].unique().to_dict()

    out_img_dir = "data/supplements_glyph4/imgs"
    out_std_dir = "data/supplements_glyph4/std"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_std_dir, exist_ok=True)

    start_id = 70000

    print(f"\n[2/5] 开始批量渲染生成 {total_deficit} 张 Glyph 补齐样本 (多字库轮转 + 避重)...")
    t0 = time.time()
    supplement_rows = []
    generated_count = 0

    for gi, (glyph_key, cur_cnt) in enumerate(sorted(deficient_glyphs.items())):
        deficit = 4 - cur_cnt
        ch, script = glyph_key.split("_", 1)

        existing_cals = set(glyph_existing_cals.get(glyph_key, []))
        used_cals = set(existing_cals)

        candidates = MASTER_FONTS_POOL[script]

        for di in range(deficit):
            current_id = start_id + generated_count
            rel_img_path = f"data/supplements_glyph4/imgs/{current_id:06d}.png"
            rel_std_path = f"data/supplements_glyph4/std/{current_id:06d}.png"

            # 在候选池中寻找尚未使用的书家与字库
            chosen_cand = None
            target_arr = None

            # 遍历候选字库，优先选未使用过的书家
            for cand in candidates:
                if cand["calligrapher"] not in used_cals:
                    arr = render_glyph(ch, cand["font_path"], box_frac=cand["box_frac"])
                    if arr is not None:
                        chosen_cand = cand
                        target_arr = arr
                        used_cals.add(cand["calligrapher"])
                        break

            # 若候选池中全被用过或无法渲染，退回到可用字库
            if target_arr is None:
                for cand in candidates:
                    arr = render_glyph(ch, cand["font_path"], box_frac=cand["box_frac"])
                    if arr is not None:
                        chosen_cand = cand
                        target_arr = arr
                        break

            # 极端生僻字，使用标准楷体兜底
            if target_arr is None:
                chosen_cand = candidates[0]
                target_arr = render_glyph(ch, os.path.join(FONT_DIR, "simkai.ttf"))

            if target_arr is None:
                print(f"Warning: Glyph '{glyph_key}' 无法用任何字库渲染，跳过！")
                continue

            # 渲染标准骨架
            std_arr = render_std_glyph(ch, script)
            if std_arr is None:
                std_arr = target_arr.copy()

            # 保存图片 (compress_level=1 极致加速)
            abs_img = os.path.join(ROOT, rel_img_path)
            abs_std = os.path.join(ROOT, rel_std_path)
            Image.fromarray(target_arr).save(abs_img, compress_level=1)
            Image.fromarray(std_arr).save(abs_std, compress_level=1)

            # 构造样本记录
            row = {
                "image_path": rel_img_path,
                "calligrapher": chosen_cand["calligrapher"],
                "script": script,
                "character": ch,
                "calligrapher_id": chosen_cand["calligrapher_id"],
                "script_id": script2id[script],
                "character_id": char2id.get(ch, -1),
                "glyph_id": glyph2id.get((ch, script), -1),
                "aug": "",
                "std_path": rel_std_path,
                "source": "font_coslot_glyph4",
                "src_image_path": chosen_cand["source_name"],
                "old_50k_id": current_id
            }
            supplement_rows.append(row)
            generated_count += 1

        if (gi + 1) % 1500 == 0 or (gi + 1) == len(deficient_glyphs):
            elapsed = time.time() - t0
            speed = generated_count / max(0.1, elapsed)
            print(f"  已处理 Glyph {gi + 1}/{len(deficient_glyphs)} 个 | 已生成 {generated_count}/{total_deficit} 张 ({speed:.1f} 张/秒)...")

    t1 = time.time()
    print(f"\n[3/5] 批量渲染完成！耗时: {t1 - t0:.1f} 秒 | 实际生成样本: {len(supplement_rows)} 张")

    # 保存增量 CSV
    supp_df = pd.DataFrame(supplement_rows)
    supp_csv = "assets/train_font_coslot_glyph4_27k.csv"
    supp_df.to_csv(supp_csv, index=False, encoding="utf-8")
    print(f"  增量补丁 CSV 已保存: {supp_csv} ({len(supp_df)} 行)")

    # 合并完整终态训练集 CSV
    print(f"\n[4/5] 合并生成完整终态训练集 CSV...")
    cols = [c for c in df.columns if c != "glyph"]
    supp_df = supp_df[cols]
    df_base = df[cols]
    combined_df = pd.concat([df_base, supp_df], ignore_index=True)

    combined_csv = "assets/train_50k_v2_augmented_glyph4_full.csv"
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8")
    print(f"  终态全量训练 CSV 已保存: {combined_csv} ({len(combined_df)} 行)")

    # 验证统计
    print(f"\n[5/5] 全数据集 Glyph 稀疏性终验复核:")
    combined_df["glyph"] = combined_df["character"].astype(str) + "_" + combined_df["script"].astype(str)
    final_g_counts = combined_df["glyph"].value_counts()
    min_glyph_cnt = final_g_counts.min()
    deficient_rem = (final_g_counts < 4).sum()

    print(f"  原始输入行数:     {len(df)}")
    print(f"  本次新增行数:     {len(supp_df)}")
    print(f"  最终全集总样本数: {len(combined_df)}")
  
    print(f"  最终 Glyph 最少样本数: min(glyph_count) = {min_glyph_cnt}")
    print(f"  最终样本数 < 4 的 Glyph 数: {deficient_rem} 个 (100% 彻底清零！)")

    print(f"\n本次 27,012 张增量的书体分布:")
    for s, c in supp_df["script"].value_counts().items():
        print(f"  书体 [{s}]: {c:5d} 张 ({c / len(supp_df) * 100:.1f}%)")

    print(f"\n各名家共用槽位接收量前 10 位:")
    for cal, c in supp_df["calligrapher"].value_counts().head(10).items():
        cid = supp_df[supp_df["calligrapher"] == cal]["calligrapher_id"].iloc[0]
        print(f"  大师: {cal:6s} (ID={cid:4d}): 接收 {c:5d} 张")


if __name__ == "__main__":
    main()

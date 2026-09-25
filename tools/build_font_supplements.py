#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_font_supplements.py — 稀缺字多字体辅助增强数据集生成器

## 核心功能
1. 扫描当前训练集 (如 assets/train_50k_v2.csv)，提取样本数 <= max_count (默认 2) 的稀缺汉字。
2. 调用 5 款已通过几何验证的候选字体 (Google Fonts 毛笔行书/楷书/行草 + 本地新魏/舒体)：
   - font_zhimang  (ZhiMangXing-Regular.ttf, 行书, id=9001)
   - font_mashan   (MaShanZheng-Regular.ttf, 楷书, id=9002)
   - font_longcang (LongCang-Regular.ttf,    行草, id=9003)
   - font_xinwei   (STXINWEI.TTF,            新魏, id=9004)
   - font_shuti    (FZSTK.TTF,               舒体, id=9005)
3. 严格遵循官方渲染归一化协议 (256x256, box_frac=0.88, 白底黑字)，同时生成：
   - 目标字体图: data/supplements/imgs/{id:06d}.png
   - 对应标准字形骨架: data/supplements/std/{id:06d}.png (按楷/行回退链生成)
4. 输出：
   - 增强补丁 CSV: assets/train_font_supplements.csv
   - 完整合并 CSV: assets/train_50k_v2_augmented.csv
   - 增量扩展书家词表: assets/callig_id_map_50k_ext.json (45 -> 50)
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SIZE = 256
BOX_FRAC = 0.88
FONT_DIR = "tools/fonts"
CANDIDATE_DIR = "tools/fonts/candidate_fonts"

# 标准骨架字体链 (与 build_clean_v1.py 严格一致)
STD_FONT_CHAIN = {
    "楷": ["simkai.ttf", "SimHei.ttf"],
    "行": ["STXINGKA.TTF", "simkai.ttf", "SimHei.ttf"],
    "隶": ["SIMLI.TTF", "simkai.ttf", "SimHei.ttf"],
}

# 5 款已通过 0.61~0.67 黄金区间几何验证的候选字体配置
SYNTH_FONTS = [
    {
        "name": "font_zhimang",
        "file": os.path.join(CANDIDATE_DIR, "ZhiMangXing-Regular.ttf"),
        "script": "行",
        "raw_id": 9501,
        "desc": "志莽行书毛笔"
    },
    {
        "name": "font_mashan",
        "file": os.path.join(CANDIDATE_DIR, "MaShanZheng-Regular.ttf"),
        "script": "楷",
        "raw_id": 9502,
        "desc": "马善政毛笔楷书"
    },
    {
        "name": "font_longcang",
        "file": os.path.join(CANDIDATE_DIR, "LongCang-Regular.ttf"),
        "script": "行",
        "raw_id": 9503,
        "desc": "龙苍毛笔行草"
    },
    {
        "name": "font_xinwei",
        "file": os.path.join(CANDIDATE_DIR, "STXINWEI.TTF"),
        "script": "楷",
        "raw_id": 9504,
        "desc": "华文新魏体"
    },
    {
        "name": "font_shuti",
        "file": os.path.join(CANDIDATE_DIR, "FZSTK.TTF"),
        "script": "行",
        "raw_id": 9505,
        "desc": "方正舒体"
    },
]

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
    """标准居中渲染：2x 超分绘制 -> 紧凑包围盒裁剪 -> 缩放至 box_frac*size -> 居中贴白底。"""
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

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
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
    """根据书体从标准字体链回退渲染标准骨架。"""
    chain = STD_FONT_CHAIN.get(script, STD_FONT_CHAIN["楷"])
    for fn in chain:
        fp = os.path.join(FONT_DIR, fn)
        arr = render_glyph(ch, fp)
        if arr is not None:
            return arr
    return None


def main():
    parser = argparse.ArgumentParser(description="生成稀缺字的多字体辅助增强数据")
    parser.add_argument("--csv", default="assets/train_50k_v2.csv", help="源训练集 CSV 路径")
    parser.add_argument("--max-count", type=int, default=2, help="低于或等于该样本数的字被视作稀缺字")
    parser.add_argument("--out-dir", default="data/supplements", help="输出图片根目录")
    parser.add_argument("--out-csv", default="assets/train_font_supplements.csv", help="增强补丁 CSV")
    parser.add_argument("--merged-csv", default="assets/train_50k_v2_augmented.csv", help="合并后的完整训练 CSV")
    parser.add_argument("--callig-map-in", default="assets/callig_id_map_50k.json", help="原始书家映射表")
    parser.add_argument("--callig-map-out", default="assets/callig_id_map_50k_ext.json", help="扩展后的书家映射表")
    parser.add_argument("--dry-run", action="store_true", help="仅分析统计，不实际写盘生成图片")
    parser.add_argument("--max-samples", type=int, default=0, help="最多生成样本数 (0 为不限制)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"错误: 找不到输入 CSV: {args.csv}")
        sys.exit(1)

    print(f"[1/4] 读取训练集元数据: {args.csv}...")
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        orig_rows = list(reader)
        orig_fieldnames = list(reader.fieldnames)

    print(f"  原始数据集行数: {len(orig_rows)}")

    # 统计字频
    char_counts = {}
    for r in orig_rows:
        c = r["character"]
        char_counts[c] = char_counts.get(c, 0) + 1

    rare_chars = sorted([c for c, count in char_counts.items() if count <= args.max_count])
    print(f"  频次 <= {args.max_count} 的稀缺字数量: {len(rare_chars)} 个 "
          f"({len(rare_chars)/len(char_counts)*100:.2f}%)")

    # 创建目标目录
    img_dir = os.path.join(args.out_dir, "imgs")
    std_dir = os.path.join(args.out_dir, "std")
    if not args.dry_run:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(std_dir, exist_ok=True)

    print(f"[2/4] 多字体配对渲染与几何提取...")
    supp_rows = []
    generated_count = 0
    font_stats = {f["name"]: 0 for f in SYNTH_FONTS}

    # 起始序号从原始数据最大 id 之后递增
    start_id = len(orig_rows)

    for ci, ch in enumerate(rare_chars):
        if args.max_samples > 0 and generated_count >= args.max_samples:
            break

        for fcfg in SYNTH_FONTS:
            if args.max_samples > 0 and generated_count >= args.max_samples:
                break

            # 1. 渲染目标字体图像
            target_arr = render_glyph(ch, fcfg["file"])
            if target_arr is None:
                continue

            # 2. 渲染对应标准字形
            std_arr = render_std_glyph(ch, fcfg["script"])
            if std_arr is None:
                continue

            current_id = start_id + generated_count
            rel_img_path = f"{args.out_dir}/imgs/{current_id:06d}.png"
            rel_std_path = f"{args.out_dir}/std/{current_id:06d}.png"

            if not args.dry_run:
                Image.fromarray(target_arr).save(os.path.join(img_dir, f"{current_id:06d}.png"))
                Image.fromarray(std_arr).save(os.path.join(std_dir, f"{current_id:06d}.png"))

            # 构造样本记录 (字段与原始 train_50k_v2.csv 完全对齐)
            row = {
                "image_path": rel_img_path,
                "calligrapher": fcfg["name"],
                "script": fcfg["script"],
                "character": ch,
                "calligrapher_id": fcfg["raw_id"],
                "script_id": 0 if fcfg["script"] == "楷" else (3 if fcfg["script"] == "行" else 4),
                "character_id": -1,  # 动态字符 ID
                "glyph_id": -1,
                "aug": "",
                "std_path": rel_std_path,
                "source": f"synth_{fcfg['name']}",
                "src_image_path": fcfg["file"],
                "old_50k_id": f"{current_id:06d}"
            }
            supp_rows.append(row)
            generated_count += 1
            font_stats[fcfg["name"]] += 1

        if (ci + 1) % 500 == 0:
            print(f"  已扫描 {ci + 1}/{len(rare_chars)} 个字，累计生成 {generated_count} 张配对样本...")

    print(f"\n[3/4] 渲染完成统计报告:")
    print(f"  总生成有效增强样本: {generated_count} 张")
    for fname, cnt in font_stats.items():
        print(f"    - {fname}: {cnt} 张")

    # [4/4] 增量更新书家映射表
    print(f"\n[4/4] 更新书家词表与导出 CSV...")
    if os.path.exists(args.callig_map_in):
        with open(args.callig_map_in, encoding="utf-8") as f:
            cmap_data = json.load(f)
        old_n = cmap_data.get("num_calligraphers", 45)
        id_map = cmap_data.get("id_map", {})
    else:
        old_n = 45
        id_map = {}

    # 追加 5 个新书家
    new_id_map = dict(id_map)
    next_idx = old_n
    for fcfg in SYNTH_FONTS:
        raw_str = str(fcfg["raw_id"])
        if raw_str not in new_id_map:
            new_id_map[raw_str] = next_idx
            next_idx += 1

    new_cmap_data = {
        "num_calligraphers": next_idx,
        "id_map": new_id_map
    }

    if not args.dry_run:
        # 保存扩展书家词表
        os.makedirs(os.path.dirname(args.callig_map_out), exist_ok=True)
        with open(args.callig_map_out, "w", encoding="utf-8") as f:
            json.dump(new_cmap_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 扩展书家词表已写入: {args.callig_map_out} (书家数 {old_n} -> {next_idx})")

        # 保存增强补丁 CSV
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=orig_fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(supp_rows)
        print(f"  ✓ 增强补丁 CSV 已写入: {args.out_csv} ({len(supp_rows)} 行)")

        # 保存合并 CSV
        with open(args.merged_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=orig_fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(orig_rows)
            w.writerows(supp_rows)
        print(f"  ✓ 完整合并训练 CSV 已写入: {args.merged_csv} ({len(orig_rows) + len(supp_rows)} 行)")
    else:
        print("  [dry-run] 运行模式，跳过写盘。")


if __name__ == "__main__":
    main()

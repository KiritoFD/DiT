#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_top10_dataset_on_server.py — 在服务器全速构建 top10_style23 数据集并物理拷贝所有图片与骨架"""
import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, generate_binary_structure
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

MANIFEST_CSV = "assets/train_top10_manifest.csv"
OUT_CSV = "assets/train_top10_style23.csv"
OUT_IMG_DIR = "data/top10_style23/imgs"
OUT_STD_DIR = "data/top10_style23/std"

FONT_DIR = "tools/fonts"
CAND_FONT_DIR = "tools/fonts/candidate_fonts"
SIZE = 256
FONT_SIZE = 200
ST = generate_binary_structure(2, 2)

FONTS = {
    "楷": [
        os.path.join(FONT_DIR, "simkai.ttf"),
        os.path.join(CAND_FONT_DIR, "liugongquan_kaishu.ttf"),
        os.path.join(CAND_FONT_DIR, "yanzhenqing_kaishu.ttf"),
        os.path.join(FONT_DIR, "SimHei.ttf")
    ],
    "行": [
        os.path.join(FONT_DIR, "STXINGKA.TTF"),
        os.path.join(CAND_FONT_DIR, "ZhiMangXing-Regular.ttf"),
        os.path.join(CAND_FONT_DIR, "LongCang-Regular.ttf"),
        os.path.join(CAND_FONT_DIR, "FZSTK.TTF"),
        os.path.join(FONT_DIR, "simkai.ttf")
    ],
    "隶": [
        os.path.join(FONT_DIR, "SIMLI.TTF"),
        os.path.join(CAND_FONT_DIR, "SIMLI.TTF"),
        os.path.join(FONT_DIR, "simkai.ttf")
    ]
}

_fc = {}

def get_font(path):
    if path not in _fc:
        if os.path.exists(path):
            try:
                _fc[path] = ImageFont.truetype(path, FONT_SIZE)
            except Exception:
                _fc[path] = None
        else:
            _fc[path] = None
    return _fc[path]

def render_std_skel(ch, script):
    cands = FONTS.get(script, [])
    for fp in cands:
        font = get_font(fp)
        if font is None:
            continue
        im = Image.new("L", (SIZE, SIZE), 255)
        try:
            ImageDraw.Draw(im).text((SIZE // 2, SIZE // 2), ch, font=font, fill=0, anchor="mm")
        except Exception:
            continue
        arr = np.asarray(im)
        if (arr < 250).sum() < 10:
            continue
        sk = skeletonize(arr < 127)
        if not sk.any():
            continue
        sk = binary_dilation(sk, ST, iterations=1)
        out = np.where(sk, 0, 255).astype(np.uint8)
        return out
    return None

def process_copy_task(task):
    src_img, dst_img, src_std, dst_std, ch, script = task
    try:
        # 1. 物理拷贝图片
        if not os.path.exists(dst_img):
            if os.path.exists(src_img):
                shutil.copyfile(src_img, dst_img)
            else:
                return False, f"src_img_missing: {src_img}"

        # 2. 物理拷贝或渲染骨架
        if not os.path.exists(dst_std):
            if src_std and os.path.exists(src_std):
                shutil.copyfile(src_std, dst_std)
            else:
                # 骨架不存在则自动高保真渲染
                skel_arr = render_std_skel(ch, script)
                if skel_arr is not None:
                    Image.fromarray(skel_arr, mode="L").save(dst_std, optimize=True)
                else:
                    return False, f"std_render_fail: {ch}_{script}"

        return True, "ok"
    except Exception as e:
        return False, str(e)

def main():
    print(f"=== 开始在服务器构建独立 Top 10 (23 黄金槽位) 数据集 ===")
    df = pd.read_csv(MANIFEST_CSV)
    total = len(df)
    print(f"待处理样本总数: {total:,} 张")

    os.makedirs(os.path.join(ROOT, OUT_IMG_DIR), exist_ok=True)
    os.makedirs(os.path.join(ROOT, OUT_STD_DIR), exist_ok=True)

    tasks = []
    out_rows = []

    for idx, r in df.iterrows():
        new_img_rel = f"{OUT_IMG_DIR}/{idx:06d}.png"
        new_std_rel = f"{OUT_STD_DIR}/{idx:06d}.png"

        src_img_full = os.path.join(ROOT, r["image_path"])
        dst_img_full = os.path.join(ROOT, new_img_rel)

        src_std_full = os.path.join(ROOT, r["std_path"]) if pd.notna(r["std_path"]) else None
        dst_std_full = os.path.join(ROOT, new_std_rel)

        tasks.append((
            src_img_full,
            dst_img_full,
            src_std_full,
            dst_std_full,
            r["character"],
            r["script"]
        ))

        row_dict = r.to_dict()
        row_dict["image_path"] = new_img_rel
        row_dict["std_path"] = new_std_rel
        out_rows.append(row_dict)

    success_cnt = 0
    fail_reasons = []

    with ProcessPoolExecutor(max_workers=16) as executor:
        for ok, msg in tqdm(executor.map(process_copy_task, tasks), total=total, desc="Copying & Rendering"):
            if ok:
                success_cnt += 1
            else:
                fail_reasons.append(msg)

    print("\n" + "=" * 65)
    print(f"🎉 Top 10 数据集物理落盘完成！成功: {success_cnt}/{total} 张 ({success_cnt/total*100:.2f}%)")
    if fail_reasons:
        print(f"失败样本数: {len(fail_reasons)} 张，样例原因: {fail_reasons[:5]}")

    # 导出最终对齐的 CSV
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"全量独立 CSV 已保存至: {OUT_CSV} ({len(df_out)} 行)")

    img_cnt = len(os.listdir(os.path.join(ROOT, OUT_IMG_DIR)))
    std_cnt = len(os.listdir(os.path.join(ROOT, OUT_STD_DIR)))
    print(f"实测落盘图像: {img_cnt:,} 张")
    print(f"实测落盘骨架: {std_cnt:,} 张")
    print("=" * 65)

if __name__ == "__main__":
    main()

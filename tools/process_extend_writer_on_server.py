#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/process_extend_writer_on_server.py — 在服务器全速多进程处理并生成 extend_writer 图像与骨架"""
import os
import sys
import re
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

CSV_PATH = "assets/train_extend_writer.csv"
OUT_IMG_DIR = "data/extend_writer/imgs"
OUT_STD_DIR = "data/extend_writer/std"
FONT_DIR = "tools/fonts"
CAND_FONT_DIR = "tools/fonts/candidate_fonts"
SIZE = 256
FONT_SIZE = 200
BOX_FRAC = 0.88
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

def process_single(item):
    src_raw, dst_img_p, dst_std_p, is_inv, ch, script, source = item
    try:
        # 路径转换
        if source == "mccd_salvage":
            # 将 Windows 路径映射到服务器路径
            # 形如 G:\GitHub\DiT\MCCD\MCCD\... 或 G:/GitHub/DiT/MCCD/...
            clean_p = src_raw.replace("\\", "/")
            if "MCCD/MCCD" in clean_p:
                sub_p = clean_p.split("MCCD/MCCD")[-1]
                src_p = os.path.join(ROOT, "MCCD/MCCD" + sub_p)
            else:
                src_p = clean_p
        else:
            src_p = src_raw

        if not os.path.exists(src_p):
            return False, f"not_found: {src_p}"

        # 1. 图像归一化
        im = Image.open(src_p).convert("L")
        arr = np.asarray(im)
        if is_inv:
            arr = 255 - arr

        m = arr < 128
        if not m.any():
            return False, "blank"
        
        ys, xs = np.where(m)
        crop = Image.fromarray(arr).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        
        t = int(SIZE * BOX_FRAC)
        s = t / max(crop.size)
        tw, th = max(int(crop.size[0] * s), 1), max(int(crop.size[1] * s), 1)
        crop = crop.resize((tw, th), Image.LANCZOS)
        
        cv = Image.new("L", (SIZE, SIZE), 255)
        cv.paste(crop, ((SIZE - tw) // 2, (SIZE - th) // 2))
        
        out_img = np.where(np.asarray(cv) < 128, 0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(dst_img_p), exist_ok=True)
        Image.fromarray(out_img, mode="L").save(dst_img_p, optimize=True)

        # 2. 标准骨架渲染
        os.makedirs(os.path.dirname(dst_std_p), exist_ok=True)
        if not os.path.exists(dst_std_p):
            skel_arr = render_std_skel(ch, script)
            if skel_arr is not None:
                Image.fromarray(skel_arr, mode="L").save(dst_std_p, optimize=True)

        return True, "ok"
    except Exception as e:
        return False, str(e)

def main():
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"=== 开始在服务器全速处理 extend_writer 集合: {total} 张 ===")

    tasks = []
    for _, r in df.iterrows():
        tasks.append((
            r["src_image_path"],
            os.path.join(ROOT, r["image_path"]),
            os.path.join(ROOT, r["std_path"]),
            r["is_inverted"],
            r["character"],
            r["script"],
            r["source"]
        ))

    os.makedirs(os.path.join(ROOT, OUT_IMG_DIR), exist_ok=True)
    os.makedirs(os.path.join(ROOT, OUT_STD_DIR), exist_ok=True)

    success_cnt = 0
    fail_reasons = []

    with ProcessPoolExecutor(max_workers=16) as executor:
        for ok, msg in tqdm(executor.map(process_single, tasks), total=total, desc="Processing on Server"):
            if ok:
                success_cnt += 1
            else:
                fail_reasons.append(msg)

    print("\n" + "=" * 65)
    print(f"🎉 服务器处理完成！成功生成: {success_cnt}/{total} 张 ({success_cnt/total*100:.2f}%)")
    if fail_reasons:
        print(f"失败/跳过样本数: {len(fail_reasons)} 张")
        print("样例原因:", fail_reasons[:5])

    # 校验最终落盘文件数量
    img_cnt = len(os.listdir(os.path.join(ROOT, OUT_IMG_DIR)))
    std_cnt = len(os.listdir(os.path.join(ROOT, OUT_STD_DIR)))
    print(f"实测落盘图像: {img_cnt} 张 (预期 {success_cnt})")
    print(f"实测落盘骨架: {std_cnt} 张 (预期 {success_cnt})")
    print("=" * 65)

if __name__ == "__main__":
    main()

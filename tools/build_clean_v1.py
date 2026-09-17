# -*- coding: utf-8 -*-
"""build_clean_v1.py — 用"干净原图 + 字库从头渲染的标准字"重建一份自包含数据集.

用户裁定 (2026-09-15):
  之前的资产(latent / g 条件)链路太长, 怀疑有污染。改为**从零重建、自包含**:
    - 图片: 清洗后的 40,606 张原图
    - 标准字: **用字库从头渲染**(不再复用任何旧 std skel 资产)
    - 图与标准字**放在同一个目录**, 便于核对
    - 然后对这些 PNG **重新 encode**

产物:
  data/clean_v1/imgs/{img_id}.png     40,606 张 (copy)
  data/clean_v1/std/{img_id}.png      同数量的标准字渲染图 (白底黑字)
  assets/train_clean_v1.csv           列: image_path, std_path, character, script, ...
  data/clean_v1/_preview/*.png        抽样对照 (图 | 标准字)
  data/clean_v1/_report.json          覆盖率 / 缺字

字体映射 (script -> 字体, 取自 tools/fonts/):
  楷 -> simkai.ttf     行 -> STXINGKA.TTF     隶 -> SIMLI.TTF
渲染: 复用 tools/build_std_glyph_latents.py 的 render_glyph 归一化口径
      (先 2x 大图渲染, 裁墨迹 bbox, 等比缩放到 box_frac*256, 居中)

用法: python tools/build_clean_v1.py [--apply]
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_CSV = "assets/train_base_clean.csv"
OUT_DIR = "data/clean_v1"
OUT_CSV = "assets/train_clean_v1.csv"
FONT_DIR = "tools/fonts"
# 书体 -> 字体**回退链**: 首选最贴书体的字体, 该字体缺字时依次回退。
# 依据: STXINGKA.TTF(行楷 4MB)/SIMLI.TTF(隶) 收录字少, 对异体字/繁体缺失率高;
#       simkai.ttf(11.8MB) 覆盖较广; simhei.ttf(9.7MB) 覆盖最广。
# g 条件是**字形结构**引导, 书体差异由 calligrapher embedding 承担,
# 故回退到楷体/黑体只损失少量风格, 远优于"无标准字"。
FONT_CHAIN = {
    "楷": ["simkai.ttf", "simhei.ttf"],
    "行": ["STXINGKA.TTF", "simkai.ttf", "simhei.ttf"],
    "隶": ["SIMLI.TTF", "simkai.ttf", "simhei.ttf"],
    "草": ["STXINGKA.TTF", "simkai.ttf", "simhei.ttf"],
}
SIZE = 256
BOX_FRAC = 0.88

_cache = {}


def get_font(fname, size):
    key = (fname, size)
    if key not in _cache:
        p = os.path.join(FONT_DIR, fname)
        _cache[key] = ImageFont.truetype(p, size) if os.path.exists(p) else None
    return _cache[key]


def _tofu_sig(fname, size=SIZE):
    """渲染私用区字符得到"缺字(豆腐块)"签名, 用于识别字体是否收录某字。"""
    font = get_font(fname, size)
    if font is None:
        return None
    img = Image.new("L", (size * 2, size * 2), 255)
    ImageDraw.Draw(img).text((size, size), "\ue000", font=font, fill=0, anchor="mm")
    a = np.asarray(img)
    ink = a < 250
    if not ink.any():
        return ("blank",)
    ys, xs = np.where(ink)
    return (int(ink.sum()), int(ys.max() - ys.min()), int(xs.max() - xs.min()))


def _raw_render(ch, fname, size=SIZE):
    font = get_font(fname, size)
    if font is None:
        return None, None
    img = Image.new("L", (size * 2, size * 2), 255)
    try:
        ImageDraw.Draw(img).text((size, size), ch, font=font, fill=0, anchor="mm")
    except Exception:
        return None, None
    a = np.asarray(img)
    ink = a < 250
    if not ink.any():
        return None, None
    ys, xs = np.where(ink)
    sig = (int(ink.sum()), int(ys.max() - ys.min()), int(xs.max() - xs.min()))
    return a, sig


def render_glyph(ch, fname, size=SIZE, box_frac=BOX_FRAC):
    """渲染单字 -> (size,size) uint8 白底黑字; 缺字/失败返回 None。"""
    a, sig = _raw_render(ch, fname, size)
    if a is None:
        return None
    if sig == _tofu_sig(fname, size):      # 与缺字模板一致 -> 该字体没收录此字
        return None
    ink = a < 250
    ys, xs = np.where(ink)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = Image.fromarray(a).crop((x0, y0, x1, y1))
    target = box_frac * size
    h, w = crop.size[1], crop.size[0]
    s = target / max(h, w)
    tw, th = max(int(w * s), 1), max(int(h * s), 1)
    crop = crop.resize((tw, th), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(crop, ((size - tw) // 2, (size - th) // 2))
    return np.asarray(canvas)


def _work(task):
    iid, src, ch, script = task
    for fname in FONT_CHAIN.get(script, FONT_CHAIN["楷"]):
        arr = render_glyph(ch, fname)
        if arr is not None:
            return (iid, arr, ch, script, fname)
    return (iid, None, ch, script, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
    print(f"[build] {SRC_CSV}: {len(rows)} 行")
    keys = sorted({(r.get("script", ""), r.get("character", "")) for r in rows})
    print(f"  唯一 (script,char) = {len(keys)}")
    print(f"  script 分布 = {dict(Counter(r.get('script','') for r in rows))}")
    print(f"  字体回退链: {FONT_CHAIN}")

    # ---- 渲染全部 (script,char) ----
    jobs = [(r.get("script", ""), r.get("character", "")) for r in rows]
    uniq_jobs = sorted(set(jobs))
    tasks = [(i, None, ch, sc) for i, (sc, ch) in enumerate(uniq_jobs)]
    print(f"\n[render] {len(tasks)} 个 (script,char) ...", flush=True)
    with mp.Pool(32) as pool:
        res = pool.map(_work, tasks, chunksize=32)

    key2arr, missing, used_font = {}, [], Counter()
    for _, arr, ch, sc, fn in res:
        if arr is None:
            missing.append((sc, ch))
        else:
            key2arr[(sc, ch)] = arr
            used_font[fn] += 1
    print(f"  渲染成功 {len(key2arr)}, 失败 {len(missing)}")
    print(f"  实际使用字体分布: {dict(used_font)}")
    if missing:
        print(f"  失败样例: {missing[:20]}")

    hit = sum(1 for k in jobs if k in key2arr)
    print(f"  样本级覆盖率: {hit}/{len(rows)} ({100*hit/len(rows):.2f}%)")

    if not a.apply:
        print("\n[DRY-RUN] 加 --apply 执行。")
        return

    # ---- 写新目录 ----
    for d in ("imgs", "std", "_preview"):
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)
    n_img = n_std = 0
    out_rows = []
    miss_rows = []
    for r in rows:
        p = r["image_path"]
        if not os.path.exists(p):
            continue
        iid = os.path.basename(p)[:-4]
        dst_img = os.path.join(OUT_DIR, "imgs", f"{iid}.png")
        if not os.path.exists(dst_img):
            shutil.copy(p, dst_img)
        n_img += 1
        key = (r.get("script", ""), r.get("character", ""))
        arr = key2arr.get(key)
        if arr is None:
            miss_rows.append(r)
            continue
        dst_std = os.path.join(OUT_DIR, "std", f"{iid}.png")
        Image.fromarray(arr).save(dst_std)
        n_std += 1
        rr = dict(r)
        rr["image_path"] = f"{OUT_DIR}/imgs/{iid}.png"
        rr["std_path"] = f"{OUT_DIR}/std/{iid}.png"
        out_rows.append(rr)

    fields = list(rows[0].keys()) + ["std_path"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rr in out_rows:
            w.writerow(rr)
    print(f"\n[apply] {OUT_DIR}/imgs   {n_img} 张")
    print(f"[apply] {OUT_DIR}/std    {n_std} 张")
    print(f"[apply] {OUT_CSV}        {len(out_rows)} 行 "
          f"(剔除无标准字的 {len(miss_rows)} 行)")

    # ---- preview 对照 ----
    step = max(1, len(out_rows) // 24)
    CELL = 160
    picks = out_rows[::step][:24]
    cols = 6
    rws = (len(picks) + cols - 1) // cols
    cv = Image.new("RGB", (cols * CELL * 2, max(rws, 1) * CELL), (255, 255, 255))
    for i, rr in enumerate(picks):
        try:
            im = Image.open(rr["image_path"]).convert("L").resize((CELL, CELL))
            st = Image.open(rr["std_path"]).convert("L").resize((CELL, CELL))
        except Exception:
            continue
        y, x = (i // cols) * CELL, (i % cols) * CELL * 2
        cv.paste(im.convert("RGB"), (x, y))
        cv.paste(st.convert("RGB"), (x + CELL, y))
    cv.save(os.path.join(OUT_DIR, "_preview", "pairs.png"))
    print(f"[apply] preview -> {OUT_DIR}/_preview/pairs.png (左=图, 右=标准字)")

    rep = {
        "src_csv": SRC_CSV, "rows_in": len(rows), "rows_out": len(out_rows),
        "uniq_script_char": len(uniq_jobs), "render_ok": len(key2arr),
        "render_missing": len(missing), "missing_sample": missing[:50],
        "font_chain": FONT_CHAIN, "used_font": dict(used_font),
        "box_frac": BOX_FRAC, "size": SIZE,
    }
    with open(os.path.join(OUT_DIR, "_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[apply] report -> {OUT_DIR}/_report.json")


if __name__ == "__main__":
    main()

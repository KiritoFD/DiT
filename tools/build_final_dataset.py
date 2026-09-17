# -*- coding: utf-8 -*-
"""build_final_dataset.py — 最终数据集: fame3 + tongji, 弃用 UniCalli.

背景 (2026-09-15 用户裁定):
  UniCalli 的字被切得乱七八糟 —— 根因是 crop_unicalli.py 直接用 data.csv 的
  column-level `location` 坐标裁剪, 而该坐标系与实际图不匹配(实测 4.91% bbox
  越界, 越界时 PIL 填黑, 加上部分原图为黑底拓片) -> 大量半个字/两字混切/
  黑块/灰块。修 bbox 成本远高于收益, 直接弃用该源。

  保留: final_imgs_fame_v8 (27,552) + calli_tongji_imgs (2,092) = 29,644
  这两源此前多轮清洗中存活率均 ~100%, 本身干净。

产物:
  assets/train_final.csv                  最终训练 csv
  data/final/imgs/{id}.png                (copy, 自包含)
  data/final/std/{id}.png                 (字库渲染标准字, 与 fig 同 id)
  data/final/_preview/pairs.png           抽样对照
用法: python tools/build_final_dataset.py [--apply]
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

SRC = "assets/train_base_noaug.csv"
OUT_DIR = "data/final"
OUT_CSV = "assets/train_final.csv"
KEEP_SRC = ("final_imgs_fame_v8", "calli_tongji_imgs")
FONT_DIR = "tools/fonts"
FONT_CHAIN = {
    "楷": ["simkai.ttf", "simhei.ttf"],
    "行": ["STXINGKA.TTF", "simkai.ttf", "simhei.ttf"],
    "隶": ["SIMLI.TTF", "simkai.ttf", "simhei.ttf"],
    "草": ["STXINGKA.TTF", "simkai.ttf", "simhei.ttf"],
}
SIZE, BOX_FRAC = 256, 0.88
_cache = {}
_tofu = {}


def get_font(fn, size):
    k = (fn, size)
    if k not in _cache:
        p = os.path.join(FONT_DIR, fn)
        _cache[k] = ImageFont.truetype(p, size) if os.path.exists(p) else None
    return _cache[k]


def tofu_sig(fn, size=SIZE):
    if fn in _tofu:
        return _tofu[fn]
    f = get_font(fn, size)
    if f is None:
        _tofu[fn] = None
        return None
    img = Image.new("L", (size * 2, size * 2), 255)
    ImageDraw.Draw(img).text((size, size), "\ue000", font=f, fill=0, anchor="mm")
    a = np.asarray(img)
    m = a < 250
    sig = ("blank",) if not m.any() else (
        int(m.sum()), int(np.where(m.any(1))[0].ptp()), int(np.where(m.any(0))[0].ptp()))
    _tofu[fn] = sig
    return sig


def render(ch, fn, size=SIZE, bf=BOX_FRAC):
    f = get_font(fn, size)
    if f is None:
        return None
    img = Image.new("L", (size * 2, size * 2), 255)
    try:
        ImageDraw.Draw(img).text((size, size), ch, font=f, fill=0, anchor="mm")
    except Exception:
        return None
    a = np.asarray(img)
    m = a < 250
    if not m.any():
        return None
    sig = (int(m.sum()), int(np.where(m.any(1))[0].ptp()), int(np.where(m.any(0))[0].ptp()))
    if sig == tofu_sig(fn, size):
        return None
    ys, xs = np.where(m)
    crop = Image.fromarray(a).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    t = bf * size
    s = t / max(crop.size)
    tw, th = max(int(crop.size[0] * s), 1), max(int(crop.size[1] * s), 1)
    crop = crop.resize((tw, th), Image.LANCZOS)
    cv = Image.new("L", (size, size), 255)
    cv.paste(crop, ((size - tw) // 2, (size - th) // 2))
    return np.asarray(cv)


def _w(task):
    sc, ch = task
    for fn in FONT_CHAIN.get(sc, FONT_CHAIN["楷"]):
        a = render(ch, fn)
        if a is not None:
            return (sc, ch, a, fn)
    return (sc, ch, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8"))
            if any(s in r["image_path"] for s in KEEP_SRC)]
    print(f"[final] 源 {SRC} -> 保留源 {KEEP_SRC}")
    print(f"  行数 {len(rows)}")
    print(f"  分层 {dict(Counter(r['image_path'].split('/')[2] for r in rows))}")

    keys = sorted({(r.get("script", ""), r.get("character", "")) for r in rows})
    print(f"  唯一 (script,char) {len(keys)}", flush=True)
    with mp.Pool(32) as pool:
        res = pool.map(_w, keys, chunksize=32)
    k2a = {(sc, ch): arr for sc, ch, arr, _ in res if arr is not None}
    miss = [(sc, ch) for sc, ch, arr, _ in res if arr is None]
    used = Counter(fn for *_, fn in res if fn)
    print(f"  渲染成功 {len(k2a)}, 失败 {len(miss)}; 用字 {dict(used)}")

    if not a.apply:
        print("[DRY-RUN] 加 --apply 执行。")
        return

    for d in ("imgs", "std", "_preview"):
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)
    out, n_i, n_s = [], 0, 0
    for r in rows:
        p = r["image_path"]
        iid = os.path.basename(p)[:-4]
        arr = k2a.get((r.get("script", ""), r.get("character", "")))
        if arr is None:
            continue
        di = os.path.join(OUT_DIR, "imgs", f"{iid}.png")
        ds = os.path.join(OUT_DIR, "std", f"{iid}.png")
        if not os.path.exists(di):
            shutil.copy(p, di)
            n_i += 1
        Image.fromarray(arr).save(ds)
        n_s += 1
        rr = dict(r)
        rr["image_path"] = f"{OUT_DIR}/imgs/{iid}.png"
        rr["std_path"] = f"{OUT_DIR}/std/{iid}.png"
        out.append(rr)

    fields = list(rows[0].keys()) + ["std_path"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rr in out:
            w.writerow(rr)
    print(f"[apply] imgs copied {n_i}, std rendered {n_s}")
    print(f"[apply] {OUT_CSV}: {len(out)} 行")

    step = max(1, len(out) // 24)
    CELL, COLS = 150, 6
    picks = out[::step][:24]
    rws = max((len(picks) + COLS - 1) // COLS, 1)
    cv = Image.new("RGB", (COLS * CELL * 2, rws * CELL), (255, 255, 255))
    for i, rr in enumerate(picks):
        try:
            im = Image.open(rr["image_path"]).convert("L").resize((CELL, CELL))
            st = Image.open(rr["std_path"]).convert("L").resize((CELL, CELL))
        except Exception:
            continue
        y, x = (i // COLS) * CELL, (i % COLS) * CELL * 2
        cv.paste(im.convert("RGB"), (x, y))
        cv.paste(st.convert("RGB"), (x + CELL, y))
    cv.save(os.path.join(OUT_DIR, "_preview", "pairs.png"))
    print(f"[apply] preview -> {OUT_DIR}/_preview/pairs.png")
    with open(os.path.join(OUT_DIR, "_report.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": len(out), "render_miss": len(miss),
                   "used_font": dict(used), "keep_src": list(KEEP_SRC)},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

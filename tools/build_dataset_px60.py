# -*- coding: utf-8 -*-
"""build_dataset_px60.py — 构建最终数据集 fame-kxl-tj-px60.

用户裁定 (2026-09-15):
  * **drop UniCalli** —— 其字被切得乱七八糟（crop_unicalli.py 直接用 column-level
    location 坐标裁剪, 与实际图不匹配, 实测 4.91% bbox 越界, 越界填黑; 另有黑底拓片）。
    修 bbox 成本远高于收益。
  * **保留 fame3 + tongji(2,092)**（沿用旧白名单, 不用多出的 900 张）。
  * **唯一判据**: 黑色部分(含外部包络)的最粗宽度 > 60px -> 剔除。
    实测定界: >60px 组是"糊成块/残字/超粗"; 55~60px 组字迹完整只是偏粗(55px 会误伤),
    50~55px 是完全正常的工整书法 -> 60px 是合理阈值。
  * 标准字: **用字库从头渲染**（不复用任何旧 skel 资产）, 与图同目录同 id。
  * 命名: `fame-kxl-tj-px60`

产物:
  data/fame-kxl-tj-px60/imgs/{id}.png     图 (copy)
  data/fame-kxl-tj-px60/std/{id}.png      字库渲染的标准字 (白底黑字)
  assets/train_fame-kxl-tj-px60.csv       训练 csv
  data/fame-kxl-tj-px60/_preview/*.png    抽样对照 (kept / dropped)
  data/fame-kxl-tj-px60/_report.json      统计

用法: python tools/build_dataset_px60.py [--apply]
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
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = "assets/train_base_noaug.csv"
KEEP_SRC = ("final_imgs_fame_v8", "calli_tongji_imgs")
NAME = "fame-kxl-tj-px60"
OUT_DIR = f"data/{NAME}"
OUT_CSV = f"assets/train_{NAME}.csv"
SCAN_CACHE = f"/tmp/{NAME}_wmax.npz"
TH = 60.0

FONT_DIR = "tools/fonts"
FONT_CHAIN = {
    "楷": ["simkai.ttf", "simhei.ttf"],
    "行": ["STXINGKA.TTF", "simkai.ttf", "simhei.ttf"],
    "隶": ["SIMLI.ttf", "SIMLI.TTF", "simkai.ttf", "simhei.ttf"],
    "草": ["STXINGKA.TTF", "simkai.ttf", "simhei.ttf"],
}
SIZE, BOX_FRAC = 256, 0.88
_fc, _tofu = {}, {}


def wmax(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return -1.0
    m = a < 128
    return float(distance_transform_edt(m).max()) * 2.0 if m.any() else 0.0


def get_font(fn, size=SIZE):
    if (fn, size) not in _fc:
        p = os.path.join(FONT_DIR, fn)
        _fc[(fn, size)] = ImageFont.truetype(p, size) if os.path.exists(p) else None
    return _fc[(fn, size)]


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
    _tofu[fn] = ("blank",) if not m.any() else (
        int(m.sum()), int(np.where(m.any(1))[0].ptp()), int(np.where(m.any(0))[0].ptp()))
    return _tofu[fn]


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


def montage(items, out, cols=8, cell=110):
    n = min(len(items), cols * 6)
    rows = max((n + cols - 1) // cols, 1)
    cv = Image.new("RGB", (cols * cell, rows * cell), (170, 30, 30))
    for i, p in enumerate(items[:n]):
        try:
            cv.paste(Image.open(p).convert("L").resize((cell, cell)).convert("RGB"),
                     ((i % cols) * cell, (i // cols) * cell))
        except Exception:
            pass
    cv.save(out)
    print(f"    montage -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8"))
            if any(s in r["image_path"] for s in KEEP_SRC)]
    print(f"[{NAME}] 源 {SRC}, 保留源 {KEEP_SRC}")
    print(f"  初始 {len(rows)} 行  分层 {dict(Counter(r['image_path'].split('/')[2] for r in rows))}")
    ids = [os.path.basename(r["image_path"])[:-4] for r in rows]
    print(f"  唯一 id {len(set(ids))} / {len(ids)}"
          + ("  ⚠ 有重复!" if len(set(ids)) != len(ids) else "  ✓"))

    # ---- 60px 扫描 (保序!) ----
    if os.path.exists(SCAN_CACHE):
        d = np.load(SCAN_CACHE)
        w = d["w"]
        assert len(w) == len(rows), "cache mismatch"
        print(f"  [scan] 复用缓存 {SCAN_CACHE}")
    else:
        print(f"  [scan] 计算 w_max ...", flush=True)
        with mp.Pool(40) as pool:
            w = np.array(pool.map(wmax, [r["image_path"] for r in rows], chunksize=128),
                         dtype=np.float32)
        np.savez(SCAN_CACHE, w=w)
    kill = w > TH
    print(f"\n  === 判据 w_max > {TH:.0f}px ===")
    print(f"  剔除 {int(kill.sum())} / {len(rows)} ({100*kill.mean():.2f}%)")
    for s in KEEP_SRC:
        ix = [i for i, r in enumerate(rows) if s in r["image_path"]]
        k = int(kill[ix].sum())
        print(f"    {s:22s} {len(ix):6d} -> 剔 {k:5d}  剩 {len(ix)-k}")
    keep_idx = [i for i in range(len(rows)) if not kill[i]]
    print(f"\n  ★ 最终 {len(keep_idx)} 张")
    print(f"  书家 {len({rows[i].get('calligrapher','') for i in keep_idx})}, "
          f"字符 {len({rows[i].get('character','') for i in keep_idx})}, "
          f"(script,char) {len({(rows[i].get('script',''), rows[i].get('character','')) for i in keep_idx})}")
    print(f"  书体 {dict(Counter(rows[i].get('script','') for i in keep_idx))}")

    # ---- 标准字渲染 ----
    keys = sorted({(rows[i].get("script", ""), rows[i].get("character", "")) for i in keep_idx})
    print(f"\n  [std] 渲染 {len(keys)} 个 (script,char) ...", flush=True)
    with mp.Pool(32) as pool:
        res = pool.map(_w, keys, chunksize=32)
    k2a = {(sc, ch): arr for sc, ch, arr, _ in res if arr is not None}
    miss = [(sc, ch) for sc, ch, arr, _ in res if arr is None]
    used = Counter(fn for *_, fn in res if fn)
    print(f"    成功 {len(k2a)}, 失败 {len(miss)}  用字 {dict(used)}")
    ready = [i for i in keep_idx
             if (rows[i].get("script", ""), rows[i].get("character", "")) in k2a]
    print(f"    有标准字的样本 {len(ready)}")

    os.makedirs(f"{OUT_DIR}/_preview", exist_ok=True)
    montage([rows[i]["image_path"] for i in keep_idx], f"{OUT_DIR}/_preview/kept.png")
    montage([rows[i]["image_path"] for i in range(len(rows)) if kill[i]],
            f"{OUT_DIR}/_preview/dropped.png")

    if not a.apply:
        print("\n[DRY-RUN] 加 --apply 执行。")
        return

    os.makedirs(f"{OUT_DIR}/imgs", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/std", exist_ok=True)
    out, n_copy = [], 0
    for i in ready:
        r = rows[i]
        p = r["image_path"]
        iid = os.path.basename(p)[:-4]
        di = f"{OUT_DIR}/imgs/{iid}.png"
        if not os.path.exists(di):
            shutil.copy(p, di)
            n_copy += 1
        Image.fromarray(k2a[(r.get("script", ""), r.get("character", ""))]).save(
            f"{OUT_DIR}/std/{iid}.png")
        rr = dict(r)
        rr["image_path"] = di
        rr["std_path"] = f"{OUT_DIR}/std/{iid}.png"
        rr["source"] = p.split("/")[2]
        out.append(rr)

    fields = list(rows[0].keys()) + ["std_path", "source"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wtr.writeheader()
        for rr in out:
            wtr.writerow(rr)
    print(f"\n[apply] imgs copy {n_copy}, std {len(out)}")
    print(f"[apply] {OUT_CSV}: {len(out)} 行")
    with open(f"{OUT_DIR}/_report.json", "w", encoding="utf-8") as f:
        json.dump({"name": NAME, "rows": len(out), "th_px": TH,
                   "killed": int(kill.sum()), "render_miss": len(miss),
                   "used_font": dict(used), "keep_src": list(KEEP_SRC),
                   "calligrapher": len({r.get('calligrapher', '') for r in out}),
                   "character": len({r.get('character', '') for r in out})},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

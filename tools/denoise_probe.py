# -*- coding: utf-8 -*-
"""denoise_probe.py — 去噪算法迭代探测 (本地 CPU, 原生分辨率 + 白底 letterbox).

用法: python tools/denoise_probe.py [round]
  读取 12 worst + 8 random fame 原图 (本地 MCCD), 处理, 输出
  _diag/probe/round{N}/<idx>_<char>.png  [raw-stretch | vN-clean-letterbox]
"""
import os, sys, csv, json, random
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image, ImageDraw
from scipy import ndimage

ROUND = sys.argv[1] if len(sys.argv) > 1 else "1"

# ---- v3 算法参数 (相对量, 随原生分辨率缩放) ----
BAND_FRAC = 0.05        # 边带 = 5% of min(H,W)
SMALL_FRAC = 0.001      # 小件面积 < 0.1% of 图像面积 → 赃物
THIN_RATIO = 0.01       # area/max(bw,bh) < 1% of min(H,W) → 细线
BORDER_BIG_FRAC = 0.3   # 非主体贴边大块 < 0.3×主体面积 → 删
MAIN_KEEP = True

def polarity_norm(gray):
    ink = gray < 128
    if ink.mean() > 0.5:
        gray = 255 - gray
    return gray

def clean_v3(gray_u8):
    """原生分辨率清洗, 返回 (clean_gray_u8, metrics)."""
    g = polarity_norm(gray_u8)
    ink = g < 128
    H, W = ink.shape
    m = min(H, W)
    band = int(BAND_FRAC * m)
    yy, xx = np.mgrid[0:H, 0:W]
    bord = (yy < band) | (yy >= H - band) | (xx < band) | (xx >= W - band)
    ink = ink & ~bord                                  # 抹边带
    lab, n = ndimage.label(ink)
    if n == 0:
        return np.full_like(g, 255), {'empty': 1}
    areas = ndimage.sum(ink, lab, range(1, n + 1))
    main = int(np.argmax(areas)) + 1
    main_area = float(areas[main - 1])
    small_thr = SMALL_FRAC * H * W
    thin_thr = THIN_RATIO * m
    keep = (lab == main)
    small = border_big = 0.0
    for ci in range(1, n + 1):
        if ci == main:
            continue
        comp = lab == ci
        area = float(areas[ci - 1])
        ys, xs = np.where(comp)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if area < small_thr:
            small += area
            continue
        if area / max(bw, bh, 1) < thin_thr and area < BORDER_BIG_FRAC * main_area:
            border_big += area
            continue
        if area < BORDER_BIG_FRAC * main_area and area < 0.2 * H * W:
            border_big += area
            continue
        keep |= comp
    small /= max(float(ink.sum()), 1)
    border_big /= max(float(ink.sum()), 1)
    out = np.where(keep, 0, 255).astype("uint8")
    return out, {'n_comp': int(n), 'small_frac': round(small, 4),
                 'border_big_frac': round(border_big, 4),
                 'main_area_frac': round(main_area / (H * W), 4)}

def letterbox(clean_u8, orig_size, out=256):
    """清洗结果按比例缩放, 白底居中."""
    im = Image.fromarray(clean_u8)
    w, h = orig_size
    s = out / max(w, h)
    nw, nh = max(int(w * s), 1), max(int(h * s), 1)
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (out, out), 255)
    canvas.paste(im, ((out - nw) // 2, (out - nh) // 2))
    return canvas

def main():
    # fame 原图 → 本地 MCCD 路径
    map_rows = list(csv.DictReader(open("5script/mccd_image_map.csv", encoding="utf-8")))
    by_id = {}
    for r in map_rows:
        base = os.path.basename(r["filepath"])
        m = None
        import re
        mm = re.search(r"-(\d+)\.png$", base)
        if mm:
            by_id[int(mm.group(1))] = r["filepath"]
    tr = list(csv.DictReader(open("5script/train_fame.csv", encoding="utf-8")))
    worst = json.load(open("_diag/probe/worst_ids.json", encoding="utf-8")) \
        if os.path.exists("_diag/probe/worst_ids.json") else []
    # 探测集: worst(来自 fame 噪声扫描) + 随机
    random.seed(7)
    fame_orig = []
    for r in tr:
        iid = int(r["image_path"].split("/")[-1][:-4])
        if iid in by_id:
            fame_orig.append((iid, r, by_id[iid]))
    picks = []
    seen = set()
    for iid, r, fp in fame_orig:
        if iid in worst and iid not in seen:
            picks.append((iid, r, fp)); seen.add(iid)
    rest = [(i, r, fp) for i, r, fp in fame_orig if i not in seen]
    for x in random.sample(rest, min(8, len(rest))):
        picks.append(x)
    picks = picks[:20]
    outdir = f"_diag/probe/round{ROUND}"
    os.makedirs(outdir, exist_ok=True)
    for j, (iid, r, fp) in enumerate(picks):
        im = Image.open(fp).convert("L")
        raw_stretch = im.resize((256, 256))
        clean, metrics = clean_v3(np.asarray(im))
        lb = letterbox(clean, im.size)
        H = 256
        canvas = Image.new("RGB", (H * 2 + 30, H + 46), "white")
        d = ImageDraw.Draw(canvas)
        canvas.paste(raw_stretch, (0, 36))
        canvas.paste(lb.convert("RGB"), (H + 30, 36))
        d.text((5, 6), "OLD: stretch-256 (no clean)", fill="#CC0000")
        d.text((H + 35, 6), f"v{ROUND}: native clean + letterbox", fill="#227722")
        d.text((5, H + 40), f"{r['script']} {r['character']} ({r['calligrapher']}) {im.size} {metrics}",
               fill="black")
        canvas.save(os.path.join(outdir, f"{j:02d}_{r['character']}.png"))
    json.dump([{"id": i, "char": r["character"], "path": fp}
               for i, r, fp in picks],
              open(os.path.join(outdir, "probe_list.json"), "w"), ensure_ascii=False, indent=1)
    print(f"round{ROUND}: {len(picks)} probes -> {outdir}", flush=True)

if __name__ == "__main__":
    main()

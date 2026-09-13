# -*- coding: utf-8 -*-
"""denoise_v7.py — v7 pipeline: 极性(黑组件>55%→反色) → 切边条 → 外框白化 → 去小件. 无缩放."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, '/root/Workspace/xy/DiT'); os.chdir('/root/Workspace/xy/DiT')
sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image, ImageDraw
from scipy import ndimage

SMALL_AREA = 80

def v7(a):
    # ---- 0) 极性: 最大黑组件 >55% → 黑是背景 → 反色 ----
    ink = a < 128
    lab, n = ndimage.label(ink)
    inverted = False
    if n:
        areas = ndimage.sum(ink, lab, range(1, n+1))
        if float(areas.max()) / ink.size > 0.55:
            a = 255 - a
            inverted = True
    H, W = a.shape
    # ---- 1) 切边缘黑条: 从四边连续向内, 行/列黑占比 >50% → 白化 (安全上限 25%) ----
    bars = 0
    r = 0
    while r < H // 4 and (a[r] < 128).mean() > 0.5:
        a[r] = 255; r += 1; bars += 1
    r = H - 1
    while r > 3 * H // 4 and (a[r] < 128).mean() > 0.5:
        a[r] = 255; r -= 1; bars += 1
    c = 0
    while c < W // 4 and (a[:, c] < 128).mean() > 0.5:
        a[:, c] = 255; c += 1; bars += 1
    c = W - 1
    while c > 3 * W // 4 and (a[:, c] < 128).mean() > 0.5:
        a[:, c] = 255; c -= 1; bars += 1
    # ---- 2) 外框 3px 白化 (细框线残留) ----
    a[:3] = 255; a[H-3:] = 255; a[:, :3] = 255; a[:, W-3:] = 255
    # ---- 3) 去小件: <80px 连通域删除, 保最大 + ≥80px 其他 ----
    ink = a < 128
    lab, n = ndimage.label(ink)
    if n == 0:
        return np.full_like(a, 255), {'inverted': inverted, 'bars': bars, 'empty': True}
    areas = ndimage.sum(ink, lab, range(1, n+1))
    main = int(np.argmax(areas)) + 1
    keep = lab == main
    removed = 0.0
    for ci in range(1, n+1):
        if ci == main:
            continue
        comp = lab == ci
        area = float(areas[ci-1])
        if area < SMALL_AREA:
            removed += area
            continue
        keep |= comp
    out = np.where(keep, 0, 255).astype('uint8')
    return out, {'inverted': inverted, 'bars': bars,
                 'removed_frac': round(removed / (H*W), 4),
                 'main_frac': round(float(areas[main-1])/(H*W), 4)}

def main():
    ids = json.load(open('_diag/probe/worst_ids.json', encoding='utf-8'))
    recs = {}
    for l in open('/tmp/fame_full_scan.jsonl', encoding='utf-8'):
        r = json.loads(l)
        recs[os.path.basename(r.get('path', 'x'))] = r
    os.makedirs('_diag/probe/v7', exist_ok=True)
    ok = 0
    for iid in ids:
        p = f'data/imgs/final_imgs_256/{iid}.png'
        if not os.path.exists(p):
            continue
        a = np.array(Image.open(p).convert("L"))
        out, m = v7(a)
        H = 256
        canvas = Image.new('RGB', (H*2+30, H+46), 'white')
        d = ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(a), (0, 36))
        canvas.paste(Image.fromarray(out), (H+30, 36))
        d.text((5, 6), 'raw (current)', fill='black')
        d.text((H+35, 6), 'v7 cleaned', fill='#227722')
        d.text((5, H+40), f'{iid} {json.dumps(m)}', fill='black')
        canvas.save(f'_diag/probe/v7/{ok:02d}_{iid}.png')
        ok += 1
    print('v7 probe:', ok, 'images', flush=True)

if __name__ == '__main__':
    main()

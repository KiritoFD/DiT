# -*- coding: utf-8 -*-
"""scan_noise2.py — v2 cleaner (极性归一/抹边带/保主体) + 前后对照图."""
import os, sys, csv, json, random
import numpy as np
sys.path.insert(0, '/root/Workspace/xy/DiT'); os.chdir('/root/Workspace/xy/DiT')
sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image, ImageDraw
from scipy import ndimage
import multiprocessing as mp

SMALL_AREA = 80
BORDER_BAND = 14

def clean_v2(arr_u8):
    # 输入约定: 白底黑字 (极性已在数据层归一, 见 flip_all_fame.py)
    ink = arr_u8 < 128
    H, W = ink.shape
    yy, xx = np.mgrid[0:H, 0:W]
    band = (yy < BORDER_BAND) | (yy >= H - BORDER_BAND) | (xx < BORDER_BAND) | (xx >= W - BORDER_BAND)
    ink = ink & ~band             # 1) 抹边带: 切断贴边线/贴边块的边缘连接
    lab, n = ndimage.label(ink)
    if n == 0:
        return np.full_like(arr_u8, 255), {'empty': 1}
    areas = ndimage.sum(ink, lab, range(1, n + 1))
    main = int(np.argmax(areas)) + 1
    main_area = float(areas[main - 1])
    keep = (lab == main)
    small_frac = border_big = 0.0
    total = float(ink.sum())
    for ci in range(1, n + 1):
        if ci == main:
            continue
        comp = lab == ci
        area = float(areas[ci - 1])
        ys, xs = np.where(comp)
        bw = xs.max() - xs.min() + 1; bh = ys.max() - ys.min() + 1
        thin = area / max(bw, bh, 1) < 2.5          # 细线 (贴边残留/装订线)
        if area < SMALL_AREA:
            small_frac += area
            continue                                 # 赃物 → 删
        if thin and area < 0.5 * main_area:
            continue                                 # 贴边细线 → 删
        if area < 0.3 * main_area and (ys.min() < 2*BORDER_BAND or ys.max() >= H-2*BORDER_BAND
                                       or xs.min() < 2*BORDER_BAND or xs.max() >= W-2*BORDER_BAND):
            border_big += area
            continue                                 # 贴边大块 (非主体) → 删
        keep |= comp                                 # 其他保留 (如印章/款识)
    small_frac /= max(total, 1)
    border_big /= max(total, 1)
    out = np.where(keep, 0, 255).astype('uint8')
    return out, {'ink_frac': round(float(ink.sum())/(H*W), 4),
                 'small_frac': round(small_frac, 4),
                 'border_big_frac': round(border_big, 4)}

def _work(item):
    k, r = item
    a = np.asarray(Image.open(r['image_path']).convert('L'))
    if a.shape != (256, 256):
        a = np.asarray(Image.open(r['image_path']).convert('L').resize((256, 256)))
    # v1 指标 (极性归一前) 用于排序找 worst
    ink = a < 128
    if ink.mean() > 0.5:
        ink = ~ink
    lab, n = ndimage.label(ink)
    H, W = ink.shape
    yy, xx = np.mgrid[0:H, 0:W]
    band = (yy < 14) | (yy >= H-14) | (xx < 14) | (xx >= W-14)
    areas = ndimage.sum(ink, lab, range(1, n+1)) if n else []
    small = sum(a for a in areas if a < 80) if n else 0
    border = 0.0
    main_a = max(areas) if n else 1
    for ci in range(1, n+1):
        if ci == int(np.argmax(areas)) + 1:
            continue
        comp = lab == ci
        area = float(areas[ci-1])
        if (comp & band).sum() / max(area, 1) > 0.35 and area > 300:
            border += area
    tot = float(ink.sum())
    pre = {'small_frac': round(small/max(tot,1), 4), 'border_frac': round(border/max(tot,1), 4),
           'inverted': bool((a < 128).mean() > 0.5)}
    clean, post = clean_v2(a)
    score = pre['small_frac'] + pre['border_frac']
    return (k, r, a, clean, pre, post, score)

def main():
    rows = list(csv.DictReader(open('5script/train_fame.csv', encoding='utf-8')))
    random.seed(0)
    sample = random.sample(rows, 4000)
    os.makedirs('/tmp/noise2', exist_ok=True)
    stats_f = open('/tmp/noise2/stats.jsonl', 'w', encoding='utf-8')
    scored = []
    print('scanning v2...', flush=True)
    with mp.Pool(48) as pool:
        for n, (k, r, a, clean, pre, post, score) in enumerate(pool.imap_unordered(
                _work, list(enumerate(sample)), chunksize=8)):
            stats_f.write(json.dumps({**pre, **{'post': post}, 'char': r['character'],
                                      'path': r['image_path'], 'score': round(score, 4)}) + '\n')
            scored.append((score, r, a, clean, pre))
            if (n + 1) % 1000 == 0:
                print(f'{n+1}/{len(sample)}', flush=True)
    stats_f.close()
    scored.sort(key=lambda t: -t[0])
    for j, (score, r, a, clean, pre) in enumerate(scored[:12]):
        H = 256
        canvas = Image.new('RGB', (H*2+30, H+30), 'white')
        d = ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(a), (0, 30))
        canvas.paste(Image.fromarray(clean), (H+30, 30))
        d.text((5, H+8), f"{r['script']} {r['character']} ({r['calligrapher']}) inv={pre['inverted']} score={score:.3f}", fill='black')
        d.text((10, 8), 'raw', fill='black'); d.text((H+40, 8), 'cleaned v2', fill='black')
        canvas.save(f'/tmp/noise2/worst_{j:02d}_{r["character"]}.png')
    print('DONE', flush=True)

if __name__ == '__main__':
    main()

"""P1: pixel-level verification panels for the writer-by-char interaction.

For the writer with the strongest interaction signature, selects characters
with the largest and smallest interaction residual norms and composes, per
character: the STD render plus every writer's real training image (so the
human eye can judge whether large-residual characters correspond to visibly
idiosyncratic writer treatments — brush behavior, not noise).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag', default='assets/u100_pretrain_20261024/exp_a2_variance.json')
    parser.add_argument('--images-csv', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--writers', type=int, default=6)
    parser.add_argument('--per-side', type=int, default=3)
    parser.add_argument('--out', default='assets/posters/u100_pipeline_20261024/p1_pixel_panels.png')
    args = parser.parse_args()
    W = args.writers
    rows = [r for r in csv.DictReader(open(args.images_csv, encoding='utf-8-sig')) if r['generic'] == '0']
    TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}
    paths = defaultdict(list)
    for r in rows:
        paths[(TOP10_RAW[int(r['calligrapher_id'])], ord(r['character']))].append(r['image_path'])
    names = {}
    for r in rows: names[TOP10_RAW[int(r['calligrapher_id'])]] = r['calligrapher']

    std_rows = {}
    for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig')):
        std_rows[(int(r['script_id']), r['character'])] = r['path']

    # interaction norms per (writer, char): reuse the balanced panel computation inline
    import torch
    d = np.load('assets/u100_pretrain_20261024/style_bank_top10.npz')
    feats, writers, chars = d['features'], d['writers'], d['chars']
    wc = defaultdict(list)
    for i in range(len(chars)):
        if int(writers[i]) < W: wc[(int(writers[i]), int(chars[i]))].append(feats[i])
    wc_mean = {k: np.mean(v, 0) for k, v in wc.items() if len(v) >= 2}
    panel = sorted(c for c in {c for (w, c) in wc_mean}
                   if all((w, c) in wc_mean for w in range(W)))
    F = np.stack([np.stack([wc_mean[(w, c)] for w in range(W)]) for c in panel])
    m = F.mean(1); g = F - m[:, None]; a = g.mean(0); I = g - a[None]
    inorm = np.linalg.norm(I, axis=2)          # [C,W]
    win = int(np.argmax(inorm.mean(0)))
    order = np.argsort(-inorm[:, win])
    picked = [('large', panel[order[k]]) for k in range(args.per_side)] + \
             [('small', panel[order[-k - 1]]) for k in range(args.per_side)]
    print('writer', win, names.get(win, win), 'picked', [(t, chr(c)) for t, c in picked])

    def F(sz):
        for name in ('msyh.ttc', 'arial.ttf'):
            try: return ImageFont.truetype('C:/Windows/Fonts/' + name, sz)
            except Exception:
                try: return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', sz)
                except Exception: return ImageFont.load_default()
    CELL = 160; PAD = 6; LBL = 40
    pages = []
    for kind, c in picked:
        cells = []
        for w in range(W):
            ps = paths.get((win if False else w, c), [])
            img = Image.open(ps[0]).convert('L').resize((CELL, CELL)) if ps else Image.new('L', (CELL, CELL), 255)
            cells.append((img, names.get(w, str(w))))
        stdp = None
        for (sid, ch), p in std_rows.items():
            if ch == chr(c): stdp = p; break
        std = Image.open(Path(args.std_root) / stdp).convert('L').resize((CELL, CELL)) if stdp else Image.new('L', (CELL, CELL), 255)
        Wt = PAD + (CELL + PAD) * (W + 1)
        page = Image.new('RGB', (Wt, LBL + CELL + LBL), (16, 16, 20))
        dr = ImageDraw.Draw(page)
        dr.text((8, 4), f'{kind} residual  char {chr(c)}  (writer {win}={names.get(win,win)})', font=F(26), fill=(235, 235, 150))
        page.paste(std, (PAD, LBL))
        dr.text((PAD, LBL + CELL + 2), 'STD', font=F(20), fill=(150, 225, 255))
        for w, (im, nm) in enumerate(cells):
            x = PAD + (CELL + PAD) * (w + 1)
            page.paste(im, (x, LBL))
            mark = ' <' if w == win else ''
            dr.text((x, LBL + CELL + 2), f'{nm}{mark}', font=F(20), fill=(150, 225, 255) if w == win else (170, 170, 170))
        pages.append(page)
    total = Image.new('RGB', (pages[0].width, sum(p.height for p in pages)), (16, 16, 20))
    y = 0
    for p in pages: total.paste(p, (0, y)); y += p.height
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    total.save(args.out)
    print('P1_DONE', args.out, total.size)


if __name__ == '__main__':
    main()

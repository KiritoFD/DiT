"""Fixed-character review sheet: GT + every arm side by side, same rows.

Two sheets, both deterministic:
  spread  -- every 5th row of val100, so all writers are represented
  delta   -- the 20 rows where the cross-combination gained the most LPIPS vs K,
             i.e. the places the arm claims to have fixed

The point is human inspection of actual glyphs, which the panel means cannot show.
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CELL, GAP, PAD, LABEL_W = 148, 5, 16, 108
HEADER_H, FOOTER_H = 196, 58
PANEL = ROOT / 'assets/u100_pretrain_20261024/validation_val100.csv'
ARMS = [
    ('__gt__', 'GT', '真迹'),
    ('assets/f_20261101/eval/K_A', 'K', '43,429 基线'),
    ('assets/d_20261101/eval/D1_2500', 'D1', 'decoder-only'),
    ('assets/d_20261101/eval/D2_3000', 'D2', 'U-Net+dec'),
    ('assets/d_20261101/eval/D3_11000', 'D3', 'hard-char FM'),
    ('assets/d_20261101/eval/D3_11000_plus_D1dec', 'D3+D1dec', '交叉组合'),
]
WRITER_COLORS = {'赵孟頫': (37, 99, 235), '文徵明': (13, 148, 136),
                 '王羲之': (225, 29, 72), '苏轼': (147, 51, 234),
                 '何绍基': (217, 119, 6), '米芾': (22, 163, 74),
                 '褚遂良': (100, 116, 139)}
_CACHE = {}


def png_dir(cdir):
    if cdir in _CACHE:
        return _CACHE[cdir]
    base = ROOT / cdir
    best, best_key = None, None
    for cand in [base, *[p for p in base.rglob('*') if p.is_dir()]]:
        n_g = sum(1 for f in cand.glob('g[0-9]*.png') if f.stem[1:].isdigit())
        n_gt = sum(1 for f in cand.glob('gt*.png'))
        if n_g == 0:
            continue
        key = (1 if n_gt > 0 else 0, n_g)
        if best_key is None or key > best_key:
            best, best_key = cand, key
    _CACHE[cdir] = best
    return best


def font(size):
    for p in (ROOT / 'tools/fonts/SimHei.ttf', Path('C:/Windows/Fonts/simhei.ttf')):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def cell_png(cdir, i, rows):
    if cdir == '__gt__':
        return ROOT / rows[i]['image_path']
    return png_dir(cdir) / f'g{i}.png'


def sheet(rows, idx, title, sub, out):
    n_row, n_col = len(idx), len(ARMS)
    W = PAD + LABEL_W + n_col * (CELL + GAP) + PAD
    H = HEADER_H + n_row * (CELL + GAP) + FOOTER_H
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    f_t, f_s, f_h, f_h2, f_r, f_n = (font(36), font(18), font(19), font(14),
                                     font(20), font(15))
    d.text((PAD, 12), title, font=f_t, fill=(15, 23, 42))
    d.text((PAD, 62), sub, font=f_s, fill=(71, 85, 105))
    d.text((PAD, 92), '所有臂同一 (书家,字) 行、同一 std 骨架、同一噪声种子；'
                      '★ 标红列为本次后训练臂。', font=f_s, fill=(71, 85, 105))

    y0 = HEADER_H - 52
    for j, (_c, l1, l2) in enumerate(ARMS):
        x = PAD + LABEL_W + j * (CELL + GAP)
        ours = j >= 2
        d.text((x + 2, y0), l1, font=f_h, fill=(185, 28, 28) if ours else (15, 23, 42))
        d.text((x + 2, y0 + 22), l2, font=f_h2, fill=(100, 116, 139))

    miss = 0
    for r, i in enumerate(idx):
        y = HEADER_H + r * (CELL + GAP)
        if r % 2 == 0:
            d.rectangle([PAD, y - 2, W - PAD, y + CELL + 1], fill=(248, 250, 252))
        d.text((PAD + 4, y + 26), rows[i]['calligrapher'], font=f_r,
               fill=WRITER_COLORS.get(rows[i]['calligrapher'], (15, 23, 42)))
        d.text((PAD + 4, y + 56), rows[i]['character'], font=f_r, fill=(15, 23, 42))
        d.text((PAD + 4, y + 86), f'#{i}', font=f_n, fill=(148, 163, 184))
        for j, (cdir, _l1, _l2) in enumerate(ARMS):
            x = PAD + LABEL_W + j * (CELL + GAP)
            p = cell_png(cdir, i, rows)
            if not p.exists():
                miss += 1
                d.rectangle([x, y, x + CELL, y + CELL], fill=(254, 226, 226))
                d.text((x + 40, y + 60), 'MISS', font=f_n, fill=(185, 28, 28))
                continue
            im = Image.open(p).convert('L').resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(im.convert('RGB'), (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1],
                        outline=(185, 28, 28) if j >= 2 else (226, 232, 240),
                        width=2 if j >= 2 else 1)

    fy = H - FOOTER_H + 8
    d.line([PAD, fy - 6, W - PAD, fy - 6], fill=(203, 213, 225))
    d.text((PAD, fy), f'{n_row} 字 · {n_col} 列（含 GT） · 缺失 {miss} 格  |  '
                      f'面板 validation_val100.csv  |  2026-11-01',
           font=f_n, fill=(100, 116, 139))
    canvas.save(out)
    print(f'wrote {out}  {W}x{H}  miss={miss}')


def main():
    rows = list(csv.DictReader(open(PANEL, encoding='utf-8-sig')))
    out_dir = ROOT / 'assets/d_20261101/review'
    out_dir.mkdir(parents=True, exist_ok=True)

    spread = list(range(0, len(rows), 5))[:20]
    sheet(rows, spread, '后训练臂对照 · 20 字（跨书家均匀取样）',
          f'行 = val100 第 {spread[0]}–{spread[-1]} 行，每 5 行取 1，覆盖全部书家。',
          out_dir / 'review_20_spread.png')

    def lpips(arm):
        p = ROOT / f'scratch/d_eval/{arm}.csv'
        if not p.exists():
            return {}
        return {int(r['idx']): float(r['lpips']) for r in
                csv.DictReader(open(p, encoding='utf-8-sig'))}

    k = lpips('K_A')
    best = lpips('D3_11000_plus_D1dec')
    if k and best:
        gain = sorted((i for i in k if i in best), key=lambda i: best[i] - k[i])[:20]
        sheet(rows, sorted(gain), '后训练臂对照 · D3+D1dec 相对 K 改进最大的 20 字',
              '按 LPIPS 逐字降幅排序取前 20；用于确认「改进」在肉眼上是否成立。',
              out_dir / 'review_20_bestgain.png')
    else:
        print('skip delta sheet: scratch/d_eval csv not found on this host')


if __name__ == '__main__':
    main()

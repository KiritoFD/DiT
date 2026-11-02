"""Rebuild the decoder/style poster with S4, S5, J2, J3 added.

What each arm is (from docs/1031/01_EVIDENCE_MATRIX.md):
  S4  LocalCNN -> Spatial AdaGN          style-representation arm
  S5  LocalAttention -> XAttn            style-representation arm
  J2  unfreeze Decoder + image loss + GT reconstruction
  J3  unfreeze Decoder + image loss, no GT reconstruction

Comparability, verified rather than assumed
-------------------------------------------
The row axis is the 21 (writer, character) cells shared by the panels, chosen so that one
GT column is valid for every row. That holds only if each column's GT is the same image.

  * S4/S5 poster images are indexed by val100 `idx` (verified: all 21 indices match
    batch.csv, and cell characters agree 21/21). Their GT is BIT-IDENTICAL to K's val100
    GT (checked by hashing and by mean|diff| = 0.0000 on every index tested). So S4/S5 are
    directly comparable to K on the same rows.
  * J2/J3 are the heldout187 panel. They cover all 21 (writer, character) pairs but from a
    DIFFERENT source image in every single case (0/21 share a GT file).

J2/J3 have NO surviving generated images. Their `batch.csv` under docs/1028/raw/ carries
per-row metrics but the eval image trees were not archived, and no checkpoint remains to
regenerate them (checked locally and on the 4090). Rather than silently drop the columns
or -- worse -- pair J2/J3 metrics with somebody else's pictures, this poster renders them
as a METRIC-ONLY strip: the per-cell number is printed where the glyph would be, against
their own GT, and the header says plainly that the image is missing.

Usage:
  python scratch/panels/poster_s4s5j2j3.py
"""
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / 'scratch/panels/poster_imgs'
KDIR = next((ROOT / 'assets/d_20261101/reports_assets/gar_crossdecode/K_A')
            .glob('eval_samples_ctrl/*/eval200fix'))
OUT = ROOT / 'assets/metrics_index/poster_s4s5j2j3.png'
OUT_RAW = ROOT / 'assets/metrics_index/poster_s4s5j2j3_raw.png'

CELL = 132
GAP = 5
LABEL_W = 128
HEADER_H = 236
FOOTER_H = 158
PAD = 18

VAL100 = 'val100'      # K and its decoder variants; S4/S5 provably share this GT
METRIC_ONLY = 'metric'  # J2/J3: metrics survive, generated images do not


def load_jsonl_rows(path):
    return list(csv.DictReader(open(path, encoding='utf-8')))


def build_columns():
    """(key, dir/kind, line1, line2, panel, is_ours)."""
    cols = []
    # ---- val100 group -------------------------------------------------------
    cols.append(('gt', None, 'GT 真迹', 'val100 同源', VAL100, False))
    for step in (100, 336, 672):
        d = ROOT / f'assets/d_20261101/reports_assets/gar_crossdecode/GAR_s{step:06d}'
        cols.append((f'GAR{step}', d, f'GAR@{step}', '冻结K+解码器', VAL100, False))
    cols.append(('K_plus_D1dec',
                 ROOT / 'assets/d_20261101/reports_assets/gar_crossdecode/K_plus_D1dec',
                 'K+D1dec', '换解码器', VAL100, False))
    cols.append(('S4', 'S4_50k', 'S4', 'LocalCNN→AdaGN', VAL100, False))
    cols.append(('S5', 'S5_50k', 'S5', 'LocalAttn→XAttn', VAL100, False))
    cols.append(('K', ROOT / 'assets/d_20261101/reports_assets/gar_crossdecode/K_A',
                 'K ★ 基线', 'abc 43,429', VAL100, True))
    # ---- J2/J3: metrics survive, images do not (see module docstring) --------
    cols.append(('J2', 'J2_187', 'J2', '解冻解码器+GT重建', METRIC_ONLY, False))
    cols.append(('J3', 'J3_187', 'J3', '解冻解码器,无GT重建', METRIC_ONLY, False))
    return cols


def load_cells():
    return json.loads((ROOT / 'scratch/panels/cells.json').read_text(encoding='utf-8'))['cells']


def j2j3_lookup(arm):
    """(writer, char) -> row, because these panels key on their own GT instances."""
    rows = load_jsonl_rows(ROOT / f'docs/1028/raw/{arm}/batch.csv')
    out = {}
    for r in rows:
        out.setdefault((r['calligrapher'], r['char']), r)
    return out


_METRIC_CACHE = {}


def j2j3_metric(arm, cell):
    """Per-cell metrics for the metric-only columns."""
    if arm not in _METRIC_CACHE:
        _METRIC_CACHE[arm] = j2j3_lookup(arm)
    return _METRIC_CACHE[arm].get((cell['writer'], cell['char']))


def font(size):
    for p in (ROOT / 'tools/fonts/SimHei.ttf', Path('C:/Windows/Fonts/simhei.ttf'),
              Path('C:/Windows/Fonts/msyh.ttc')):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _leaf(base):
    """Resolve an arm root to its eval200fix leaf dir.

    Column definitions name the arm ROOT (the stable, meaningful path); the images sit
    under eval_samples_ctrl/<step>/eval200fix. Resolving here keeps the column table
    readable and means a step change does not silently empty a column.
    """
    if base is None:
        return KDIR
    if (base / 'g0.png').is_file():
        return base                      # already a leaf
    found = sorted(base.glob('eval_samples_ctrl/*/eval200fix'))
    return found[-1] if found else base


def cell_image(col, cell, key):
    """Resolve one cell to a grayscale image, or (None, None) if unavailable."""
    ckey, base, _l1, _l2, panel, _ours = col
    idx100 = cell['idx100']
    if panel != VAL100:
        return None, None                # metric-only columns have no pictures
    if ckey == 'gt':
        p = KDIR / f'gt{idx100}.png'
    elif isinstance(base, Path):
        p = _leaf(base) / f'g{idx100}.png'
    else:
        p = PANEL / base / f'g{idx100}.png'
    if not p.is_file():
        return None, None
    return Image.open(p).convert('L').resize((CELL, CELL), Image.LANCZOS), p


def main():
    cells = load_cells()
    cols = build_columns()
    n_row, n_col = len(cells), len(cols)

    W = PAD + LABEL_W + n_col * (CELL + GAP) + PAD
    H = HEADER_H + n_row * (CELL + GAP) + FOOTER_H
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    f_title, f_sub = font(38), font(19)
    f_head, f_head2 = font(20), font(15)
    f_row, f_note = font(21), font(16)

    d.text((PAD, 12), 'Decoder / 风格表征臂对比 · 21 单元  S4 · S5 · J2 · J3',
           font=f_title, fill=(15, 23, 42))
    d.text((PAD, 60),
           '行轴 = 各面板共有的 21 个 (书家, 字) 单元。S4/S5 与 K 同行 GT 逐位相同（已校验，可直接对比）；',
           font=f_sub, fill=(71, 85, 105))
    d.text((PAD, 86),
           '⚠ J2/J3 为 metric-only 列：其生成图未归档、checkpoint 亦不存在（本机与 4090 均已确认），'
           '故格内只印该行自己的实测指标，不借用他人图像。',
           font=f_sub, fill=(180, 83, 9))
    d.text((PAD, 112),
           'S4/S5 = 风格表征臂；J2/J3 = 解冻解码器臂。生成列均为 EMA 权重、CFG 1.0、50 步 Heun、同 std 骨架条件。'
           '★ = 当前基线 K（A@43429）。',
           font=f_sub, fill=(71, 85, 105))

    y0 = HEADER_H - 74
    for j, col in enumerate(cols):
        ckey, _b, l1, l2, panel, ours = col
        x = PAD + LABEL_W + j * (CELL + GAP)
        cc = (185, 28, 28) if ours else (15, 23, 42)
        d.text((x + 2, y0), l1, font=f_head, fill=cc)
        d.text((x + 2, y0 + 26), l2, font=f_head2, fill=(100, 116, 139))
        tag = 'val100' if panel == VAL100 else '仅指标'
        d.text((x + 2, y0 + 46), tag, font=f_head2,
               fill=(22, 163, 74) if panel == VAL100 else (180, 83, 9))
        if ours:
            d.rectangle([x - 2, y0 - 6, x + CELL + 1, y0 + 66], outline=(185, 28, 28), width=2)

    # group separator between the comparable and the metric-only columns
    n_val = sum(1 for c in cols if c[4] == VAL100)
    sx = PAD + LABEL_W + n_val * (CELL + GAP) - GAP // 2 - 2
    d.line([sx, HEADER_H - 10, sx, H - FOOTER_H + 6], fill=(180, 83, 9), width=3)

    misses = 0
    for i, c in enumerate(cells):
        y = HEADER_H + i * (CELL + GAP)
        if i % 2 == 0:
            d.rectangle([PAD, y - 2, W - PAD, y + CELL + 1], fill=(248, 250, 252))
        d.text((PAD + 4, y + 22), c['writer'], font=f_row, fill=(15, 23, 42))
        d.text((PAD + 4, y + 50), c['char'], font=f_row, fill=(15, 23, 42))
        d.text((PAD + 4, y + 82), f'#{c["idx100"]}', font=f_note, fill=(148, 163, 184))

        for j, col in enumerate(cols):
            x = PAD + LABEL_W + j * (CELL + GAP)
            if col[4] == METRIC_ONLY:
                # No pictures survive for this arm: show the cell's own measured
                # numbers instead of borrowing someone else's render.
                row = j2j3_metric(col[0], c)
                d.rectangle([x, y, x + CELL, y + CELL], fill=(255, 251, 235),
                            outline=(253, 230, 138), width=1)
                if row is None:
                    d.text((x + CELL // 3, y + CELL // 2 - 8), 'N/A', font=f_note,
                           fill=(185, 28, 28))
                    misses += 1
                    continue
                f_m = font(17)
                d.text((x + 6, y + 14), f'SSIM {float(row["ssim"]):.4f}', font=f_m,
                       fill=(120, 53, 15))
                d.text((x + 6, y + 38), f'LPIPS {float(row["lpips"]):.4f}', font=f_m,
                       fill=(120, 53, 15))
                d.text((x + 6, y + 62), f'IoU {float(row["ink_iou"]):.4f}', font=f_m,
                       fill=(120, 53, 15))
                d.text((x + 6, y + 86), f'frag {float(row["frag_ratio"]):.2f}', font=f_m,
                       fill=(120, 53, 15))
                d.text((x + 6, y + 108), '无生成图', font=font(14), fill=(180, 83, 9))
                continue
            im, _src = cell_image(col, c, 'gen')
            if im is None:
                misses += 1
                d.rectangle([x, y, x + CELL, y + CELL], fill=(254, 226, 226))
                d.text((x + CELL // 3, y + CELL // 2 - 8), 'MISS', font=f_note,
                       fill=(185, 28, 28))
                continue
            canvas.paste(im.convert('RGB'), (x, y))
            if col[5]:
                d.rectangle([x - 1, y - 1, x + CELL, y + CELL], outline=(185, 28, 28), width=2)
            else:
                d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(226, 232, 240), width=1)

    fy = H - FOOTER_H + 14
    d.line([PAD, fy - 8, W - PAD, fy - 8], fill=(203, 213, 225), width=1)
    d.text((PAD, fy),
           f'书家 {len({c["writer"] for c in cells})} 家 · {n_row} 字 · {n_col} 列  |  '
           '左侧 val100 组（GT/K/GAR/K+D1dec/S4/S5）同行 GT 逐位相同，可直接逐行相减',
           font=f_note, fill=(100, 116, 139))
    d.text((PAD, fy + 28),
           'S4/S5 的 GT 与 K 的 val100 GT 经哈希与逐像素比对确认一致（mean|diff|=0.0000）。'
           'J2/J3 仅存指标、无生成图，故不与任何列做像素比较；其指标为其自身 187 面板口径。',
           font=f_note, fill=(100, 116, 139))
    if misses:
        d.text((PAD, fy + 56), f'⚠ {misses} 个单元无可用数据', font=f_note,
               fill=(185, 28, 28))

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_RAW)
    canvas.resize((int(W * 0.62), int(H * 0.62)), Image.LANCZOS).save(OUT)
    print(f'wrote {OUT_RAW}  {W}x{H}')
    print(f'wrote {OUT}  (0.62x)')
    print(f'missing cells: {misses}')


if __name__ == '__main__':
    main()

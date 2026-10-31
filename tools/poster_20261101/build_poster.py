"""Build the cross-lineage calligraphy poster.

Row axis = the 21 (calligrapher, character) cells that exist in BOTH eval panels,
so a single GT column is valid and every column is directly comparable.
Columns are ordered chronologically along the model lineage.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / 'scratch/panels/poster_imgs'
OUT = ROOT / 'assets/posters/poster_crosslineage_21.png'
OUT_RAW = ROOT / 'assets/posters/poster_crosslineage_21_raw.png'

CELL = 132
GAP = 5
LABEL_W = 126
# Header geometry: title, then two explanation lines, then a real gap before the
# column headers. The old layout put the second line at y=96 and the headers at
# y=114, so they nearly touched.
TITLE_Y = 14
SUB1_Y = 74
SUB2_Y = 106
HEAD_BLOCK = 62
HEADER_H = 230
FOOTER_H = 122
PAD = 18

WRITER_COLORS = {
    '赵孟頫': (37, 99, 235),
    '文徵明': (13, 148, 136),
    '王羲之': (225, 29, 72),
    '苏轼': (147, 51, 234),
    '何绍基': (217, 119, 6),
    '米芾': (22, 163, 74),
    '褚遂良': (100, 116, 139),
}

# (base dir, header line 1, header line 2, panel kind, is_ours)
# base dir may be absolute-in-repo; the pulled columns live under scratch/panels/poster_imgs.
REMOTE = PANEL
COLUMNS = [
    ('GT', 'GT 真迹', '原始真迹（data/top10_style23）', '100', False),
    ('assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/'
     'eval_samples_ctrl/step0200000/eval200fix', 'v80 physics', 'CalliVAE 200k', '200', False),
    ('assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/'
     'eval_samples_ctrl/step0030000/eval200fix', 'v85A style_ada', '风格注入 30k', '200', False),
    ('S9_50k', 'S9', 'style_matrix 50k', '200', False),
    ('S0_85k', 'S0', 'style_focus 85k', '200', False),
    # matrix64 siblings of S9, both at 50k on the val100 panel. S5 was trained on the
    # 48 box and copied back, so its panel has no request.json -- its 21 cells were
    # exported and are used here on the same (writer, char) rows as every other column.
    ('S4_50k', 'S4 ★', 'matrix64 50k', '100', True),
    ('S5_50k', 'S5 ★', 'matrix64 50k（48 机）', '100', True),
    ('docs/106moyi/moyi/results/moyi_top10_rf/eval_ours200fix', 'Moyi 12ch', '跨架构基线 50k', '200', False),
    ('P3A_20k', 'P3A', 'P125 段 20k', '100', False),
    ('A_35k', 'A', 'abc 链 35k', '200', False),
    ('A_43429', 'A ★ K', 'abc 链 43,429（最近最佳）', '100', True),
]


def col_base(cdir):
    p = ROOT / cdir
    if p.is_dir():
        return p
    return REMOTE / cdir


def font(size):
    for p in (ROOT / 'tools/fonts/SimHei.ttf', Path('C:/Windows/Fonts/simhei.ttf'),
              Path('C:/Windows/Fonts/msyh.ttc')):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def load_cell(cdir, kind, idx200, idx100):
    base = col_base(cdir)
    if kind == 'gt':
        p = base / f'gt{idx200}.png'
    elif kind == '200':
        p = base / f'g{idx200}.png'
    else:
        p = base / f'g{idx100}.png'
    if not p.exists():
        return None
    im = Image.open(p).convert('L')
    return im.resize((CELL, CELL), Image.LANCZOS)


def main():
    cells = json.loads((ROOT / 'tools/poster_20261101/cells.json').read_text(encoding='utf-8'))['cells']
    n_row, n_col = len(cells), len(COLUMNS)

    W = PAD + LABEL_W + n_col * (CELL + GAP) + PAD
    H = HEADER_H + n_row * (CELL + GAP) + FOOTER_H
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    f_title, f_sub = font(40), font(20)
    f_head, f_head2 = font(21), font(16)
    f_row, f_note = font(22), font(17)

    # ---- header
    d.text((PAD, TITLE_Y), '书家风格全谱系对比 · 21 字级', font=f_title, fill=(15, 23, 42))
    d.text((PAD, SUB1_Y),
           '行轴 = eval200fix(187) 与 val100(100) 两个评测面板共有的 21 个 (书家, 字) 单元；'
           '同一行在两面板中 GT 同源，故单列 GT 对全表成立。列按血统时间顺序排列。',
           font=f_sub, fill=(71, 85, 105))
    d.text((PAD, SUB2_Y),
           '★ = 当前最优 K（A@43429）与新增的 matrix64 S4/S5@50k。'
           '所有列均为 EMA 权重、CFG 1.0、50 步 Heun 采样、同 std 骨架条件。',
           font=f_sub, fill=(71, 85, 105))

    # column headers
    y0 = HEADER_H - HEAD_BLOCK
    for j, (cdir, l1, l2, kind, ours) in enumerate(COLUMNS):
        x = PAD + LABEL_W + j * (CELL + GAP)
        col = (185, 28, 28) if ours else (15, 23, 42)
        d.text((x + 2, y0), l1, font=f_head, fill=col)
        d.text((x + 2, y0 + 26), l2, font=f_head2, fill=(100, 116, 139))
        if ours:
            d.rectangle([x - 2, y0 - 6, x + CELL + 1, y0 + 52], outline=(185, 28, 28), width=2)

    # ---- rows
    for i, c in enumerate(cells):
        y = HEADER_H + i * (CELL + GAP)
        wc = WRITER_COLORS.get(c['writer'], (15, 23, 42))
        if i % 2 == 0:
            d.rectangle([PAD, y - 2, W - PAD, y + CELL + 1], fill=(248, 250, 252))
        d.text((PAD + 4, y + 26), c['writer'], font=f_row, fill=wc)
        d.text((PAD + 4, y + 56), c['char'], font=f_row, fill=(15, 23, 42))
        d.text((PAD + 4, y + 88), f'#{c["idx200"]}/{c["idx100"]}', font=f_note, fill=(148, 163, 184))

        for j, (cdir, _l1, _l2, kind, ours) in enumerate(COLUMNS):
            x = PAD + LABEL_W + j * (CELL + GAP)
            im = load_cell(cdir, kind, c['idx200'], c['idx100'])
            if im is None:
                d.rectangle([x, y, x + CELL, y + CELL], fill=(254, 226, 226))
                d.text((x + CELL // 3, y + CELL // 2 - 8), 'MISS', font=f_note, fill=(185, 28, 28))
                continue
            canvas.paste(im.convert('RGB'), (x, y))
            if ours:
                d.rectangle([x - 1, y - 1, x + CELL, y + CELL], outline=(185, 28, 28), width=2)
            else:
                d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(226, 232, 240), width=1)

    # ---- footer
    fy = H - FOOTER_H + 12
    d.line([PAD, fy - 8, W - PAD, fy - 8], fill=(203, 213, 225), width=1)
    d.text((PAD, fy),
           f'书家 {len({c["writer"] for c in cells})} 家 · {n_row} 字 · {n_col} 列（含 GT）  |  '
           'Moyi 4ch@80k 因仅有 40 字官方海报、面板不同，未纳入逐行对比  |  2026-11-01',
           font=f_note, fill=(100, 116, 139))
    d.text((PAD, fy + 26),
           '列说明：v80/v85A/S9/S0/Moyi/A@35k 走 187 行面板；P3A/S4/S5/A@43429 走 100 行面板。'
           '两类面板的同一 (书家,字) 行 GT 完全一致（21/21 校验通过）。',
           font=f_note, fill=(100, 116, 139))
    d.text((PAD, fy + 52),
           'S4/S5 为 matrix64 家族 50k 臂，与 S9 同期同族；S5 由 48 机训练后拷回 4090，'
           '其面板无 request.json，故以 21 格逐行导出方式纳入。',
           font=f_note, fill=(100, 116, 139))

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_RAW)
    canvas.resize((int(W * 0.62), int(H * 0.62)), Image.LANCZOS).save(OUT)
    print(f'wrote {OUT_RAW}  {W}x{H}')
    print(f'wrote {OUT}  (0.62x)')


if __name__ == '__main__':
    main()

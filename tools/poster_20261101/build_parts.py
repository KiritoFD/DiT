"""Build the cross-lineage poster in parts.

Two panels exist, and they do not share enough rows to hold every model in one
table:

* val100 (100 rows)  -- matrix64 S0/S1/S4/S5/S6/S8@50k, P3A@20k, K@43429
* eval200_fixed (187 rows) -- v80, v85A, S9, S0@85k, Moyi 12ch, A@35k

So the poster is split by panel, each part getting its own GT column taken from
that panel's CSV `image_path`. Every model column is validated against that GT
before anything is drawn: the panel's own `gt{i}.png` must match the CSV original
at index i, which is what proves the column is row-aligned (S5 has no
request.json, so this check is the only proof it lines up).

Run this on the machine that holds the panels; only the finished PNGs travel.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

CELL = 132
GAP = 5
LABEL_W = 126
PAD = 18
FOOTER_H = 74
# Header geometry: the title, then two explanation lines, then a clear gap before
# the column headers. The earlier layout let the second line touch the headers.
TITLE_Y = 14
SUB1_Y = 74
SUB2_Y = 106
HEAD_BLOCK = 62
HEADER_H = 230

WRITER_COLORS = {
    '赵孟頫': (37, 99, 235), '文徵明': (13, 148, 136), '王羲之': (225, 29, 72),
    '苏轼': (147, 51, 234), '何绍基': (217, 119, 6), '米芾': (22, 163, 74),
    '褚遂良': (100, 116, 139),
}

M64 = 'assets/style_matrix_20261027/matrix64'
VAL100 = 'assets/u100_pretrain_20261024/validation_val100.csv'
FIXED187 = 'exp-std/csv/eval200_fixed.csv'

# (dir, line1, line2, highlight)  -- kind is implied by the part's panel
VAL100_COLUMNS = [
    ('__gt__', 'GT 真迹', '原始真迹', False),
    (f'{M64}/4090/execution/runs/S0/evaluations/0050000/heldout/eval_samples_ctrl',
     'S0', 'matrix64 50k', False),
    (f'{M64}/4090/execution/runs/S1/evaluations/0050000/heldout/eval_samples_ctrl',
     'S1', 'matrix64 50k', False),
    (f'{M64}/4090/execution/runs/S4/evaluations/0050000/heldout/eval_samples_ctrl',
     'S4 ★ 新增', 'matrix64 50k', True),
    (f'{M64}/48/execution/runs/S5/evaluations/0050000/heldout_s/eval_samples_ctrl',
     'S5 ★ 新增', 'matrix64 50k（48 机）', True),
    (f'{M64}/4090/execution/runs/S6/evaluations/0050000/heldout/eval_samples_ctrl',
     'S6', 'matrix64 50k', False),
    (f'{M64}/4090/execution/runs/S8/evaluations/0050000/heldout/eval_samples_ctrl',
     'S8', 'matrix64 50k', False),
    ('assets/p125_20261030/execution/runs/P3A/evaluations/0020000/heldout/eval_samples_ctrl',
     'P3A', 'P125 段 20k', False),
    ('assets/f_20261101/eval/K_A/eval_samples_ctrl', 'A ★ K', 'abc 链 43,429（最优）', True),
]

FIXED_COLUMNS = [
    ('__gt__', 'GT 真迹', '原始真迹', False),
    ('assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/'
     'eval_samples_ctrl/step0200000/eval200fix', 'v80 physics', 'CalliVAE 200k', False),
    ('assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/'
     'eval_samples_ctrl/step0030000/eval200fix', 'v85A style_ada', '风格注入 30k', False),
    ('assets/style_matrix_20261028/execution/runs/S9/evaluations/0050000/heldout/eval_samples_ctrl',
     'S9', 'style_matrix 50k', False),
    ('assets/style_focus_85k_20261029/4090/runs/S0/evaluations/0085000/shared_train_panel/eval_samples_ctrl',
     'S0', 'style_focus 85k', False),
    ('docs/106moyi/moyi/results/moyi_top10_rf/eval_ours200fix', 'Moyi 12ch',
     '跨架构基线 50k', False),
    ('assets/abc_20261031/execution/runs/A/evaluations/0035000/heldout187_seed0/eval_samples_ctrl',
     'A', 'abc 链 35k', False),
]

PARTS = {
    'val100': dict(csv=VAL100, columns=VAL100_COLUMNS, rows=100,
                   title='书家风格全谱系对比 · val100 面板（100 字）'),
    'fixed187': dict(csv=FIXED187, columns=FIXED_COLUMNS, rows=100,
                     title='书家风格全谱系对比 · eval200fix 面板（187 字抽样 100）'),
}


def font(size):
    for p in (ROOT / 'tools/fonts/SimHei.ttf', Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def gray(p, size=CELL):
    return Image.open(p).convert('L').resize((size, size), Image.LANCZOS)


_PNG_DIR_CACHE = {}


def png_dir(cdir):
    """Resolve the directory that actually holds g*.png.

    The panels sit at different depths (``eval_samples_ctrl/step0050000/eval200fix``
    for the shared trainer, but straight in ``eval_ours200fix`` for the moyi run), so
    pick the deepest directory with the most numbered g files instead of hard-coding.
    """
    if cdir in _PNG_DIR_CACHE:
        return _PNG_DIR_CACHE[cdir]
    base = ROOT / cdir
    best, best_key = None, None
    if base.is_dir():
        for cand in [base, *[p for p in base.rglob('*') if p.is_dir()]]:
            n_g = sum(1 for f in cand.glob('g[0-9]*.png') if f.stem[1:].isdigit())
            n_gt = sum(1 for f in cand.glob('gt*.png'))
            if n_g == 0:
                continue
            # A sibling dir holds the *input* G images and no GT, so requiring GT
            # files is what separates the real output panel from that decoy.
            key = (1 if n_gt > 0 else 0, n_g)
            if best_key is None or key > best_key:
                best, best_key = cand, key
    _PNG_DIR_CACHE[cdir] = best
    return best


def pick_indices(n_total, n_want):
    """Evenly sample n_want row indices out of n_total, keeping them ascending."""
    if n_want >= n_total:
        return list(range(n_total))
    idx = sorted({round(i * (n_total - 1) / (n_want - 1)) for i in range(n_want)})
    return idx


def check_alignment(dirpath, rows, indices, tag):
    """The panel's own gt{i}.png must match the CSV original at row i."""
    worst, bad = 0.0, []
    for i in indices:
        p = dirpath / f'gt{i}.png'
        src = ROOT / rows[i]['image_path']
        if not p.exists() or not src.exists():
            bad.append((i, 'missing gt' if not p.exists() else 'missing src'))
            continue
        a = np.asarray(gray(p, 96), dtype=np.float32) / 255.
        b = np.asarray(gray(src, 96), dtype=np.float32) / 255.
        diff = float(np.abs(a - b).mean())
        worst = max(worst, diff)
        if diff > 0.06:
            bad.append((i, round(diff, 4)))
    print(f'  align {tag}: worst |gt-original| = {worst:.4f}, failures = {len(bad)}')
    if bad:
        print(f'    first failures: {bad[:5]}')
    return not bad


def draw_sheet(part, rows, indices, columns, sheet_no, n_sheets, out_path):
    n_row, n_col = len(indices), len(columns)
    W = PAD + LABEL_W + n_col * (CELL + GAP) + PAD
    H = HEADER_H + n_row * (CELL + GAP) + FOOTER_H
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f_title, f_sub = font(40), font(20)
    f_head, f_head2 = font(20), font(15)
    f_row, f_note = font(21), font(16)

    d.text((PAD, TITLE_Y), part['title'], font=f_title, fill=(15, 23, 42))
    d.text((PAD, SUB1_Y),
           f'第 {sheet_no}/{n_sheets} 张 · 行 {indices[0]}–{indices[-1]}（共 {len(indices)} 字）。'
           f'GT 列取自本面板 CSV 的 image_path 原始真迹。',
           font=f_sub, fill=(71, 85, 105))
    d.text((PAD, SUB2_Y),
           '所有列为 EMA 权重、CFG 1.0、50 步 Heun 采样、同 std 骨架条件；'
           '每列已逐行校验 gt{i} 与 CSV 原始真迹一致。',
           font=f_sub, fill=(71, 85, 105))

    y0 = HEADER_H - HEAD_BLOCK
    for j, (_cd, l1, l2, ours) in enumerate(columns):
        x = PAD + LABEL_W + j * (CELL + GAP)
        col = (185, 28, 28) if ours else (15, 23, 42)
        d.text((x + 2, y0), l1, font=f_head, fill=col)
        d.text((x + 2, y0 + 24), l2, font=f_head2, fill=(100, 116, 139))
        if ours:
            d.rectangle([x - 3, y0 - 7, x + CELL + 2, y0 + 50], outline=(185, 28, 28), width=2)

    miss = 0
    for r, i in enumerate(indices):
        y = HEADER_H + r * (CELL + GAP)
        wc = WRITER_COLORS.get(rows[i]['calligrapher'], (15, 23, 42))
        if r % 2 == 0:
            d.rectangle([PAD, y - 2, W - PAD, y + CELL + 1], fill=(248, 250, 252))
        d.text((PAD + 4, y + 24), rows[i]['calligrapher'], font=f_row, fill=wc)
        d.text((PAD + 4, y + 56), rows[i]['character'], font=f_row, fill=(15, 23, 42))
        d.text((PAD + 4, y + 88), f'#{i}', font=f_note, fill=(148, 163, 184))

        for j, (cdir, _l1, _l2, ours) in enumerate(columns):
            x = PAD + LABEL_W + j * (CELL + GAP)
            if cdir == '__gt__':
                p = ROOT / rows[i]['image_path']
            else:
                p = png_dir(cdir) / f'g{i}.png'
            if not p.exists():
                miss += 1
                d.rectangle([x, y, x + CELL, y + CELL], fill=(254, 226, 226))
                d.text((x + CELL // 3, y + CELL // 2 - 8), 'MISS', font=f_note,
                       fill=(185, 28, 28))
                continue
            canvas.paste(gray(p).convert('RGB'), (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1],
                        outline=(185, 28, 28) if ours else (226, 232, 240),
                        width=2 if ours else 1)

    fy = H - FOOTER_H + 10
    d.line([PAD, fy - 8, W - PAD, fy - 8], fill=(203, 213, 225), width=1)
    d.text((PAD, fy),
           f'{len({rows[i]["calligrapher"] for i in indices})} 家书家 · {n_row} 字 · '
           f'{n_col} 列（含 GT） · 缺失 {miss} 格  |  '
           f'面板 {Path(part["csv"]).name}  |  2026-11-01',
           font=f_note, fill=(100, 116, 139))
    d.text((PAD, fy + 24),
           'S4/S5 为 matrix64 家族 50k 臂（S5 由 48 机产出后拷回，无 request.json，'
           '故用 gt{i} 逐行对齐校验代替 sha 比对）。',
           font=f_note, fill=(100, 116, 139))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f'  wrote {out_path}  {W}x{H}  miss={miss}')
    return miss


def build_part(name, part, out_dir, per_sheet, limit_rows):
    print(f'=== part {name}')
    rows = list(csv.DictReader(open(ROOT / part['csv'], encoding='utf-8-sig')))
    print(f'  panel rows: {len(rows)}')
    n_total = len(rows) if name == 'val100' else len(rows)
    indices = pick_indices(n_total, min(part['rows'], limit_rows, n_total))
    ok = True
    for cdir, l1, _l2, _o in part['columns']:
        if cdir == '__gt__':
            continue
        ok &= check_alignment(png_dir(cdir), rows, indices, l1)
    if not ok:
        raise SystemExit(f'part {name}: a column is not row-aligned to {part["csv"]}')
    sheets = [indices[i:i + per_sheet] for i in range(0, len(indices), per_sheet)]
    total_miss = 0
    for s, chunk in enumerate(sheets, 1):
        out = out_dir / f'poster_{name}_p{s}.png'
        total_miss += draw_sheet(part, rows, chunk, part['columns'], s, len(sheets), out)
    print(f'  {name}: {len(sheets)} sheets, {total_miss} missing cells')
    return len(sheets), total_miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='assets/posters')
    ap.add_argument('--per-sheet', type=int, default=15)
    ap.add_argument('--rows', type=int, default=100)
    ap.add_argument('--parts', default='val100,fixed187')
    a = ap.parse_args()
    out_dir = ROOT / a.out_dir
    summary = {}
    for name in [x.strip() for x in a.parts.split(',')]:
        summary[name] = build_part(name, PARTS[name], out_dir, a.per_sheet, a.rows)
    (out_dir / 'parts_summary.json').write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print('\n', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()

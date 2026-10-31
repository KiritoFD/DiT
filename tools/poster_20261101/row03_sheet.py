"""Build a large side-by-side sheet for row 3 (赵孟頫/火), including the 大 truth as a reference."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'scratch/panels/poster_imgs'
D = ROOT / 'scratch/panels/huoprobe'
cells = json.loads((ROOT / 'tools/poster_20261101/cells.json').read_text(encoding='utf-8'))['cells']

COLS = [
    ('GT 火 (真迹)', P / 'GT' / 'g43.png', 'truth'),
    ('v80', ROOT / 'assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/'
            'eval_samples_ctrl/step0200000/eval200fix/g159.png', 'model'),
    ('v85A', ROOT / 'assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/'
             'eval_samples_ctrl/step0030000/eval200fix/g159.png', 'model'),
    ('S9', P / 'S9_50k' / 'g159.png', 'model'),
    ('S0', P / 'S0_85k' / 'g159.png', 'model'),
    ('Moyi', ROOT / 'docs/106moyi/moyi/results/moyi_top10_rf/eval_ours200fix/g159.png', 'model'),
    ('P3A', P / 'P3A_20k' / 'g43.png', 'model'),
    ('A35k', P / 'A_35k' / 'g159.png', 'model'),
    ('A43429', P / 'A_43429' / 'g43.png', 'model'),
    ('大 (真迹, 参照)', D / 'da_truth_002356.png', 'ref'),
]

CELL, GAP, HEAD, FOOT = 220, 10, 46, 34
W = len(COLS) * (CELL + GAP) + GAP
H = HEAD + CELL + FOOT
sheet = Image.new('RGB', (W, H), 'white')
dr = ImageDraw.Draw(sheet)
try:
    f_big = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    f_sm = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
except Exception:
    f_big = f_sm = ImageFont.load_default()

import numpy as np
for j, (name, p, kind) in enumerate(COLS):
    x = GAP + j * (CELL + GAP)
    if kind == 'truth':
        dr.rectangle([x - 3, HEAD - 3, x + CELL + 3, HEAD + CELL + 3], outline='#0d9488', width=3)
    elif kind == 'ref':
        dr.rectangle([x - 3, HEAD - 3, x + CELL + 3, HEAD + CELL + 3], outline='#b91c1c', width=3)
    dr.text((x + 2, 4), name, fill='black', font=f_big)
    if not p.exists():
        dr.rectangle([x, HEAD, x + CELL, HEAD + CELL], outline='red')
        dr.text((x + 20, HEAD + CELL // 2), 'MISS', fill='red', font=f_big)
        continue
    im = Image.open(p).convert('L').resize((CELL, CELL), Image.LANCZOS)
    sheet.paste(im.convert('RGB'), (x, HEAD))
    a = np.array(im).astype(np.float32) / 255.
    ink = 100 * ((1 - a) > 0.5).mean()
    dr.text((x + 2, HEAD + CELL + 6), f'ink {ink:.1f}%', fill='#334155', font=f_sm)

dr.text((GAP, HEAD + CELL + 6), '  行 3 · 赵孟頫 / 火 · idx200=159, idx100=43   '
        '青框=真迹(与原始文件逐像素相同)  红框=「大」真迹参照', fill='#0f172a', font=f_sm)

dst = ROOT / 'assets/posters/row03_huo_vs_da.png'
sheet.save(dst)
print('wrote', dst, sheet.size)

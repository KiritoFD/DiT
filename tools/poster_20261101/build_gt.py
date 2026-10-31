"""Materialise the GT column from the ORIGINAL真迹 files named in the panel CSV.

The eval's own gt{i}.png was verified pixel-stat-identical to these originals, but
the moyi run's gt{i}.png is a thinned variant, so we never use that one again.
"""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'scratch/panels/poster_imgs/GT'
CELLS = ROOT / 'tools/poster_20261101/cells.json'

cells = json.loads(CELLS.read_text(encoding='utf-8'))['cells']
rows = list(csv.DictReader(open(ROOT / 'scratch/panels/validation_val100.csv', encoding='utf-8-sig')))
OUT.mkdir(parents=True, exist_ok=True)

ok, miss = 0, []
for c in cells:
    r = rows[c['idx100']]
    if (r['calligrapher'], r['character']) != (c['writer'], c['char']):
        raise SystemExit(f"panel mismatch at idx100={c['idx100']}: "
                         f"{r['calligrapher']}/{r['character']} != {c['writer']}/{c['char']}")
    src = ROOT / r['image_path']
    if not src.exists():
        miss.append((c['writer'], c['char'], str(src)))
        continue
    dst = OUT / f'g{c["idx100"]}.png'
    Image.open(src).convert('L').resize((256, 256), Image.LANCZOS).save(dst)
    ok += 1

print(f'wrote {ok}/{len(cells)} GT files to {OUT.relative_to(ROOT)}')
if miss:
    print('missing originals:')
    for m in miss:
        print('  ', m)


def st(p):
    a = np.array(Image.open(p).convert('L')).astype(np.float32) / 255.
    ink = (1.0 - a) > 0.5
    w = 2.0 * distance_transform_edt(ink)[ink]
    return 100 * ink.mean(), float(np.percentile(w, 50))


inks = [st(OUT / f'g{c["idx100"]}.png')[0] for c in cells if (OUT / f'g{c["idx100"]}.png').exists()]
print(f'GT ink%: mean={np.mean(inks):.2f} min={min(inks):.2f} max={max(inks):.2f}')

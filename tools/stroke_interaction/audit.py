"""F1 audit: draw the detected events on real glyphs so a human can check them.

The plan asks for a small-sample manual review before anything is measured at
scale, and for a same-character / same-stroke-region / different-writer plate so
numbers never stand alone. This produces both.

  palette.png  - a few glyphs per script with END / JUNCTION / TURN marked, so we
                 can see whether the detector lands on 起收笔 / 转折 / 交叉.
  compare.png  - the same character written by several writers, event cloud
                 overlaid, so the visual difference can be compared with the
                 numbers that F2 will report.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stroke_interaction.features import load_ink, extract_events  # noqa: E402

COLOR = {'end': (255, 60, 60), 'junction': (60, 120, 255), 'turn': (40, 200, 90)}


def draw_events(img_rel, rng, size=320):
    ink, binary = load_ink(ROOT / img_rel)
    events, _, _ = extract_events(ink, binary, rng)
    base = Image.fromarray((np.clip(1.0 - ink, 0, 1) * 255).astype(np.uint8)).convert('RGB')
    d = ImageDraw.Draw(base)
    for e in events:
        y, x = e['y'] * ink.shape[0], e['x'] * ink.shape[1]
        r = 4
        col = COLOR[e['kind']]
        d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=2)
    return base.resize((size, size), Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='exp-std/csv/train_top10_noaug.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--per-script', type=int, default=4)
    ap.add_argument('--compare-chars', default='為,焦,堅')
    a = ap.parse_args()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / a.csv, encoding='utf-8')))
    rng = np.random.default_rng(0)

    by_script = defaultdict(list)
    for r in rows:
        by_script[r['script']].append(r)

    # ---- palette: a few glyphs per script with events marked ----
    tiles = []
    labels = []
    for script in sorted(by_script):
        picks = by_script[script][:a.per_script]
        for r in picks:
            tiles.append(draw_events(r['image_path'], rng))
            labels.append(f"{r['calligrapher']}·{script}·{r['character']}")
    cols = len(tiles)
    sheet = Image.new('RGB', (320 * cols, 320 + 26), 'white')
    for i, t in enumerate(tiles):
        sheet.paste(t, (320 * i, 26))
    d = ImageDraw.Draw(sheet)
    for i, lab in enumerate(labels):
        d.text((320 * i + 6, 6), lab, fill=(0, 0, 0))
    sheet.save(out / 'audit_palette.png')
    print(f'[audit] wrote {out / "audit_palette.png"}  ({cols} tiles)', flush=True)
    print('[audit] legend: red=end(端点) blue=junction(连接) green=turn(转折)', flush=True)

    # ---- same character, different writers ----
    for ch in [c for c in a.compare_chars.split(',') if c]:
        got = defaultdict(list)
        for r in rows:
            if r['character'] == ch and r['script'] in ('楷', '行'):
                got[(r['script'], r['calligrapher'])].append(r)
        if len(got) < 2:
            continue
        entries = sorted(got.items())[:6]
        tiles, labels = [], []
        for (script, w), rs in entries:
            tiles.append(draw_events(rs[0]['image_path'], rng))
            labels.append(f'{w}·{script}')
        sheet = Image.new('RGB', (320 * len(tiles), 320 + 26), 'white')
        for i, t in enumerate(tiles):
            sheet.paste(t, (320 * i, 26))
        d = ImageDraw.Draw(sheet)
        d.text((6, 6), f'character {ch}', fill=(0, 0, 0))
        for i, lab in enumerate(labels):
            d.text((320 * i + 6, 6), lab, fill=(0, 0, 0))
        p = out / f'audit_compare_{ch}.png'
        sheet.save(p)
        print(f'[audit] wrote {p} ({len(tiles)} writers)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

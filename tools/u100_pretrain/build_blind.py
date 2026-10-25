"""Assemble a blinded paired-comparison sheet for stroke-quality judging.

For each of N stratified characters (5 per writer): [GT anchor][A][B] where A/B
are two models' generations in per-pair random order; the model identity lives
only in the key file. Judges score A and B on four criteria (1-5) and pick a
winner per criterion; the sheet itself never names the models.
"""
import argparse
import csv
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

WRITER_NAMES = {0: 'w0', 1: 'w1', 2: 'w2', 3: 'w3', 4: 'w4',
                5: 'w5', 6: 'w6', 7: 'w7', 8: 'w8', 9: 'w9'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a-samples', required=True)
    parser.add_argument('--b-samples', required=True)
    parser.add_argument('--meta-csv', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--per-writer', type=int, default=5)
    parser.add_argument('--seed', type=int, default=90)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.meta_csv, encoding='utf-8-sig')))
    by_writer = {}
    for idx, r in enumerate(rows):
        by_writer.setdefault(int(r['calligrapher_id']), []).append(idx)
    rng = random.Random(args.seed)
    chosen = []
    for w in sorted(by_writer):
        chosen.extend(sorted(rng.sample(by_writer[w], args.per_writer)))

    cell = 192; anchor = 128; pad = 8; head = 22
    per_page = 10
    key = []
    pages = (len(chosen) + per_page - 1) // per_page
    for p in range(pages):
        chunk = chosen[p * per_page:(p + 1) * per_page]
        H = head + len(chunk) * (cell + 30)
        page = Image.new('RGB', (anchor + 2 * cell + 4 * pad, H), (16, 16, 20))
        d = ImageDraw.Draw(page)
        d.text((6, 4), f'BLIND PAGE {p + 1}/{pages}  pair: [GT anchor][A][B]  '
                       'score 1-5: qibi(起收笔) thickness(粗细) topology(拓扑) style(风格)', fill=(230, 230, 150))
        y = head
        for k, idx in enumerate(chunk):
            r = rows[idx]
            gt = Image.open(Path(args.a_samples) / f'gt{idx}.png').convert('RGB').resize((anchor, anchor), Image.LANCZOS)
            a = Image.open(Path(args.a_samples) / f'g{idx}.png').convert('RGB').resize((cell, cell), Image.LANCZOS)
            b = Image.open(Path(args.b_samples) / f'g{idx}.png').convert('RGB').resize((cell, cell), Image.LANCZOS)
            left_is_a = rng.random() < 0.5
            cells = [gt, (a if left_is_a else b), (b if left_is_a else a)]
            x = pad
            for im in cells:
                page.paste(im, (x, y)); x += im.width + pad
            d.text((6, y + cell + 2),
                   f"pair {p * per_page + k:02d}  writer {WRITER_NAMES[int(r['calligrapher_id'])]}  char {r['character']}",
                   fill=(150, 225, 255))
            key.append(dict(pair=p * per_page + k, idx=idx, char=r['character'],
                            writer=int(r['calligrapher_id']),
                            A='a_model', B='b_model', left_is_A=left_is_a))
            y += cell + 30
        page.save(out_dir / f'blind_page_{p + 1}.png')
    (out_dir / 'key.json').write_text(json.dumps(
        dict(a=args.a_samples, b=args.b_samples, seed=args.seed, pairs=key), indent=2))
    print('BLIND_SHEET_READY', out_dir, 'pairs', len(chosen))


if __name__ == '__main__':
    main()

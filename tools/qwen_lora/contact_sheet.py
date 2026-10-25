"""Contact sheet of all test pairs: GT (plain) | generated (red border), grouped by writer."""
import json
import statistics as st
from pathlib import Path
import sys

root = Path('/home/ds/Workspace/xy/qwen-lora-top10-20261024')
OUT = root / 'test_eval_20k'
sys.path.insert(0, str(root / 'code'))
from test_eval import load_font, WRITER_EN  # noqa: E402

from PIL import Image, ImageDraw

CELL, GAP, PAIRS_PER_ROW = 128, 6, 5
pair_w = 2 * CELL + 4
W = PAIRS_PER_ROW * (pair_w + GAP) + 20
font, cjk = load_font(18)
small, _ = load_font(13)


def main():
    results = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    for r, row in zip(results, data['datasets']['test']):
        r['gt'] = row['file']
    by_writer = {}
    for r in results:
        by_writer.setdefault(r['calligrapher'], []).append(r)
    writers = sorted(by_writer)
    rows_total = sum(-(-len(by_writer[w]) // PAIRS_PER_ROW) for w in writers)
    H = 70 + len(writers) * 34 + rows_total * (CELL + 22) + 30
    canvas = Image.new('RGB', (W, H), '#f4f4f4')
    draw = ImageDraw.Draw(canvas)
    label = '左 GT ｜ 右 生成（红框）' if cjk else 'left GT | right GEN (red border)'
    draw.text((12, 10), f'test200 全量对照 · 20k LoRA · 256 · Turbo 8步 · seed90 · {label}',
              font=font, fill='black')
    y = 52
    for w in writers:
        items = sorted(by_writer[w], key=lambda r: (r['script'], r['character']))
        m = {f: st.mean(r[f] for r in items) for f in ('ssim', 'lpips', 'ink_iou')}
        wl = f'{w} ({WRITER_EN[w]})' if cjk else WRITER_EN[w]
        draw.text((12, y), f'{wl}  n={len(items)}  S {m["ssim"]:.3f}  L {m["lpips"]:.3f}  I {m["ink_iou"]:.3f}',
                  font=font, fill='black')
        y += 34
        for i, r in enumerate(items):
            col = i % PAIRS_PER_ROW
            x = 12 + col * (pair_w + GAP)
            if col == 0 and i > 0:
                pass
            gt = Image.open(root / r['gt']).convert('RGB').resize((CELL, CELL), Image.Resampling.LANCZOS)
            pred = Image.open(OUT / 'images' / r['image']).convert('RGB').resize(
                (CELL, CELL), Image.Resampling.LANCZOS)
            canvas.paste(gt, (x, y))
            canvas.paste(pred, (x + CELL + 4, y))
            draw.rectangle((x + CELL + 4, y, x + 2 * CELL + 3, y + CELL - 1), outline='#c0392b', width=2)
            draw.text((x + 2, y + CELL + 2),
                      f'{r["character"]}{r["script"]} S{r["ssim"]:.2f}', font=small, fill='#333')
            if col == PAIRS_PER_ROW - 1 or i == len(items) - 1:
                y += CELL + 22
        y += 8
    canvas.save(OUT / 'contact_sheet.jpg', quality=72, optimize=True)
    print('CONTACT_SHEET_READY', OUT / 'contact_sheet.jpg', 'cjk', cjk, flush=True)


if __name__ == '__main__':
    main()

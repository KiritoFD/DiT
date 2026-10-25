"""Model-comparison matrix: 20 test characters as columns, one model per row."""
import json
import sys
from pathlib import Path

root = Path('/home/ds/Workspace/xy/qwen-lora-top10-20261024')
base_root = Path('/home/ds/Workspace/xy/qwen-lora-top10-base-20261024')
sys.path.insert(0, '/home/ds/Workspace/DiT')
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(root / 'vendor'))

from PIL import Image, ImageDraw, ImageFont

OUT = root / 'lora_line_poster'
OUT.mkdir(exist_ok=True)
FONT = '/home/ds/Workspace/DiT/tools/fonts/NotoSerifSC-VF.ttf'
data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
test = data['datasets']['test']

C, G, PAD, LABEL_W = 88, 4, 16, 170
NCOLS = 20
W = PAD * 2 + LABEL_W + NCOLS * C + (NCOLS - 1) * G
NAVY, RED, GRAY = (23, 55, 110), (180, 40, 30), (96, 96, 96)
f_t = ImageFont.truetype(FONT, 30)
f_h = ImageFont.truetype(FONT, 19)
f_r = ImageFont.truetype(FONT, 16)
f_s = ImageFont.truetype(FONT, 12)

turbo = json.loads((root / 'test_eval_20k' / 'metrics.json').read_text(encoding='utf-8'))
base_metrics = json.loads((root / 'base_ref_diag' / 'metrics_noref.json').read_text(encoding='utf-8'))

# 20 evenly spaced test indices spanning all 10 writers x 3 scripts as available
IDX = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]


def img_of(path):
    return Image.open(path).convert('RGB').resize((C, C), Image.Resampling.LANCZOS)


def main():
    H = 54 + 26 + NCOLS * 0 + len(IDX) * 0 + 30  # computed below
    rows = [
        ('GT（真迹）', GRAY, lambda i: img_of(root / test[i]['file']),
         None),
        ('Turbo 底模 8步', GRAY, lambda i: img_of(root / 'base_ref_diag' / 'images' / f'noref_{i:03d}_'
                                                 f'{test[i]["calligrapher"]}_{test[i]["script"]}_'
                                                 f'{test[i]["character"]}.png'),
         {f: round(sum(r[f] for r in base_metrics) / len(base_metrics), 4)
          for f in ('ssim', 'lpips', 'ink_iou')}),
        ('Turbo LoRA 20k 8步', NAVY, lambda i: img_of(root / 'test_eval_20k' / 'images' / f'{i:03d}_'
                                                     f'{test[i]["calligrapher"]}_{test[i]["script"]}_'
                                                     f'{test[i]["character"]}.png'),
         {f: round(sum(r[f] for r in turbo) / len(turbo), 4)
          for f in ('ssim', 'lpips', 'ink_iou')}),
        ('非Turbo LoRA 20k 40步', GRAY, lambda i: img_of(base_root / 'test_eval_20k' / 'images' / f'{i:03d}_'
                                                        f'{test[i]["calligrapher"]}_{test[i]["script"]}_'
                                                        f'{test[i]["character"]}.png'),
         None),
        ('Turbo LoRA +2参考图', RED, lambda i: img_of(root / 'ref_eval_20k' / 'images' / f'{i:03d}_'
                                                     f'{test[i]["calligrapher"]}_{test[i]["script"]}_'
                                                     f'{test[i]["character"]}.png'),
         dict(ssim=.5705, lpips=.4453, ink_iou=.2576)),
    ]
    head_h = 54 + 3 * 27
    row_h = C + 20
    H = head_h + len(rows) * row_h + 90
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 12), '同一批 20 字 × 各模型一行（test200 抽样，seed90）', font=f_t, fill=NAVY)
    y = 54
    d.text((PAD, y), '列 = 字（书家·书体），行 = 模型；量级：LoRA 使 LPIPS −13%、Ink +45% vs 底模；'
                     '最后一行可见参考图污染（目标字被写成参考字）', font=f_r, fill=(60, 60, 60))
    y += 26

    # column headers: character + writer·script
    for j, i in enumerate(IDX):
        x = PAD + LABEL_W + j * (C + G)
        r = test[i]
        d.text((x + 2, y), f'{r["character"]}', font=ImageFont.truetype(FONT, 20), fill=(0, 0, 0))
        d.text((x + 2, y + 24), f'{r["calligrapher"][:2]}·{r["script"]}', font=f_s, fill=GRAY)
    y += 50

    for name, color, getter, mean in rows:
        d.text((PAD, y + C // 2 - 22), name, font=f_r, fill=color)
        if mean:
            d.text((PAD, y + C // 2 + 2), f'S {mean["ssim"]:.3f}', font=f_s, fill=color)
            d.text((PAD, y + C // 2 + 18), f'L {mean["lpips"]:.3f}', font=f_s, fill=color)
            d.text((PAD, y + C // 2 + 34), f'I {mean["ink_iou"]:.3f}', font=f_s, fill=color)
        for j, i in enumerate(IDX):
            x = PAD + LABEL_W + j * (C + G)
            try:
                canvas.paste(getter(i), (x, y))
            except FileNotFoundError:
                d.rectangle((x, y, x + C - 1, y + C - 1), outline=(210, 210, 210), width=1)
        d.rectangle((PAD + LABEL_W, y, x + C - 1, y + C - 1), outline=(225, 225, 225), width=1)
        y += row_h

    d.text((PAD, y + 6), '非Turbo 行均值（40步全量 test200）：SSIM .5758 / LPIPS .4299 / Ink .2791；'
                         'Turbo LoRA +参考行均值：SSIM .5705 / LPIPS .4453 / Ink .2576（配对全降，且目标字可能被参考污染）',
           font=f_r, fill=GRAY)
    y += 40
    d.text((PAD, y), 'GT 列采样自 test200 十书家；完整 200 对对照见 test200_contact_sheet_20k.jpg', font=f_s, fill=GRAY)

    canvas = canvas.crop((0, 0, W, y + 30))
    canvas.save(OUT / 'model_matrix_poster.jpg', quality=82, optimize=True)
    print('MATRIX_READY', canvas.size)


if __name__ == '__main__':
    main()

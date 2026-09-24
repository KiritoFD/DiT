#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重画 eval poster，在 GT 行下面**再加两行训练集里的"类似条件" GT**。

为什么有用: 看 eval 样本时, 光有 GT 无法判断"模型写不好"是
  ① 不会这个字的结构, 还是 ② 不会这个书家的笔法。
加两行参照就能分开:
  · 同书家·异字 (训练集)  -> 这个书家的笔法长什么样
  · 同字·异书家 (训练集)  -> 这个字的结构有多少种写法

行序: input(标准字 g) / 每个 step 的 gen / GT / 同书家异字 / 同字异书家
列序: 与评测 csv 行序一致（make_eval_cache 按 [:n] 取行, 与 poster 列一一对应）

用法:
  python tools/poster_with_train_refs.py <results_dir> [--set strict] [--train-csv ...]
                                         [--sets-spec "seen:csv:20,strict:csv:249"]
"""
import argparse
import csv
import glob
import os
import re
import sys

from PIL import Image, ImageDraw

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument('results_dir')
ap.add_argument('--set', dest='setname', default='strict')
ap.add_argument('--train-csv', default='')
ap.add_argument('--eval-csv', default='')
ap.add_argument('--n', type=int, default=0)
ap.add_argument('--cell', type=int, default=224)
ap.add_argument('--max-cols', type=int, default=0,
                help='>0 只画前 N 列（便于看清新增行）')
ap.add_argument('--gap', type=int, default=6)
ap.add_argument('--out', default='')
a = ap.parse_args()

# ── 从 resolved_config.json 补全缺的参数 ─────────────────────────────
cfg = {}
_cf = glob.glob(os.path.join(a.results_dir, '*', 'resolved_config.json'))
if _cf:
    import json
    cfg = json.load(open(sorted(_cf)[-1], encoding='utf-8'))
train_csv = a.train_csv or cfg.get('data_csv', 'assets/train_50k_v2_fixed.csv')
if not a.eval_csv or not a.n:
    spec = cfg.get('in_mem_eval_sets', '')
    for part in spec.split(','):
        bits = part.split(':')
        if len(bits) == 3 and bits[0] == a.setname:
            a.eval_csv = a.eval_csv or bits[1]
            a.n = a.n or int(bits[2])
print(f'results_dir = {a.results_dir}')
print(f'set={a.setname}  eval_csv={a.eval_csv}  n={a.n}')
print(f'train_csv   = {train_csv}')

# ── 读评测行（列序 = poster 列序）────────────────────────────────────
ev = list(csv.DictReader(open(a.eval_csv, encoding='utf-8')))
if a.n:
    ev = ev[:a.n]
print(f'评测行 {len(ev)} 条')

# ── 训练集索引 ──────────────────────────────────────────────────────
tr = list(csv.DictReader(open(train_csv, encoding='utf-8')))
by_cal, by_char = {}, {}
for r in tr:
    if not r.get('image_path'):
        continue
    by_cal.setdefault((str(r.get('calligrapher_id', '')), str(r.get('script_id', ''))),
                      []).append(r)
    by_cal.setdefault(('*', str(r.get('calligrapher_id', ''))), []).append(r)
    by_char.setdefault(str(r.get('character', '')), []).append(r)
print(f'训练集 {len(tr)} 条；书家组 {len(by_cal)}，字组 {len(by_char)}')


def pick_same_callig(other_char, cal_id, script_id, k):
    """同书家、异字。优先同书体。"""
    for key in [(str(cal_id), str(script_id)), ('*', str(cal_id))]:
        cands = [r for r in by_cal.get(key, [])
                 if str(r.get('character', '')) != str(other_char)]
        if cands:
            return cands[k % len(cands)]
    return None


def pick_same_char(ch, cal_id, k):
    """同字、异书家。"""
    cands = [r for r in by_char.get(str(ch), [])
             if str(r.get('calligrapher_id', '')) != str(cal_id)]
    if not cands:
        cands = [r for r in by_char.get(str(ch), []) if str(r.get('calligrapher_id', '')) != str(cal_id)]
    return cands[k % len(cands)] if cands else None


def img_of(row):
    if row is None:
        return None
    p = row.get('image_path', '')
    if not p:
        return None
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    return p if os.path.exists(p) else None


# ── 扫描 step 目录 ──────────────────────────────────────────────────
base = os.path.join(a.results_dir, 'eval_samples_ctrl')
sub = 'g' if a.setname in ('seen', 'g') else a.setname
steps = []
for d in sorted(glob.glob(os.path.join(base, 'step*'))):
    n = 0
    while os.path.exists(os.path.join(d, sub, f'g{n}.png')):
        n += 1
    if n:
        steps.append((int(re.search(r'step(\d+)', os.path.basename(d)).group(1)),
                      os.path.join(d, sub), n))
if not steps:
    raise SystemExit(f'✗ {base} 下没有 step*/{sub}/g*.png')
n_max = max(s[2] for s in steps)
if a.max_cols:
    n_max = min(n_max, a.max_cols)
print(f'{len(steps)} 个 step, 最多 {n_max} 列')

input_dir = os.path.join(base, f'{a.setname}_input_g')
has_input = os.path.isdir(input_dir) and bool(glob.glob(os.path.join(input_dir, 'g*.png')))

cell = max(64, min(a.cell, (2600 if a.max_cols else 1400) // max(n_max, 1)))
gap, label_h, hdr_h = a.gap, max(40, cell // 4), 56


def _cell(path, bg):
    if path and os.path.exists(path):
        return Image.open(path).convert('RGB').resize((cell, cell), Image.LANCZOS)
    return Image.new('RGB', (cell, cell), bg)


def _font(sz, bold=False):
    cands = ['/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
             '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for c in cands:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, sz)
            except Exception:
                pass
    return ImageFont.load_default()


from PIL import ImageFont  # noqa: E402

font = _font(max(14, cell // 9))
font_big = _font(max(24, cell // 4), bold=True)

extra_rows = [
    ('训练集·同书家异字', lambda i, r: img_of(pick_same_callig(
        r.get('character'), r.get('calligrapher_id'), r.get('script_id'), i))),
    ('训练集·同字异书家', lambda i, r: img_of(pick_same_char(
        r.get('character'), r.get('calligrapher_id'), i))),
]

W = cell * n_max + gap * 2
n_rows = (len(steps) + 1 + (1 if has_input else 0) + len(extra_rows))
H = hdr_h + gap + n_rows * (label_h + cell + gap) + gap + 30
canvas = Image.new('RGB', (W, H), (15, 17, 22))
draw = ImageDraw.Draw(canvas)
y = gap
draw.rectangle([0, y, W, y + hdr_h], fill=(0, 0, 0))
draw.text((gap, y + 12),
          f'{a.setname} (n={n_max}) — input g / per-ckpt gen / GT / 训练集类似条件 GT',
          font=font, fill=(255, 200, 120))
y += hdr_h + gap

if has_input:
    draw.text((gap, y + 8), 'input (标准字 g)', font=font_big, fill=(120, 220, 255))
    y += label_h
    for i in range(n_max):
        canvas.paste(_cell(os.path.join(input_dir, f'g{i}.png'), (30, 40, 60)),
                     (gap + i * cell, y))
    y += cell + gap

for step, d, n in steps:
    draw.text((gap, y + 8), f'step {step}', font=font_big, fill=(160, 200, 255))
    y += label_h
    for i in range(n):
        canvas.paste(_cell(os.path.join(d, f'g{i}.png'), (40, 40, 40)), (gap + i * cell, y))
    y += cell + gap

gt_dir = steps[-1][1]
draw.text((gap, y + 8), 'GT (评测样本真迹)', font=font_big, fill=(255, 160, 160))
y += label_h
for i in range(n_max):
    canvas.paste(_cell(os.path.join(gt_dir, f'gt{i}.png'), (50, 50, 50)), (gap + i * cell, y))
y += cell + gap

for label, fn in extra_rows:
    draw.text((gap, y + 8), label, font=font_big, fill=(170, 255, 170))
    y += label_h
    for i in range(n_max):
        r = ev[i] if i < len(ev) else {}
        canvas.paste(_cell(fn(i, r), (25, 25, 25)), (gap + i * cell, y))
    y += cell + gap

out = a.out or os.path.join(a.results_dir, 'posters', f'{a.setname}_poster_withtrain.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
canvas.save(out)
print(f'-> {out}  ({canvas.size[0]}x{canvas.size[1]})')

# 顺带打印一份对照文本，便于核对
print('\n列  评测样本(字/书家/书体)          -> 同书家异字        同字异书家')
for i, r in enumerate(ev[:n_max]):
    r1 = pick_same_callig(r.get('character'), r.get('calligrapher_id'), r.get('script_id'), i)
    r2 = pick_same_char(r.get('character'), r.get('calligrapher_id'), i)
    f = lambda x: (f'{x.get("character","?")}/{x.get("calligrapher","?")[:6]}' if x else '—')
    print(f'{i:>3}  {r.get("character","?")}/{str(r.get("calligrapher","?"))[:6]}/'
          f'{r.get("script","?")}  -> {f(r1):<16} {f(r2)}')

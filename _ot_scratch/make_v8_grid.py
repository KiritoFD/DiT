import os, glob, re
from PIL import Image, ImageDraw

os.makedirs('/tmp/v8_montages', exist_ok=True)

def ids_in(d):
    out = set()
    for f in os.listdir(d):
        m = re.match(r'gt_(\d+)\.png', f)
        if m:
            out.add(m.group(1))
    return out

common = sorted(ids_in('/tmp/grid_s30_130k') & ids_in('/tmp/grid_v8a_95k'))
print('common ids:', common)

def load(id_):
    g_old = '/tmp/grid_s30_130k/gt_%s.png' % id_
    s_old = '/tmp/grid_s30_130k/sample_%s.png' % id_
    g_new = '/tmp/grid_v8a_95k/gt_%s.png' % id_
    s_new = '/tmp/grid_v8a_95k/sample_%s.png' % id_
    return g_old, s_old, g_new, s_new

LABELS = ['GT', 'S30-old@130k', 'v8a@95k']
LABEL_H = 26

def one_row(g, s_old, s_new):
    ims = [Image.open(g).convert('RGB'), Image.open(s_old).convert('RGB'), Image.open(s_new).convert('RGB')]
    w = sum(im.size[0] for im in ims)
    h = ims[0].size[1] + LABEL_H
    canvas = Image.new('RGB', (w, h), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    x = 0
    for lab, im in zip(LABELS, ims):
        d.text((x + 6, 5), lab, fill=(20, 20, 20))
        canvas.paste(im, (x, LABEL_H))
        x += im.size[0]
    return canvas

rows = []
for id_ in common:
    g_old, s_old, g_new, s_new = load(id_)
    if not all(os.path.exists(p) for p in [g_old, s_old, g_new, s_new]):
        print('skip', id_); continue
    r = one_row(g_old, s_old, g_new)
    rows.append(r)
    r.save('/tmp/v8_montages/row_%s.png' % id_)
    print('row_%s saved %s' % (id_, r.size))

# 总 grid: 4列 x 2行
if rows:
    cols = 4
    n = len(rows)
    nrows = (n + cols - 1) // cols
    cw = max(r.size[0] for r in rows)
    ch = max(r.size[1] for r in rows)
    grid = Image.new('RGB', (cols * cw, nrows * ch), (250, 250, 250))
    for i, r in enumerate(rows):
        grid.paste(r, ((i % cols) * cw, (i // cols) * ch))
    grid.save('/tmp/v8_montages/grid_all.png')
    print('grid_all saved', grid.size)
print('DONE')
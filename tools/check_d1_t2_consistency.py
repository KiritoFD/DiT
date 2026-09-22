# -*- coding: utf-8 -*-
"""检查 D1（dmod）与 T2（风格可分性）是否互相印证。"""
import json, io, glob, os, math

d1 = {}
for f in glob.glob('assets/d1_remote/assets/d1_*.json'):
    d1[os.path.basename(f)[3:-5]] = json.load(io.open(f, encoding='utf-8'))
t2r = {}
for f in glob.glob('assets/t2_remote/assets/t2_*.json'):
    base = os.path.basename(f)[3:-5].rsplit('__', 1)[0]
    d = json.load(io.open(f, encoding='utf-8'))
    rec = {}
    for r in d:
        tt = r['tag']
        if '生成图' in tt and 'style' in tt: rec['gs'] = r
        elif '生成图' in tt and 'dino' in tt: rec['gd'] = r
        elif 'GT' in tt and 'style' in tt: rec['ts'] = r
        elif 'GT' in tt and 'dino' in tt: rec['td'] = r
    t2r[base] = rec

PAIR = [('v13_base_50k', 'v13_base_155k'), ('v13_12ch_post', 'v13_12ch_225k'),
        ('v13_wd01', 'v13_wd01_125k'), ('v15a_multistyle_k4', 'v15a_150k'),
        ('v15b_supcon', 'v15b_supcon_70k'), ('v15c_fixed', 'v15c_fixed_210k')]

dm, cen, lin, cenD, linD = [], [], [], [], []
print('%-22s %8s %8s %8s %8s %8s' % ('run', 'dmod', '质心/GT', '线性/GT', 'Dc/GT', 'Dc线性/GT'))
print('-' * 74)
for k, b in PAIR:
    rec = t2r[k]; dd = d1[b]
    gs, gd, ts, td = rec['gs'], rec['gd'], rec['ts'], rec['td']
    r1 = gs['acc_centroid'] / ts['acc_centroid']
    r2 = gs['acc_logreg'] / ts['acc_logreg']
    r3 = gd['acc_centroid'] / td['acc_centroid']
    r4 = gd['acc_logreg'] / td['acc_logreg']
    dm.append(dd['delta_mod_mean']); cen.append(gs['acc_centroid'])
    lin.append(gs['acc_logreg']); cenD.append(gd['acc_centroid']); linD.append(gd['acc_logreg'])
    print('%-22s %8.4f %8.3f %8.3f %8.3f %8.3f' % (k, dd['delta_mod_mean'], r1, r2, r3, r4))


def pear(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a); vb = sum((x - mb) ** 2 for x in b)
    if va < 1e-12 or vb < 1e-12: return float('nan')
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


print()
for nm, a, b in [('dmod ~ T2质心  ', dm, cen), ('dmod ~ T2线性  ', dm, lin),
                 ('dmod ~ DINO质心', dm, cenD), ('dmod ~ DINO线性', dm, linD),
                 ('dmod ~ GT质心  ', dm, [0, 0, 0, 0, 0, 0])]:
    if nm.endswith('GT质心  '):
        continue
    print('%s  pearson r = %+.3f' % (nm, pear(a, b)))

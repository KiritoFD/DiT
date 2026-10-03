#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep diagnosis from per-sample eval batch csvs (v22 / v24fs / v25)."""
import csv, glob, statistics as st, collections, os

EXPS = ['v22_aug_skelnet_200k', 'v21_skelnet_200k', 'v23_splitnorm',
        'v24_frozenskel', 'v25_stdskel']
NUM = ['ssim','ink_ssim','ink_iou','skel_iou','frag_ratio','hole_pred','hole_gt','mse','lpips']

def load(exp):
    f = f'assets/results/{exp}/eval_stdskel_batch.csv'
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        for k in NUM:
            try: r[k] = float(r[k])
            except Exception: r[k] = None
    return rows

def pct(v, p):
    if not v: return float('nan')
    v = sorted(v); i = min(len(v)-1, int(round(p/100*(len(v)-1))))
    return v[i]

for exp in EXPS:
    rows = load(exp)
    if not rows:
        print(f'== {exp}: no batch csv'); continue
    steps = sorted({int(r['step']) for r in rows})
    sets = sorted({r['set'] for r in rows})
    print(f'\n===== {exp}  steps {steps[0]}..{steps[-1]} ({len(steps)}) sets={sets} =====')
    for s in sets:
        sub = [r for r in rows if r['set'] == s]
        sp = max({int(r['step']) for r in sub})
        last = [r for r in sub if int(r['step']) == sp]
        n = len(last)
        g = lambda k: [r[k] for r in last if r[k] is not None]
        ss = g('ssim'); ik = g('ink_ssim'); iou = g('ink_iou'); frag = g('frag_ratio')
        print(f'  [{s}] latest step={sp} n={n} ssim={st.mean(ss):.4f} '
              f'ink_ssim={st.mean(ik):.4f}(p10={pct(ik,10):.3f} p50={pct(ik,50):.3f} p90={pct(ik,90):.3f}) '
              f'ink_iou={st.mean(iou):.4f} frag={st.mean(frag):.2f} '
              f'hole={st.mean(g("hole_pred")):.3f}/{st.mean(g("hole_gt")):.3f}')
        bad = [r for r in last if r['ink_iou'] is not None and r['ink_iou'] < 0.05]
        hor = [r for r in last if r['ink_ssim'] is not None and r['ink_ssim'] < 0.15]
        print(f'       灾难样本 ink_iou<0.05: {len(bad)}/{n} ({100*len(bad)/max(n,1):.0f}%)   '
              f'ink_ssim<0.15: {len(hor)}/{n} ({100*len(hor)/max(n,1):.0f}%)')
        if s == 'strict' and len(last) >= 20:
            byc = collections.defaultdict(list); byk = collections.defaultdict(list)
            for r in last:
                if r['ink_ssim'] is not None:
                    byc[r['char']].append(r['ink_ssim'])
                    byk[r['calligrapher']].append(r['ink_ssim'])
            print('       最差字(ink_ssim):', ', '.join(
                f"{c}:{st.mean(v):.2f}({len(v)})" for c, v in
                sorted(byc.items(), key=lambda kv: st.mean(kv[1]))[:8]))
            print('       最好字(ink_ssim):', ', '.join(
                f"{c}:{st.mean(v):.2f}({len(v)})" for c, v in
                sorted(byc.items(), key=lambda kv: -st.mean(kv[1]))[:6]))
            print('       书家(ink_ssim):', ', '.join(
                f"{c}:{st.mean(v):.2f}" for c, v in
                sorted(byk.items(), key=lambda kv: -st.mean(kv[1]))))

print('\n===== 轨迹: strict ink_ssim / ink_iou / frag 随步数 =====')
for exp in EXPS:
    rows = load(exp)
    if not rows: continue
    sub = [r for r in rows if r['set'] == 'strict']
    if not sub: continue
    steps = sorted({int(r['step']) for r in sub})
    sel = steps[::max(1, len(steps)//8)]
    if steps[-1] not in sel: sel.append(steps[-1])
    for sp in sel:
        last = [r for r in sub if int(r['step']) == sp]
        g = lambda k: [r[k] for r in last if r[k] is not None]
        if not g('ssim'): continue
        print(f'{exp:24s} {sp:>7d} n={len(last):3d} ssim={st.mean(g("ssim")):.4f} '
              f'ink={st.mean(g("ink_ssim")):.4f} iou={st.mean(g("ink_iou")):.4f} frag={st.mean(g("frag_ratio")):.2f}')

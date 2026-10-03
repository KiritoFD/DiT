#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 5: v17 vs v18 (style-rank verdict), v21 vs v22 ink plateau,
   seen/strict gap, eval-set break, top10 trainable-skelnet collapse."""
import csv, os, re, glob, collections, statistics as st

LOGS = {
 'v17_inj3_fixed_aug': ['logs/v17_series/*.log', 'logs/v17*fixed_aug*.log'],
 'v18_style_rank':     ['logs/v18_series/*.log'],
 'v21':                ['logs/v21_skelnet_200k.log'],
 'v22':                ['logs/v22_aug_skelnet_200k.log'],
 'v24top10':           ['logs/v24_top10_style23.log'],
 'v24fs':              ['logs/v24_frozenskel.log'],
}
PAT = re.compile(r'\[in-mem-eval\] step=(\d+) set=(\w+) n=(\d+) ssim=([\d.]+).*?ink_ssim=([\d.]+) frag=([\d.]+).*?nn=([\d.]+) tgt_spec=([+\-\d.]+) cal_enrich=([\d.]+)x')

def traj(tag):
    out = {}
    for g in LOGS[tag]:
        for fp in glob.glob(g):
            try: txt = open(fp, errors='ignore').read()
            except Exception: continue
            for m in PAT.finditer(txt):
                out[(m.group(2), int(m.group(1)))] = dict(
                    n=int(m.group(3)), ssim=float(m.group(4)), ink=float(m.group(5)),
                    frag=float(m.group(6)), nn=float(m.group(7)), spec=float(m.group(8)),
                    enr=float(m.group(9)))
    return out

print('==== A. style-rank 判决: v17(无) vs v18(w=0.005) 同步长 strict ====')
a, b = traj('v17_inj3_fixed_aug'), traj('v18_style_rank')
if not a:
    print('  (v17 日志缺失, 用文档记录: strict 0.5454@100k, seen 0.5818)')
steps = sorted({k[1] for k in b if k[0]=='strict'})
for sp in steps[::6][:-1] + [steps[-1]]:
    r = b.get(('strict', sp)); s = b.get(('seen', sp))
    if r: print(f'  v18 strict {sp:>7d} ssim={r["ssim"]:.4f} ink={r["ink"]:.4f} enrich={r["enr"]:.2f} spec={r["spec"]:+.4f} frag={r["frag"]:.2f}')
for sp in steps[::12] + [steps[-1]]:
    s = b.get(('seen', sp))
    if s: print(f'  v18 seen   {sp:>7d} ssim={s["ssim"]:.4f} ink={s["ink"]:.4f} enrich={s["enr"]:.2f} spec={s["spec"]:+.4f}')

print('\n==== B. seen − strict 差 (ink_ssim) 各实验 ====')
B = 'assets/results/{}/eval_stdskel_batch.csv'
def m_last(exp, s, k):
    f = B.format(exp)
    if not os.path.exists(f): return None, None
    rs = [r for r in csv.DictReader(open(f)) if r['set'] == s]
    if not rs: return None, None
    sp = max(int(r['step']) for r in rs)
    cur = [float(r[k]) for r in rs if int(r['step']) == sp and r[k] not in ('', None)]
    return st.mean(cur), sp
for exp in ['v18_style_rank_200k','v21_skelnet_200k','v22_aug_skelnet_200k','v23_splitnorm',
            'v24_frozenskel','v25_stdskel']:
    si, ssp = m_last(exp, 'seen', 'ink_ssim')
    ti, tsp = m_last(exp, 'strict', 'ink_ssim')
    su, _ = m_last(exp, 'seen', 'ink_iou')
    tu, _ = m_last(exp, 'strict', 'ink_iou')
    if si and ti:
        print(f'  {exp:24s} seen@{ssp:>6d} ink={si:.4f} iou={su:.3f} | strict@{tsp:>6d} ink={ti:.4f} iou={tu:.3f} | Δink={si-ti:+.4f} iou比={su/tu:.2f}x')

print('\n==== C. v24_top10(trainable SkelNet) 崩在哪: cfg 差异 ====')
import json
c1 = json.load(open('src/train/configs/v24_top10_style23.json'))
c2 = json.load(open('src/train/configs/v24_frozenskel.json'))
for k in sorted(set(c1) | set(c2)):
    v1, v2 = c1.get(k, '<abs>'), c2.get(k, '<abs>')
    if str(v1) != str(v2):
        print(f'  {k}: top10={str(v1)[:60]} | frozenskel={str(v2)[:60]}')

print('\n==== D. eval 汇总断档 (表头不匹配次数) ====')
for d in sorted(glob.glob('assets/results/*/')):
    n = len(glob.glob(d + 'eval_stdskel_summary.csv.bak_oldcols*'))
    if n: print(f'  {d}: 汇总表被重置 {n} 次')

print('\n==== E. v22 strict ink_ssim 峰值 vs 终点 (是否后期退化) ====')
f = B.format('v22_aug_skelnet_200k')
rs = [r for r in csv.DictReader(open(f)) if r['set'] == 'strict']
bys = collections.defaultdict(list)
for r in rs: bys[int(r['step'])].append(float(r['ink_ssim']))
best = max(bys.items(), key=lambda kv: st.mean(kv[1]))
last = max(bys)
print(f'  峰值 ink_ssim={st.mean(best[1]):.4f}@{best[0]}  终点={st.mean(bys[last]):.4f}@{last}  '
      f'峰值后 {"退化" if st.mean(bys[last]) < st.mean(best[1]) - 0.001 else "未退化"}')
for sp in sorted(bys)[::4] + [last]:
    cur = [r for r in rs if int(r['step']) == sp]
    fr = st.mean([float(r['frag_ratio']) for r in cur])
    iu = st.mean([float(r['ink_iou']) for r in cur])
    print(f'    {sp:>7d} ink={st.mean(bys[sp]):.4f} iou={iu:.4f} frag={fr:.2f}')

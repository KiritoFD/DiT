# -*- coding: utf-8 -*-
"""远程: 查 s26 训练速度与进度, 估算实验规模."""
import os, glob, re
os.chdir('/root/Workspace/xy/DiT')

cands = ['assets/results/s26_ctrl_gt_skel/train.log',
         'assets/results/s26_ctrl_gt_skel/s26_tmux.log']
cands += sorted(glob.glob('assets/results/s26_ctrl_gt_skel/*/log.txt'),
                key=os.path.getmtime, reverse=True)
cands = [c for c in cands if os.path.isfile(c)]
if not cands:
    print('NO LOG')
    raise SystemExit

pat = re.compile(r'step[:\s=]+(\d+)', re.I)
ploss = re.compile(r'loss[:\s=]+([\d.]+)', re.I)
for L in cands[:3]:
    print('=' * 60)
    print('LOG:', L, os.path.getsize(L), 'bytes')
    lines = open(L, errors='replace').read().splitlines()
    print('lines:', len(lines))
    hits = []
    for ln in lines:
        m = pat.search(ln)
        if m:
            lm = ploss.search(ln)
            hits.append((int(m.group(1)), float(lm.group(1)) if lm else None, ln[:160]))
    print('step-hits:', len(hits))
    if not hits:
        print('  tail:')
        for ln in lines[-6:]:
            print('   ', ln[:160])
        continue
    print('--- last 6 ---')
    for s, lo, ln in hits[-6:]:
        print(f'  step={s} loss={lo}')
        print(f'     | {ln}')
    if len(hits) >= 2:
        s0 = hits[0][0]
        s1 = hits[-1][0]
        ct, mt = os.path.getctime(L), os.path.getmtime(L)
        dt = mt - ct
        print('---')
        print(f'  step {s0} -> {s1} ({s1-s0} steps), 文件存续 {dt/3600:.2f} h')
        if dt > 60 and s1 > s0:
            sp = (s1 - s0) / dt
            print(f'  速度 {sp:.4f} steps/s = {sp*3600:.0f} steps/h')
            for t in (10000, 30000, 50000, 100000):
                print(f'    {t:>7d} steps ~ {t/sp/3600:6.1f} h'
                      f'  ({t/sp/86400:.1f} d)')
#!/bin/bash
# v85A: style_gain / style_ada_up 的**趋势**（全部 ckpt ✓）
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
$PY - <<'PY'
import glob, os, re, torch
cks = []
for p in glob.glob('assets/results/v85A_style_ada/*/checkpoints/*.pt'):
    m = re.search(r'(\d+)\.pt$', os.path.basename(p))
    if m:
        cks.append((int(m.group(1)), p))
cks.sort()
print('  共 %d 个 ckpt: %s' % (len(cks), [s for s, _ in cks]))
print('  %8s %10s %10s %10s | %10s %10s %10s' %
      ('step', 'g8', '|w|8', 'g10', '|w|10', 'g11', '|w|11'))
for s, p in cks:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck.get('ema') or ck.get('model') or {}
    row = []
    for i in (8, 10, 11):
        g = None
        w = None
        for k, v in sd.items():
            if k.endswith('blocks.%d.style_gain' % i):
                g = float(v.float())
            elif k.endswith('blocks.%d.style_ada_up.weight' % i):
                w = float(v.float().norm())
        row += [g, w]
    print('  %8d %10.6f %10.6f %10.6f | %10.4f %10.4f %10.4f' % (s, *row))
    del ck, sd
PY

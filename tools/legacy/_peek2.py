# -*- coding: utf-8 -*-
import json
d=json.load(open('G:/GitHub/DiT/tools/train_data.json',encoding='utf-8'))
rows=d['rows']
print('source:', d.get('source'))
print('total rows:', len(rows))
trs=[r for r in rows if not r.get('is_eval')]
evs=[r for r in rows if r.get('is_eval')]
print('train rows:', len(trs), 'eval rows:', len(evs))
if trs:
    last=trs[-1]
    print('last train: step=%d diff=%s ts=%s' % (last['step'], last.get('diff'), last.get('ts')))
if evs:
    print('evals:')
    for r in evs:
        print('  step=%d mse=%s ssim=%s' % (r['step'], r.get('mse'), r.get('ssim')))
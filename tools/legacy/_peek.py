# -*- coding: utf-8 -*-
import json
d=json.load(open('train_data.json',encoding='utf-8'))
rows=d['rows']
print('=== current source (live, s8) ===')
print(d.get('source'))
print('rows:',len(rows))
trs=[r for r in rows if not r.get('is_eval')]
for r in trs[-6:]:
    print('  step=%7d diff=%s sps=%s mem=%s/%s ts=%s' % (r['step'], r.get('diff'), r.get('stepsPerSec'), r.get('memCur'), r.get('memPeak'), r.get('ts')))

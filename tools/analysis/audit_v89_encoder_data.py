"""CPU-only cache and split provenance inspection before encoder training."""
import csv
from collections import Counter, defaultdict
import numpy as np
import torch

d = np.load('scratch/disc_lat_full.npz', allow_pickle=True)
print('CACHE', {k: [list(d[k].shape), str(d[k].dtype)] for k in d.files}, flush=True)
for k in ('writer', 'script', 'work'):
    print(k, Counter(map(str, d[k])).most_common(30), flush=True)
by = defaultdict(set)
for w, s, work in zip(d['writer'], d['script'], d['work']):
    by[str(w)+'·'+str(s)].add(str(work))
print('WORKS_PER_SLOT', {k:len(v) for k,v in by.items()}, flush=True)
print('STD_RANGE', float(d['g_std'].min()), float(d['g_std'].max()),
      'STD_IDX_MIN', int(d['std_idx'].min()), flush=True)
rows = [r for r in csv.DictReader(open('exp-std/csv/train.csv', encoding='utf-8-sig'))
        if not (r.get('aug') or '').strip()]
meta = [(str(w),str(s),str(c)) for w,s,c in zip(d['writer'],d['script'],d['char'])]
print('CSV', len(rows), 'CACHE',len(meta),'ORDER_MATCH',len(rows)==len(meta) and
      all((r['calligrapher'].strip(),r['script'].strip(),r['character'].strip())==m
          for r,m in zip(rows,meta)), flush=True)
ck = torch.load('assets/writer_brush_critic.pt', map_location='cpu', weights_only=False)
print('CRITIC',list(ck), ck.get('results'), flush=True)

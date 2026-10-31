"""Validate the 21 shared cells against the panels we will actually read."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cells = json.loads((ROOT / 'tools/poster_20261101/cells.json').read_text(encoding='utf-8'))

LOCAL_200 = {
    'v80_200k': 'assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/'
                'eval_samples_ctrl/step0200000/eval200fix',
    'v85A_30k': 'assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/'
                'eval_samples_ctrl/step0030000/eval200fix',
    'moyi12ch_50k': 'docs/106moyi/moyi/results/moyi_top10_rf/eval_ours200fix',
}
LOCAL_100 = {}

print('cells:', cells['n'], ' idx200 range:', min(c['idx200'] for c in cells['cells']),
      '-', max(c['idx200'] for c in cells['cells']),
      ' idx100 range:', min(c['idx100'] for c in cells['cells']),
      '-', max(c['idx100'] for c in cells['cells']))
bad = [c for c in cells['cells'] if c['idx200'] >= 187 or c['idx100'] >= 100]
print('out-of-range cells:', bad if bad else 'none')
assert not bad, 'a cell index exceeds its panel length'

print('\nlocal 187-panel columns:')
for name, d in LOCAL_200.items():
    p = ROOT / d
    miss = [c['idx200'] for c in cells['cells'] if not (p / f'g{c["idx200"]}.png').exists()]
    print(f'  {name:<14} {"OK" if p.is_dir() else "NO DIR"}  missing {len(miss)}/21')
    if miss:
        print('    missing idx:', miss[:10])

print('\nGT source (moyi gt{i}.png):')
gp = ROOT / LOCAL_200['moyi12ch_50k']
miss = [c['idx200'] for c in cells['cells'] if not (gp / f'gt{c["idx200"]}.png').exists()]
print('  missing gt:', miss if miss else 'none')

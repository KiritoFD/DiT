import json
import glob
import os

runs = {}
for p in sorted(glob.glob('assets/t_eval/T*/gt_vs_generated.json')):
    tag = os.path.basename(os.path.dirname(p))
    runs[tag] = json.load(open(p, encoding='utf-8'))

hdr = ('run', 'kind', 'pairs', 'cos', 'pos', 'mag')
print('%-14s %-9s %6s %8s %7s %7s' % hdr)
for tag in sorted(runs):
    for kind in ('end', 'junction', 'turn'):
        v = runs[tag]['by_kind'][kind]
        print('%-14s %-9s %6d %8.4f %7.3f %7.3f' % (
            tag, kind, v['n_pairs'], v['mean_cosine'],
            v['frac_positive'], v['median_magnitude_ratio']))

print('\n=== per-feature direction agreement (cos) ===')
feats = ['n_branches', 'orient_peak', 'end_area', 'grad_mag', 'asym_ud', 'asym_lr',
         'turn_angle', 'end_elong', 'mass', 'ink_frac']
for kind in ('end', 'junction', 'turn'):
    print('\n-- %s --' % kind)
    print('%-13s' % 'feature', end='')
    for tag in sorted(runs):
        print('%12s' % tag, end='')
    print()
    for f in feats:
        vals = [runs[t]['per_feature'][kind].get(f) for t in sorted(runs)]
        if all(v is None for v in vals):
            continue
        print('%-13s' % f, end='')
        for v in vals:
            print('%12s' % ('%.4f' % v if v is not None else '-'), end='')
        print()

print('\n=== T1 - T0 deltas (5000 steps) ===')
for kind in ('end', 'junction', 'turn'):
    a = runs['T0_0005000']['by_kind'][kind]
    b = runs['T1_0005000']['by_kind'][kind]
    print('%-9s cos %+.4f (%+.1f%%)  pos %+.4f  mag %+.4f' % (
        kind, b['mean_cosine'] - a['mean_cosine'],
        100 * (b['mean_cosine'] - a['mean_cosine']) / abs(a['mean_cosine']),
        b['frac_positive'] - a['frac_positive'],
        b['median_magnitude_ratio'] - a['median_magnitude_ratio']))

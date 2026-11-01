"""Defect-feature diagnostics for every D arm plus GT, on the val100 panel.

These are morphology proxies from tools/p125/quality.py, not official metrics --
they exist to answer *which* of the three defect classes moved, since the official
panel numbers (SSIM/LPIPS/Ink-IoU) are all within noise of each other.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.p125.quality import defect_features, load_gray  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    'K_A': 'assets/f_20261101/eval/K_A',
    'D1_2500': 'assets/d_20261101/eval/D1_2500',
    'D2_3000': 'assets/d_20261101/eval/D2_3000',
    'D3_11000': 'assets/d_20261101/eval/D3_11000',
    'D3_11000_plus_D1dec': 'assets/d_20261101/eval/D3_11000_plus_D1dec',
}
N = 100
FEATS = ('ink_ratio', 'n_components', 'fragment_share', 'width_p50', 'width_p90',
         'width_jump', 'edge_roughness', 'endpoint_density')


def png_dir(cdir):
    base = ROOT / cdir
    best, best_key = None, None
    for cand in [base, *[p for p in base.rglob('*') if p.is_dir()]]:
        n_g = sum(1 for f in cand.glob('g[0-9]*.png') if f.stem[1:].isdigit())
        n_gt = sum(1 for f in cand.glob('gt*.png'))
        if n_g == 0:
            continue
        key = (1 if n_gt > 0 else 0, n_g)
        if best_key is None or key > best_key:
            best, best_key = cand, key
    return best


def aggregate(d, prefix):
    acc = {f: [] for f in FEATS}
    miss = 0
    for i in range(N):
        p = d / f'{prefix}{i}.png'
        if not p.exists():
            miss += 1
            continue
        g = load_gray(p)
        for k, v in defect_features(g).items():
            if k in acc:
                acc[k].append(v)
    return {f: float(np.mean(v)) if v else float('nan') for f, v in acc.items()}, miss


def main():
    rows = {}
    gt_d = png_dir(ARMS['K_A'])
    gt, gmiss = aggregate(gt_d, 'gt')
    rows['GT'] = gt
    print(f'GT from {gt_d}  (missing {gmiss})')
    for name, cdir in ARMS.items():
        d = png_dir(cdir)
        feat, miss = aggregate(d, 'g')
        rows[name] = feat
        print(f'{name:<24} from {d}  (missing {miss})')

    hdr = ('arm', 'ink%', 'blobs', 'frag%', 'w50', 'w50/GT', 'w90', 'wjump', 'rough', 'ends')
    print('\n' + ''.join(f'{h:>10}' for h in hdr))
    for name, f in rows.items():
        print(f'{name:<24}' + ''.join(f'{v:>10.4f}' for v in (
            f['ink_ratio'] * 100, f['n_components'], f['fragment_share'] * 100,
            f['width_p50'], f['width_p50'] / gt['width_p50'], f['width_p90'],
            f['width_jump'], f['edge_roughness'], f['endpoint_density'])))
    print('\nwidth_p50 / width_p90 in 256px pixels; blobs = mean connected components;'
          '\nfrag% = ink share of components < 0.5% of total; ends = endpoint density.')


if __name__ == '__main__':
    main()

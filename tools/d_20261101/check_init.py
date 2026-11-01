"""Verify that a D spec really starts from K, i.e. the style branch is inherited.

`construct()` only copies the backbone when style_matrix_repr >= 0; the style
branch is copied solely under `transfer_style`. Building the model with the flag
on and diffing against K's EMA is the cheapest proof the arm starts where it
claims to.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.style_matrix.runtime.common import construct  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / 'assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt'


def diff(model, ref, tag):
    st = {k.removeprefix('_orig_mod.'): v for k, v in model.state_dict().items()}
    keys = sorted(set(st) & set(ref))
    worst, name = 0.0, ''
    big = []
    for k in keys:
        m = float((st[k].float() - ref[k].float()).abs().max())
        if m > worst:
            worst, name = m, k
        if m > 1e-5:
            big.append((m, k))
    print(f'{tag}: compared {len(keys)} tensors, max|diff|={worst:.3e} at {name or "-"}, '
          f'offenders={len(big)}')
    for m, k in sorted(big, reverse=True)[:5]:
        print(f'    {m:.3e}  {k}')
    return worst


def main():
    spec_path = ROOT / 'assets/d_20261101/specs/D1.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    ref = torch.load(K, map_location='cpu', weights_only=False)['ema']
    ref = {k.removeprefix('_orig_mod.'): v for k, v in ref.items()}

    print(f'spec {spec_path.name}: transfer_style={spec.get("transfer_style")!r}')
    model, cfg, tr = construct(spec, device='cpu')
    diff(model, ref, 'as-built (current spec)')

    spec['transfer_style'] = True
    model2, cfg2, tr2 = construct(spec, device='cpu')
    print(f'  transfer report: { {k: v for k, v in tr2.items() if k != "diagnostics"} }')
    w = diff(model2, ref, 'with transfer_style=True')
    if w > 1e-5:
        raise SystemExit('style branch still does not match K')
    print('\nOK: with transfer_style=True the arm starts exactly at K')


if __name__ == '__main__':
    main()

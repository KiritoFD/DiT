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
        m = float((st[k].float().cpu() - ref[k].float().cpu()).abs().max())
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
    spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'assets/d_20261101/specs/D1.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    ref = torch.load(K, map_location='cpu', weights_only=False)['ema']
    ref = {k.removeprefix('_orig_mod.'): v for k, v in ref.items()}

    print(f'spec {spec_path.name}: transfer_style={spec.get("transfer_style")!r} '
          f'calibrate_gains={spec.get("calibrate_gains")!r}')
    model, cfg, tr = construct(spec, device='cpu')
    diff(model, ref, 'as-built (current spec)')

    spec['transfer_style'] = True
    model2, cfg2, tr2 = construct(spec, device='cpu')
    print(f'  transfer report: { {k: v for k, v in tr2.items() if k != "diagnostics"} }')
    w = diff(model2, ref, 'with transfer_style=True')
    if w > 1e-5:
        raise SystemExit('style branch still does not match K')

    # Pre-flight for the second trap: train.py runs condition_report(calibrate=True)
    # on every fresh start, which rescales the injection gains. Replay that exact
    # sequence so a bad spec is caught before spending GPU hours.
    from tools.style_matrix.runtime.common import condition_report
    spec_cal = dict(spec, calibrate_gains=bool(spec.get('calibrate_gains', True)))
    model3, cfg3, _ = construct(spec_cal, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        condition_report(model3, cfg3, calibrate=bool(spec_cal.get('calibrate_gains', True)))
    gains = [float(m3.gain.reshape(-1)[0]) for m3 in model3.style_injections]
    print(f'  gains after condition_report(calibrate='
          f'{spec_cal.get("calibrate_gains", True)}): '
          f'{"  ".join(f"{g:.4f}" for g in gains)}')
    w3 = diff(model3, ref, 'after the train.py init sequence')
    if w3 > 1e-5:
        raise SystemExit('FAIL: the run would not start from K (gains recalibrated?)')

    print('\nOK: the run starts exactly at K, gains included')


if __name__ == '__main__':
    main()

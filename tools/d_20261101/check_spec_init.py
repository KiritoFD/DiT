"""Construction-only pre-flight for the unfrozen arm (CPU, no GPU contention).

`check_init.py` proves the same thing but its second half calls
`condition_report(..., calibrate=True)` on CUDA, which needs ~20 GiB. GARL is holding
19.7 GiB of the 24.5 GiB card, so that half cannot run concurrently. This runs ONLY the
part that matters before launch -- does the spec actually start from K? -- on CPU.

The trap this catches is the one that invalidated D1/D2/D3: construct() copies the
backbone whenever style_matrix_repr >= 0, but copies the style branch ONLY under
transfer_style. Without it the arm starts from "K's backbone + random style branch".
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
    worst, name, big = 0.0, '', []
    for k in keys:
        m = float((st[k].float().cpu() - ref[k].float().cpu()).abs().max())
        if m > worst:
            worst, name = m, k
        if m > 1e-5:
            big.append((m, k))
    print(f'  {tag}: {len(keys)} tensors, max|diff|={worst:.3e} at {name or "-"}, '
          f'offenders={len(big)}')
    for m, k in sorted(big, reverse=True)[:5]:
        print(f'      {m:.3e}  {k}')
    return worst


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    ref = {k.removeprefix('_orig_mod.'): v
           for k, v in torch.load(K, map_location='cpu', weights_only=False)['ema'].items()}

    print(f'spec {spec_path.name}')
    print(f'  transfer_style={spec.get("transfer_style")!r} '
          f'calibrate_gains={spec.get("calibrate_gains")!r}')
    print(f'  freeze_backbone={spec.get("freeze_backbone")!r} skip_fm={spec.get("skip_fm")!r}')
    print(f'  joint.arm={spec.get("joint", {}).get("arm")!r}')

    model, cfg, tr = construct(spec, device='cpu')
    w = diff(model, ref, 'as-built from spec')

    ok = True
    if w > 1e-5:
        print('\nFAIL: the arm does NOT start from K.')
        print('      Most likely transfer_style is missing -- that is exactly the D1/D2/D3 bug.')
        ok = False

    # The unfrozen arm must actually have a trainable generator, otherwise it silently
    # degrades into the decoder-only GAR arm and the experiment is uninterpretable.
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    n_all = sum(1 for p in model.parameters())
    print(f'  trainable tensors: {n_train}/{n_all}')
    frozen = bool(spec.get('freeze_backbone', False))
    if frozen and n_train != 0:
        print('FAIL: freeze_backbone=True but the generator is still trainable'); ok = False
    if not frozen and n_train == 0:
        print('FAIL: unfrozen arm has no trainable generator'); ok = False

    # The generator graph is built only when skip_fm is False.
    if spec.get('joint', {}).get('arm') == 'GARU':
        if spec.get('skip_fm'):
            print('FAIL: GARU with skip_fm=True would never produce an FM gradient'); ok = False
        if spec.get('freeze_backbone'):
            print('FAIL: GARU with freeze_backbone=True is just GAR'); ok = False

    print('\nOK: the arm starts exactly at K and has the intended trainable surface'
          if ok else '\nPRE-FLIGHT FAILED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()

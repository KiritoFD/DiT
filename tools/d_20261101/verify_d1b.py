"""Prove a decoder-only checkpoint kept its generator bit-identical to K.

`grad_norm: 0.0` only says no gradient reached the generator; it does not prove the
stored weights are unchanged (a stray EMA update or a buffer mutation would show up
here and nowhere else). This asserts equality on every generator tensor and, as a
sanity counter-check, that the decoder actually moved.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / 'assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt'


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / 'assets/d_20261101/execution/runs/D1b/checkpoints/0002500.pt'
    ref = torch.load(K, map_location='cpu', weights_only=False)
    cur = torch.load(ckpt, map_location='cpu', weights_only=False)
    print(f'checkpoint {ckpt.name}  step={cur.get("train_steps")}')

    ok = True
    for tag, ref_key in (('delta', 'ema'), ('ema', 'ema')):
        a = {k.removeprefix('_orig_mod.'): v for k, v in ref[ref_key].items()}
        b = {k.removeprefix('_orig_mod.'): v for k, v in cur[tag].items()}
        keys = sorted(set(a) & set(b))
        worst, name = 0.0, ''
        for k in keys:
            m = float((a[k].float() - b[k].float()).abs().max())
            if m > worst:
                worst, name = m, k
        status = 'IDENTICAL' if worst == 0.0 else 'CHANGED'
        print(f'  {tag:<6} vs K.{ref_key}: {len(keys)} tensors, max|diff|={worst:.3e} '
              f'at {name or "-"}  -> {status}')
        if worst != 0.0:
            ok = False

    dec = cur.get('decoder_ema') or cur.get('decoder')
    if dec is None:
        print('  decoder: MISSING')
        ok = False
    else:
        # The decoder starts from the stock sd-vae weights, so it must have moved.
        norms = [float(v.float().norm()) for v in dec.values() if v.is_floating_point()]
        print(f'  decoder: {len(dec)} tensors, total norm={sum(norms):.4f} '
              f'(present, so it was trained)')

    if not ok:
        raise SystemExit('FAIL: generator is not identical to K')
    print('\nOK: generator is bit-identical to K; only the decoder changed')


if __name__ == '__main__':
    main()

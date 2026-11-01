"""Derive D1b from the already-tuned D1 spec, changing only what was wrong.

D1's spec on the training box carries the measured joint config (image_batch 64 /
image_micro 8), so it is the right template -- regenerating from build_specs.py
would reset those to the untuned defaults. The single semantic change is
`transfer_style: True`, which is what makes the arm actually start from K.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets/d_20261101/specs/D1.json'
DST = ROOT / 'assets/d_20261101/specs/D1b.json'


def main():
    s = json.loads(SRC.read_text(encoding='utf-8'))
    print(f'source spec: {SRC.name}  transfer_style={s.get("transfer_style")!r}')
    s['arm'] = 'D1b'
    s['transfer_style'] = True
    # `inspect(calibrate=True)` would rescale K's trained injection gains back to
    # delta_rms/base_rms == 0.1 (measured: 14.81/24.27/37.06 -> 6.01/19.87/22.52).
    # A decoder-only arm must not touch the generator, so keep the inherited gains.
    s['calibrate_gains'] = False
    ident = json.loads(json.dumps(s['identity']))
    ident['arm'] = 'D1b'
    ident.setdefault('contract', {})['correction'] = 'transfer_style_from_K+keep_gains'
    s['identity'] = ident
    # Keep the same budget as D1 so the two are directly comparable.
    DST.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'wrote {DST}')
    print(f'  transfer_style={s["transfer_style"]}  arm={s["arm"]}')
    print(f'  joint={s["joint"]["arm"]} image_batch={s["joint"]["image_batch"]} '
          f'image_micro={s["joint"]["image_micro"]}')
    c = s['config']
    print(f'  horizon={c["lr_decay_steps"]} ckpt_every={c["ckpt_every"]} '
          f'batch={c["batch"]} micro={c["microbatch"]} warmup={c["warmup"]}')
    print(f'  freeze_backbone={s.get("freeze_backbone")} skip_fm={s.get("skip_fm")} '
          f'decoder_lr={s.get("decoder_lr")}')


if __name__ == '__main__':
    main()

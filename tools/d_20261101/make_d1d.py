"""Derive D1d: same as D1c but with a usable decoder learning rate.

D1c ran at decoder_lr=5e-6 and moved the decoder by only ~1e-3 relative weight change
over 1000 steps -- float-level no-op, panel metrics bit-identical to K. A decoder
finetune needs a larger step: 5e-5 (10x) is the standard range and is what this arm tests.
Everything else (K start, frozen generator, 6h budget, save cadence) is unchanged.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets/d_20261101/specs/D1c.json'
DST = ROOT / 'assets/d_20261101/specs/D1d.json'

OLD_LR = 5e-06
NEW_LR = 5e-05


def main():
    s = json.loads(SRC.read_text(encoding='utf-8'))
    s['arm'] = 'D1d'
    ident = json.loads(json.dumps(s['identity']))
    ident['arm'] = 'D1d'
    ident.setdefault('contract', {})['correction'] = (
        'transfer_style_from_K+keep_gains; lr_horizon_extended_for_6h_budget; '
        'decoder_lr_5e-5_instead_of_5e-6')
    s['identity'] = ident
    s.pop('continues_identity', None)

    assert s['decoder_lr'] == OLD_LR, f'unexpected base lr {s["decoder_lr"]}'
    s['decoder_lr'] = NEW_LR

    DST.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'wrote {DST}')
    print(f'  decoder_lr {OLD_LR:g} -> {NEW_LR:g} ({NEW_LR / OLD_LR:.0f}x)')
    print(f'  budget={s["config"]["max_steps"]} steps, lr_decay={s["config"]["lr_decay_steps"]}, '
          f'ckpt/eval every {s["config"]["ckpt_every"]}')
    print(f'  source={s["source"]}')
    print(f'  freeze_backbone={s["freeze_backbone"]} skip_fm={s["skip_fm"]} '
          f'transfer_style={s["transfer_style"]}')


if __name__ == '__main__':
    main()

"""GAR arm: decoder-only post-training of K with the GAR texture objective.

Derived from D1b so every frozen contract (K generator, VAE latent coordinates, STD
bank, data CSV, remapped writer/script ids, identity conditions) is inherited exactly.
The only changes are:
  * joint arm  G2J4 -> GAR            (image objective: MSE + LPIPS/DINO + 10*Gram, no edge)
  * decoder_lr 5e-06 -> 5e-05         (D1c/D1b showed 5e-6 moves the decoder ~1e-3 in 1000
                                       steps, i.e. no measurable effect)
  * 4 h LR horizon and 250-step checkpoints, so a long run can be observed mid-flight
    without waiting for the endpoint.

`GAR` (short) and `GARL` (long) are the SAME experiment at two budgets: both start from
K, both freeze the generator, both use the GAR objective. They differ only in the LR
horizon and the checkpoint cadence, so a GARL checkpoint at equal step count is directly
comparable to a GAR one. GAR is NOT a prefix of GARL past step 672 -- the cosine schedule
has a different horizon, so the LR paths diverge. Never resume one from the other.

CFG is untouched: K is deployed at cfg=1.0 and the frozen generator is called at
cfg=1.0 exactly as the D arms did.
"""
import json
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets/d_20261101/specs/D1b.json'
DST = ROOT / 'assets/d_20261101/specs/GAR.json'
DST_LONG = ROOT / 'assets/d_20261101/specs/GARL.json'

HORIZON = 672          # LR decay horizon == 2 epochs over 21460 unique artworks
CKPT = 100             # the shared trainer requires eval_every == ckpt_every

# 4 h at the measured 0.287 step/s (3.48 s/step), rounded to a 250-step boundary.
LONG_HORIZON = 4000
LONG_CKPT = 250


def build(src, arm, horizon, ckpt, contract_extra):
    s = json.loads(src.read_text(encoding='utf-8'))
    s['arm'] = arm
    ident = json.loads(json.dumps(s['identity']))
    ident['arm'] = arm
    ident['contract'] = dict(ident.get('contract', {}))
    ident['contract']['objective'] = 'decoder_only_GAR'
    ident['contract']['gar'] = 'mse + lpips/DINO perceptual + 10*gram, no edge, cfg=1.0'
    ident['contract'].update(contract_extra)
    s['identity'] = ident
    s.pop('continues_identity', None)

    c = s['config']
    c['max_steps'] = horizon
    c['lr_decay_steps'] = horizon
    c['ckpt_every'] = ckpt
    c['eval_every'] = ckpt
    c['compile'] = False          # skip_fm => the frozen generator is never compiled
    s['milestones'] = list(range(ckpt, horizon + 1, ckpt))
    s['decoder_lr'] = 5e-5
    s['joint'] = dict(s['joint'], arm='GAR')
    return s


def main():
    for dst, arm, horizon, ckpt in ((DST, 'GAR', HORIZON, CKPT),
                                    (DST_LONG, 'GARL', LONG_HORIZON, LONG_CKPT)):
        s = build(SRC, arm, horizon, ckpt,
                  {'exposure_epochs': round(horizon / (21460 / 64), 2)})
        dst.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'wrote {dst}')
        print(f'  arm={arm}  joint.arm={s["joint"]["arm"]}  decoder_lr={s["decoder_lr"]}  '
              f'freeze_backbone={s["freeze_backbone"]}  skip_fm={s["skip_fm"]}')
        print(f'  horizon={horizon} ({horizon/(21460/64):.2f} epoch)  ckpt_every={ckpt}  '
              f'saves={horizon//ckpt}  image_batch={s["joint"]["image_batch"]}')
        print(f'  eta={horizon/0.287/3600:.2f} h at 0.287 step/s')


if __name__ == '__main__':
    main()


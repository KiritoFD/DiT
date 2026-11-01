"""GARU: unfreeze the generator AND the decoder, train both with the GAR objective.

Why this arm exists
-------------------
The D arms already ran joint training (D2 = J3, D3 = plain FM), and both were recorded as
hurting the writer. docs/d_results.md later retracted that: `build_specs.py` had stripped
`transfer_style`, so every D arm actually started from "K's backbone + a RANDOM style
branch" (33/368 tensors differed, max|diff| 36.06 at style_injections.2.gain). D2's
direction 0.4291 was never "joint training hurts the writer" -- it was 3000 FM steps
partially re-learning a branch that should have been inherited.

So the joint hypothesis is still unmeasured. GARU measures it on a clean basis:
  * starts from K with transfer_style=True (asserted by check_init.py before launch)
  * keeps the GAR image objective (no edge term, 10x Gram) and the decoder in the loop
  * adds a real gradient path into the generator (arm GARU => detach_image=False)

What changes vs GAR (the decoder-only arm)
------------------------------------------
  freeze_backbone  True  -> False    generator becomes trainable
  skip_fm          True  -> False    `common.step(fm=True)` builds the velocity-MSE graph
  joint.arm        GAR   -> GARU     keeps the generator graph instead of no_grad

LR policy: carry over D2's small-backbone LR (lr_backbone 5e-6) rather than inventing a
new one, so the result is comparable to the D2 it replaces. The style groups keep their
D2 values. `update_ema` runs because freeze_backbone is False, so the EMA is the
deployment-relevant weight.

Acceptance is by PAIRED comparison against K, never by "identical to K": the frozen-K
audit is meaningless for this arm by construction.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets/d_20261101/specs/D2.json'
DST = ROOT / 'assets/d_20261101/specs/GARU.json'

HORIZON = 2500          # matches D2's 3000-step budget within one ckpt interval, and
CKPT = 250              # 2500 steps ~= 2.4 h at the measured ~0.28 step/s


def main():
    s = json.loads(SRC.read_text(encoding='utf-8'))
    s['arm'] = 'GARU'

    # These three were absent from every D arm. Their absence is exactly the bug that
    # invalidated D1/D2/D3; re-adding them is the precondition for this arm to mean
    # anything at all.
    s['transfer_style'] = True
    s['calibrate_gains'] = False      # keep K's injection gains, do not recalibrate them
    s['freeze_backbone'] = False      # <-- the unfreeze
    s['skip_fm'] = False              # generator graph must exist
    s.pop('continues_identity', None)

    ident = json.loads(json.dumps(s['identity']))
    ident['arm'] = 'GARU'
    ident['contract'] = dict(ident.get('contract', {}))
    ident['contract']['objective'] = 'joint_GAR_unfrozen'
    ident['contract']['gar'] = 'mse + lpips/DINO perceptual + 10*gram, no edge, cfg=1.0'
    ident['contract']['correction'] = (
        'transfer_style=True (D1/D2/D3 lacked it); generator unfrozen; '
        'acceptance is paired-vs-K, NOT frozen-K identity')
    ident['contract']['replaces'] = 'D2 (invalid: random style branch, see docs/d_results.md)'
    s['identity'] = ident

    c = s['config']
    c['max_steps'] = HORIZON
    c['lr_decay_steps'] = HORIZON
    c['ckpt_every'] = CKPT
    c['eval_every'] = CKPT          # the shared trainer asserts these are equal
    c['compile'] = True             # the generator really runs now, so compilation pays
    # D2's LR policy, carried over unchanged for comparability.
    c['lr_backbone'] = 5e-6
    c['lr_style_repr'] = 5e-5
    c['lr_style_inj'] = 8e-5
    c['warmup'] = 200
    s['milestones'] = list(range(CKPT, HORIZON + 1, CKPT))
    s['decoder_lr'] = 5e-5

    # D2's image batch was 16 with micro 1. Keep it: a larger image batch on a trainable
    # generator changes the gradient balance, and comparability with D2 is the point.
    s['joint'] = dict(s['joint'], arm='GARU')

    DST.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'wrote {DST}')
    print(f'  joint.arm={s["joint"]["arm"]}  transfer_style={s["transfer_style"]}  '
          f'calibrate_gains={s["calibrate_gains"]}')
    print(f'  freeze_backbone={s["freeze_backbone"]}  skip_fm={s["skip_fm"]}  '
          f'decoder_lr={s["decoder_lr"]}')
    print(f'  lr_backbone={c["lr_backbone"]}  lr_style_repr={c["lr_style_repr"]}  '
          f'lr_style_inj={c["lr_style_inj"]}  warmup={c["warmup"]}')
    print(f'  horizon={horizon_line(c, HORIZON)}  ckpt_every={CKPT}  saves={HORIZON//CKPT}')
    print(f'  image_batch={s["joint"]["image_batch"]} image_micro={s["joint"]["image_micro"]}')
    print(f'  NOTE: acceptance is paired-vs-K. The frozen-K audit does NOT apply.')
    print(f'  PRE-FLIGHT: python tools/d_20261101/check_init.py {DST.relative_to(ROOT)}')


def horizon_line(c, h):
    return f'{h} steps (~{h/0.28/3600:.1f} h at 0.28 step/s, batch {c["batch"]})'


if __name__ == '__main__':
    main()

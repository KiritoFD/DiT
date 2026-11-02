"""P1: train the generator with FM + image-domain reconstruction, decoder frozen.

What this arm tests
-------------------
P0 established that the broken strokes are already in K's latent, and P0.5 established
(only) that plain reconstruction can move fragmentation a long way when the latent is
optimised directly. Neither shows the U-Net can LEARN that. This arm is the minimum
experiment that does:

    L = L_FM + lambda * L_image(D(z_hat), GT)

with the VAE frozen, the U-Net unfrozen, and the image loss back-propagating THROUGH the
frozen decoder. No DINO, no Gram, no clDice -- deliberately. If a plain image L1 cannot
help, adding perceptual and topology terms is premature, and the earlier GAR/GARU arms
already showed that the complicated objective does not fix strokes.

Relation to the other arms
--------------------------
  GAR   : decoder-only, generator bit-identical (368 tensors unchanged), complex GAR loss
  GARU  : generator unfrozen, complex GAR loss, decoder also trainable
  P1    : generator unfrozen, SIMPLE image loss, decoder frozen

So P1 is the controlled counterpart to GARU: same unfreezing, much simpler objective, and
the decoder held still so the only thing that can move is the generator.

Weight selection is NOT guessed. `--probe` runs a gradient audit that reports, per t, the
ratio |g_image| / |g_FM| and the cosine between them. The review's starting scan range is
5-20%; the cosine matters because a strongly opposed direction means the image term is
fighting flow matching, which is how stroke sharpness and sample diversity degrade.

Acceptance is a PAIRED comparison against K (this arm trains the generator, so the
frozen-K audit is meaningless by construction). Readouts must include frag AND ink
retention AND OCR AND E-G -- frag alone cannot distinguish a repair from an erasure.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets/d_20261101/specs/D2.json'
DST = ROOT / 'assets/d_20261101/specs/P1.json'

HORIZON = 2000          # short and paired; the point is the A/B, not a long run
CKPT = 200


def build(lam, horizon=HORIZON, dst=DST):
    s = json.loads(SRC.read_text(encoding='utf-8'))
    s['arm'] = 'P1'

    # Carry over the three fields whose absence invalidated D1/D2/D3. Without
    # transfer_style the arm silently starts from "K backbone + random style branch",
    # which is the documented D-arm bug.
    s['transfer_style'] = True
    s['calibrate_gains'] = False
    s['freeze_backbone'] = False     # the generator is what we train
    s['skip_fm'] = False             # FM must stay: it is the main objective
    s.pop('continues_identity', None)

    ident = json.loads(json.dumps(s['identity']))
    ident['arm'] = 'P1'
    ident['contract'] = dict(ident.get('contract', {}))
    ident['contract']['objective'] = 'FM + image L1 through frozen decoder'
    ident['contract']['image_loss'] = f'l1 (lambda={lam})'
    ident['contract']['decoder'] = 'FROZEN (this is what separates P1 from GARU)'
    ident['contract']['correction'] = (
        'transfer_style=True (D1/D2/D3 lacked it); generator unfrozen; decoder frozen; '
        'acceptance is paired-vs-K, NOT frozen-K identity')
    ident['contract']['excluded_terms'] = 'no DINO, no Gram, no clDice, no edge term'
    s['identity'] = ident

    c = s['config']
    c['max_steps'] = horizon
    c['lr_decay_steps'] = horizon
    c['ckpt_every'] = CKPT
    c['eval_every'] = CKPT           # the shared trainer asserts these are equal
    c['compile'] = True
    c['lr_backbone'] = 5e-6          # D2's value, for comparability
    c['lr_style_repr'] = 5e-5
    c['lr_style_inj'] = 8e-5
    c['warmup'] = 200
    s['milestones'] = list(range(CKPT, horizon + 1, CKPT))

    # The decoder is frozen: keep it out of the optimizer entirely, and set the image
    # weight to the value the gradient audit justifies.
    j = dict(s['joint'], arm='P1')
    j['image_weight'] = lam
    j['rec_weight'] = 0.0            # pure FM + image; no separate latent reconstruction
    s['joint'] = j
    s['train_decoder'] = False
    s['decoder_lr'] = 0.0

    dst.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambda-image', type=float, default=0.05,
                    help='image loss weight; chosen from the --probe gradient audit')
    ap.add_argument('--horizon', type=int, default=HORIZON)
    ap.add_argument('--out', default=str(DST))
    a = ap.parse_args()

    dst = Path(a.out)
    s = build(a.lambda_image, a.horizon, dst)
    print(f'wrote {dst}')
    print(f'  arm={s["arm"]}  transfer_style={s["transfer_style"]}  '
          f'freeze_backbone={s["freeze_backbone"]}  skip_fm={s["skip_fm"]}')
    print(f'  train_decoder={s["train_decoder"]}  image_weight={s["joint"]["image_weight"]}')
    print(f'  lr_backbone={s["config"]["lr_backbone"]}  horizon={a.horizon}  ckpt={CKPT}')
    print(f'  NOTE: acceptance is paired-vs-K; report frag AND ink_retention AND OCR AND E-G.')
    print(f'  PRE-FLIGHT: python tools/d_20261101/check_spec_init.py {dst}')


if __name__ == '__main__':
    main()

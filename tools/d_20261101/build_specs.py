"""Build the D1/D2/D3 post-training specs from the K checkpoint.

All three start from K's EMA, keep the full Writer injection stack, and freeze the
VAE encoder (JointDecoderObjective enforces that with a hash). They differ only in
what carries the gradient:

  D1  decoder only, image loss on generated latents (U-Net frozen, FM skipped)
  D2  U-Net + decoder, FM + weak image loss + GT-reconstruction anchor
  D3  U-Net only, plain FM on a hard-character-oversampled training set
"""
import argparse
import json
from pathlib import Path

K = 'assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt'
TEMPLATE = 'assets/f_20261101/specs/F0.json'


def load_template(path=TEMPLATE):
    s = json.loads(Path(path).read_text(encoding='utf-8'))
    # K already carries the right config; drop fields that only make sense for
    # a resumed run or for the F arms' site ablation.
    for key in ('schedule_restart', 'continues_identity', 'disable_injection_sites',
                'transfer_style', 'joint', 'velocity_pair_weight',
                'velocity_pair_gradient_ratio'):
        s.pop(key, None)
    s['source'] = K
    s['initialization'] = 'stage_transfer'
    return s


def spec(arm, template, *, horizon, ckpt_every, identity_extra, **over):
    s = json.loads(json.dumps(template))
    s['arm'] = arm
    s['milestones'] = list(range(ckpt_every, horizon + 1, ckpt_every))
    s['config']['lr_decay_steps'] = horizon
    s['config']['ckpt_every'] = ckpt_every
    s['config']['eval_every'] = ckpt_every   # the trainer requires these to match
    ident = json.loads(json.dumps(s['identity']))
    ident['arm'] = arm
    ident.setdefault('contract', {}).update(identity_extra)
    s['identity'] = ident
    for k, v in over.items():
        if k in ('config', 'joint'):
            s[k] = dict(s.get(k) or {}, **v)
        else:
            s[k] = v
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='assets/d_20261101/specs')
    ap.add_argument('--calib-dir', default='assets/d_20261101')
    ap.add_argument('--template', default=TEMPLATE)
    a = ap.parse_args()
    t = load_template(a.template)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- D1: decoder-only -------------------------------------------------
    d1 = spec('D1', t, horizon=3000, ckpt_every=500,
              identity_extra=dict(objective='decoder_only_image',detach_generator=True),
              kind='joint', freeze_backbone=True, skip_fm=True, decoder_lr=5e-6,
              joint=dict(arm='G2J4',
                         calibration=f'{a.calib_dir}/calibration_D1.json',
                         image_batch=16, image_micro=2))
    # Only the decoder moves, so the LR schedule applies to a single group.
    d1['config'].update(lr_backbone=0.0, lr_style_repr=0.0, lr_style_inj=0.0,
                        lr_writer_table=0.0, warmup=50)

    # ---- D2: U-Net + decoder ---------------------------------------------
    d2 = spec('D2', t, horizon=3000, ckpt_every=500,
              identity_extra=dict(objective='fm_plus_weak_image'),
              kind='joint', decoder_lr=5e-6,
              joint=dict(arm='J3',
                         calibration=f'{a.calib_dir}/calibration_D2.json',
                         image_batch=16, image_micro=1))
    d2['config'].update(batch=512, microbatch=128, warmup=200)

    # ---- D3: hard-character FM -------------------------------------------
    d3 = spec('D3', t, horizon=3000, ckpt_every=500,
              identity_extra=dict(objective='fm_hard_chars', sampling='hard_chars'),
              kind='plain')
    d3['config'].update(data_csv=f'{a.calib_dir}/stage_p2_hard.csv',
                        batch=512, microbatch=512, warmup=200)

    for name, s in (('D1', d1), ('D2', d2), ('D3', d3)):
        p = out / f'{name}.json'
        p.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
        j = s.get('joint')
        print(f'{name}: kind={s["kind"]} horizon={s["config"]["lr_decay_steps"]} '
              f'batch={s["config"]["batch"]} micro={s["config"]["microbatch"]} '
              f'joint={None if not j else (j["arm"], j["image_batch"], j["image_micro"])} '
              f'freeze_backbone={s.get("freeze_backbone", False)} '
              f'skip_fm={s.get("skip_fm", False)} csv={s["config"]["data_csv"].split("/")[-1]}')
    print('wrote', out)


if __name__ == '__main__':
    main()

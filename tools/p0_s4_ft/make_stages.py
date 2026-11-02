"""Build the three serial P0 -> S4 fine-tuning stages.

The plan's table, made literal:

  stage   steps   scope                          lr (backbone / R2+I1 / writer)
  S4-W      3k    freeze backbone, train R2+I1    -        5e-5 / 1e-4   1e-4
  S4-FM    35k    full model, VAE frozen          1e-5     5e-5          2.5e-5
  S4-Tail  <=10k  full model, low LR              3e-6     1e-5          -

Data is Top10 for all stages: the point of S4-W being Top10-only is to teach the new I1
the target writer condition without repeating K's 70% generic mix. The mixed-ratio idea
stays an ablation, not a default -- its earlier positive result came from a different
pretrain chain.

Chaining: each stage resumes the previous one's checkpoint. Stage 1 is the only stage that
runs the P0 -> S4 partial transfer; stages 2 and 3 inherit by resume, which keeps R2/I1
weights across the boundary (check #4 in the plan).

Stage 1 cannot use `--resume` (there is no prior checkpoint), so it starts from the
`partial` initialization. `fix_note`: S4-W freezes the backbone and therefore must not run
the FM loss -- `train.py` asserts exactly that.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_SPEC = ROOT / 'assets/style_matrix_20261027/matrix64/4090/specs/S4.json'
P0_CKPT = ROOT / 'assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt'
OUT_DIR = ROOT / 'assets/p0_s4_ft/specs'

# The plan wants Top10 at 50% frequency plus 50% writer-balanced. `data_csv` is the
# Top10 train set already used by S4; the sampling fractions live in the config.
DATA_CSV = 'assets/u100_p1p2_20261025/p2_train.csv'

STAGES = [
    dict(name='S4-W', steps=3000, freeze_backbone=True, backbone_lr=None,
         repr_lr=5e-5, inj_lr=1e-4, writer_lr=1e-4, ckpt=500, warmup=200),
    dict(name='S4-FM', steps=35000, freeze_backbone=False, backbone_lr=1e-5,
         repr_lr=5e-5, inj_lr=5e-5, writer_lr=2.5e-5, ckpt=1000, warmup=500),
    dict(name='S4-Tail', steps=10000, freeze_backbone=False, backbone_lr=3e-6,
         repr_lr=1e-5, inj_lr=1e-5, writer_lr=None, ckpt=1000, warmup=200),
]


def build(stage, base, prev_ckpt=None):
    spec = json.loads(json.dumps(base))
    s = dict(stage)
    spec['arm'] = s['name']
    spec['machine'] = '4090'
    spec['kind'] = 'style'

    c = spec['config']
    c['max_steps'] = s['steps']
    c['lr_decay_steps'] = s['steps']
    c['ckpt_every'] = s['ckpt']
    c['eval_every'] = s['ckpt']        # trainer asserts equality
    c['warmup'] = s['warmup']
    c['lr_style_repr'] = s['repr_lr']
    c['lr_style_inj'] = s['inj_lr']
    if s['backbone_lr'] is not None:
        c['lr'] = s['backbone_lr']
        c['lr_backbone'] = s['backbone_lr']
    if s['writer_lr'] is not None:
        c['writer_cond_lr'] = s['writer_lr']
    c['data_csv'] = DATA_CSV
    c['compile'] = True

    spec['freeze_backbone'] = s['freeze_backbone']
    # `skip_fm` drops the FORWARD entirely (`common.step` returns before calling the
    # model), which is only correct when nothing in the generator trains. S4-W DOES train
    # the style branch, so it must keep FM -- it just routes the gradient only into
    # R2/I1/writer via `train_scope`.
    spec['skip_fm'] = False
    # Name the prefixes that stay trainable while the backbone is frozen. The trainer
    # matches these against real parameter names and errors if nothing matches.
    spec['train_scope'] = (['style_representation.', 'style_injections.', 'backbone.writer.']
                           if s['freeze_backbone'] else None)
    spec['transfer_style'] = False     # rejected for this migration; see partial branch
    spec['calibrate_gains'] = False    # the plan: no auto gain calibration on a fresh I1
    spec['milestones'] = list(range(s['ckpt'], s['steps'] + 1, s['ckpt']))

    ident = json.loads(json.dumps(spec.get('identity', {})))
    ident['arm'] = s['name']
    ident['contract'] = dict(ident.get('contract', {}))
    ident['contract']['objective'] = 'FM only (no structure loss, no extra style supervision)'
    ident['contract']['scope'] = ('backbone frozen; R2+I1+writer trained'
                                  if s['freeze_backbone'] else 'all generator trainable; VAE frozen')
    ident['contract']['data'] = 'Top10'
    ident['contract']['excluded'] = 'no clDice / clDice / GAR / DINO / Gram / image-domain loss'
    spec['identity'] = ident

    if prev_ckpt is None:
        spec['initialization'] = 'partial'
        spec['source'] = str(P0_CKPT)
        spec['partial'] = dict(writer_jitter=0.01, writer_seed=int(c.get('seed', 0)))
    else:
        # Stages 2/3 continue by resume; `construct` must still be told what to build,
        # and `transfer_style` must stay off (there is no style copy to do).
        spec['initialization'] = 'partial'
        spec['source'] = str(P0_CKPT)
        spec['partial'] = dict(writer_jitter=0.01, writer_seed=int(c.get('seed', 0)))
        spec['continues_identity'] = ident
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT_DIR))
    a = ap.parse_args()
    base = json.loads(SRC_SPEC.read_text(encoding='utf-8'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    prev = None
    plans = []
    for st in STAGES:
        spec = build(st, base, prev)
        p = out / f'{st["name"]}.json'
        p.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
        plans.append(p)
        print(f'{p.name}: steps={st["steps"]} freeze_backbone={st["freeze_backbone"]} '
              f'skip_fm={spec["skip_fm"]} repr_lr={st["repr_lr"]} inj_lr={st["inj_lr"]} '
              f'backbone_lr={st["backbone_lr"]} writer_lr={st["writer_lr"]}')
        prev = str(ROOT / f'assets/p0_s4_ft/runs/{st["name"]}/checkpoints/{st["steps"]:07d}.pt')
    print(f'\n{len(plans)} stage specs in {out}')
    print('stage 1 does the partial transfer; stages 2-3 chain by --resume')


if __name__ == '__main__':
    main()

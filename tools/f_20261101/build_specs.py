"""Build the F0/F1 post-training specs from a chosen K checkpoint.

Everything except the injection-site policy and the LR table is inherited from
the audited A recipe (`assets/abc_20261031/specs/A.json`), so the only
scientific difference between the two arms is the weak E-G writer objective.

Design decisions (evidence in docs/1031/p125_phase2_diag.md):
  * sites 0 (32x32) and 1 (16x16) are disabled AND frozen: T2 showed removing
    them lowers FM loss by 0.0154 / 0.0085, T3 showed they carry no writer
    identity (direction 0.3595 -> 0.3506 / 0.3209 when zeroed).
  * site 2 (8x8) keeps training with the highest LR: zeroing it halves the
    writer-swap direction (0.3595 -> 0.1659), so it is the only carrier.
  * backbone runs at a lower LR than A (5e-6 vs 1e-5) because this is a short
    adaptation, not a fresh segment.
"""
import argparse
import json
from pathlib import Path

A_SPEC = 'assets/abc_20261031/specs/A.json'
DISABLE_SITES = [0, 1]
EG_BLOCK = dict(arm='EG',
                refs_npz='assets/p125_20261030/eg_refs.npz',
                every=4, image_micro=4, target_ratio=0.02)


def build_arm(base, arm, k_source, stop, horizon, out_root):
    """Return a spec dict for one arm, cloned from the A recipe."""
    spec = json.loads(json.dumps(base))          # deep copy
    cfg = spec['config']

    # A carried a controlled LR restart because it continued an existing run.
    # The F arms start a fresh 12k cosine from K, so the restart block must go
    # (train.py rejects schedule_restart without --resume).
    spec.pop('schedule_restart', None)
    spec.pop('continues_identity', None)

    spec['arm'] = arm
    spec['kind'] = 'joint' if arm == 'F1' else 'style'
    spec['initialization'] = 'stage_transfer'
    spec['source'] = k_source
    spec['transfer_style'] = True
    spec['disable_injection_sites'] = list(DISABLE_SITES)
    spec['milestones'] = [stop]

    cfg.update(
        lr_backbone=5e-6,          # A used 1e-5; this is a short adaptation
        lr_style_repr=5e-5,
        lr_style_inj=8e-5,         # only site 2 survives freezing
        lr_writer_table=5e-5,      # A used 2.5e-5; writer table is the point here
        lr_decay_steps=horizon,
        max_steps=horizon,
        warmup=200,
        ckpt_every=5000,
        eval_every=5000,
        output=str(Path(out_root) / 'runs'),
        data_csv='assets/p125_20261030/stage_p2.csv',
    )

    if arm == 'F1':
        spec['joint'] = dict(EG_BLOCK)

    # identity: identical contract, arm-specific label so runs never mix
    ident = json.loads(json.dumps(base['identity']))
    ident['arm'] = arm
    ident['f_plan'] = dict(disable_injection_sites=list(DISABLE_SITES),
                           horizon=horizon,
                           lrs={k: cfg[k] for k in
                                ('lr_backbone', 'lr_style_repr', 'lr_style_inj',
                                 'lr_writer_table')},
                           aux='eg' if arm == 'F1' else 'none',
                           eg_target_ratio=EG_BLOCK['target_ratio'] if arm == 'F1' else 0.0,
                           source=k_source)
    spec['identity'] = ident
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', required=True, help='K checkpoint (stage_transfer source)')
    ap.add_argument('--out', default='assets/f_20261101/specs')
    ap.add_argument('--stop', type=int, default=5000, help='first-round stop')
    ap.add_argument('--horizon', type=int, default=12000, help='cosine horizon')
    a = ap.parse_args()

    base = json.loads(Path(A_SPEC).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in ('F0', 'F1'):
        spec = build_arm(base, arm, a.k, a.stop, a.horizon, 'assets/f_20261101')
        path = out / f'{arm}.json'
        path.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
        print(f'[spec] {path} arm={arm} kind={spec["kind"]} '
              f'disable={spec["disable_injection_sites"]} '
              f'lr_inj={spec["config"]["lr_style_inj"]:.1e} '
              f'horizon={a.horizon} stop={a.stop}')
    print('[spec] done — same file is reused for the +7k extension '
          '(resume with --stop horizon)')


if __name__ == '__main__':
    main()

"""Derive a 6-hour decoder-only continuation arm from D1b.

Only the LR horizon, the step budget and the checkpoint cadence change; the frozen
backbone, the starting point (K) and the decoder LR stay identical, so results remain
comparable with D1b.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets/d_20261101/specs/D1b.json'
DST = ROOT / 'assets/d_20261101/specs/D1c.json'

STEP_S = 0.3507          # measured on D1b (steps per second)
HOURS = 6.0
BUDGET = int(HOURS * 3600 * STEP_S)          # ~7575 steps
ROUND = 500
TOTAL = (BUDGET // ROUND) * ROUND            # 7500


def main():
    s = json.loads(SRC.read_text(encoding='utf-8'))
    s['arm'] = 'D1c'
    ident = json.loads(json.dumps(s['identity']))
    ident['arm'] = 'D1c'
    ident.setdefault('contract', {})['correction'] = (
        'transfer_style_from_K+keep_gains; lr_horizon_extended_for_6h_budget')
    s['identity'] = ident
    # A continuation of D1b must be allowed to resume from its checkpoint.
    s['continues_identity'] = json.loads(SRC.read_text(encoding='utf-8'))['identity']

    c = s['config']
    c['max_steps'] = TOTAL
    c['lr_decay_steps'] = TOTAL        # train.py asserts stop <= lr_decay_steps
    # The shared trainer requires one interval for checkpoints AND eval.
    c['ckpt_every'] = 1000
    c['eval_every'] = 1000
    s['milestones'] = [m for m in s.get('milestones', []) if m <= TOTAL]

    DST.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'wrote {DST}')
    print(f'  budget={TOTAL} steps  (~{TOTAL / STEP_S / 3600:.2f} h at {STEP_S} step/s)')
    print(f'  lr_decay_steps={c["lr_decay_steps"]}  ckpt_every={c["ckpt_every"]}')
    print(f'  decoder_lr={s["decoder_lr"]}  freeze_backbone={s["freeze_backbone"]}  '
          f'skip_fm={s["skip_fm"]}  transfer_style={s["transfer_style"]}')
    print(f'  saves={(TOTAL // c["ckpt_every"])} x 1.71GB = '
          f'{TOTAL // c["ckpt_every"] * 1.71:.0f} GB')


if __name__ == '__main__':
    main()

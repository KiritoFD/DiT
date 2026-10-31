"""Patch a D-arm spec's runtime knobs without hand-editing JSON.

Used both to probe headroom (raise image_batch until nvidia-smi shows the real
ceiling) and to freeze the final measured configuration.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', default='assets/d_20261101/specs')
    ap.add_argument('--arm', required=True)
    ap.add_argument('--image-batch', type=int)
    ap.add_argument('--image-micro', type=int)
    ap.add_argument('--batch', type=int)
    ap.add_argument('--micro', type=int)
    ap.add_argument('--horizon', type=int, help='lr_decay_steps and milestone ceiling')
    ap.add_argument('--ckpt-every', type=int)
    a = ap.parse_args()

    p = Path(a.specs) / f'{a.arm}.json'
    s = json.loads(p.read_text(encoding='utf-8'))
    j = s.get('joint') or {}
    if a.image_batch is not None:
        j['image_batch'] = a.image_batch
    if a.image_micro is not None:
        j['image_micro'] = a.image_micro
    if j:
        s['joint'] = j
    if a.batch is not None:
        s['config']['batch'] = a.batch
    if a.micro is not None:
        s['config']['microbatch'] = a.micro
    if a.ckpt_every is not None:
        s['config']['ckpt_every'] = a.ckpt_every
        s['config']['eval_every'] = a.ckpt_every
    if a.horizon is not None:
        s['config']['lr_decay_steps'] = a.horizon
        every = s['config']['ckpt_every']
        s['milestones'] = list(range(every, a.horizon + 1, every))
    p.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{p}: batch={s["config"]["batch"]} micro={s["config"]["microbatch"]} '
          f'joint={s.get("joint")} horizon={s["config"]["lr_decay_steps"]} '
          f'milestones={s.get("milestones")}')


if __name__ == '__main__':
    main()

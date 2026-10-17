"""Compute fixed real-artwork targets, never fit them on generated/eval images."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.loss.critic_loss import CriticLoss
from src.train.strong_style import real_mask, writer_margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='scratch/disc_lat_full.npz')
    ap.add_argument('--critic', default='assets/writer_brush_critic.pt')
    ap.add_argument('--out', default='assets/strong_style_real_calibration.json')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--min-per-slot', type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(4)
    d = np.load(args.data, allow_pickle=True)
    critic = CriticLoss(args.critic, args.device)
    mask = real_mask(critic.slots, args.device)
    pools = {s: {'writer': [], 'brush': []} for s in critic.slots if not s.startswith('std·')}
    # This bank is the critic-development real bank, not an independent final test.
    for start in range(0, len(d['z']), 64):
        rows = list(range(start, min(start + 64, len(d['z']))))
        rows = [i for i in rows if str(d['writer'][i]) + '·' + str(d['script'][i]) in pools]
        if not rows:
            continue
        slots = [str(d['writer'][i]) + '·' + str(d['script'][i]) for i in rows]
        ys = torch.tensor([critic.slots.index(s) for s in slots], device=args.device)
        if any(int(mask[y].sum()) < 2 for y in ys):
            raise ValueError('a script has no real writer rival')
        si = torch.tensor([critic.s2i[str(d['script'][i])] for i in rows], device=args.device)
        g = torch.tensor(d['g_std'][d['std_idx'][rows].astype(np.int64)], device=args.device).float()[:, None] / 255.
        z = torch.tensor(d['z'][rows], device=args.device).float()
        with torch.no_grad():
            out = critic.model(z, g, si)
            margins = writer_margin(out['slot_logits'], ys, mask).cpu().tolist()
            brushes = out['brush_logit'].cpu().tolist()
        for slot, margin, brush in zip(slots, margins, brushes):
            pools[slot]['writer'].append(margin)
            pools[slot]['brush'].append(brush)
    if any(len(v['writer']) < args.min_per_slot for v in pools.values()):
        raise ValueError('insufficient real samples in one or more slots')
    payload = {'critic_sha256': hashlib.sha256(Path(args.critic).read_bytes()).hexdigest(),
               'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
               'slots': critic.slots, 'source': args.data, 'split': 'critic-development bank; not independent test',
               'counts': {k: len(v['writer']) for k, v in pools.items()}}
    for kind in ('writer', 'brush'):
        payload[kind + '_targets'] = {str(q): {slot: float(np.quantile(v[kind], q))
            for slot, v in pools.items()} for q in (.25, .5)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.out, payload['counts'])


if __name__ == '__main__':
    main()

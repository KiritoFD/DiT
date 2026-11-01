"""R3: latent inversion -- can the FROZEN Decoder express the right topology?

    z* = argmin_z  L1(D(z), x_GT) + lam * ||z - E(x_GT)||^2

The decoder is frozen throughout. If the ordinary E(x_GT) gives a bad
reconstruction but an optimised z recovers it, the decoder HAS the capacity and
the fault lies in the encoder/bottleneck. If optimising z cannot recover it,
the decoder or the 4x32x32 latent capacity is the limit.

Deliberately NOT optimising any topology statistic: n_branches/end_area were
shown unstable (R2, C2), so using them in the objective would build the answer
into the loss. The objective is L1 + a mild edge term, both computed in raw
image space; topology is only used for evaluation afterwards.

A convergence guard is recorded, because "optimisation did not help" is only
meaningful if the optimisation actually converged.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stroke_interaction.r2_blur import robust_topology, THRS  # noqa: E402


def to_ink(raw):
    return 1.0 - raw if raw.mean() > 0.5 else raw


def grey(x):
    """Collapse to one channel; decoder output is 3ch, GT is 1ch."""
    return x.mean(1, keepdim=True) if x.shape[1] != 1 else x


def edge_map(x):
    x = grey(x)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                     device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='exp-std/csv/train_top10_aug_sym.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--chars', type=int, default=20)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.08)
    ap.add_argument('--lam', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    import csv
    from PIL import Image
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(ROOT / 'pretrained_models/sd-vae-ft-ema',
                                        torch_dtype=torch.float32,
                                        local_files_only=True).cuda().eval()
    for p_ in vae.parameters():
        p_.requires_grad_(False)

    rows = [r for r in csv.DictReader(open(ROOT / a.csv, encoding='utf-8'))
            if (ROOT / r['image_path']).exists()]
    rng = np.random.default_rng(a.seed)
    by = defaultdict(list)
    for r in rows:
        by[(r['script'], r['character'])].append(r)
    cand = [k for k, v in by.items() if len({r['calligrapher'] for r in v}) >= 3]
    rng.shuffle(cand)
    cand = cand[:a.chars]
    print(f'[R3] {len(cand)} characters, {a.steps} steps, lr={a.lr}', flush=True)

    rep = {'n_chars': len(cand), 'steps': a.steps, 'lr': a.lr, 'lam': a.lam,
           'per_char': {}}
    agg = defaultdict(list)
    for ci, (sc, ch) in enumerate(cand):
        r = by[(sc, ch)][0]
        raw = np.array(Image.open(ROOT / r['image_path']).convert('L'),
                       np.float32) / 255.0
        x_gt = torch.from_numpy(raw)[None, None].cuda() * 2 - 1
        x3 = x_gt.repeat(1, 3, 1, 1).float()
        with torch.no_grad():
            z_enc = vae.encode(x3).latent_dist.mode()
            rec0 = vae.decode(z_enc).sample.float()
        # optimise in the pre-scaling latent space, as the sampler does
        z0 = z_enc.detach().clone().float()
        z = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=a.lr)
        hist = []
        for it in range(a.steps):
            dec = vae.decode(z).sample.float()
            l1 = (grey(dec) - grey(x_gt)).abs().mean()
            edge = (edge_map(dec) - edge_map(x_gt)).abs().mean()
            reg = (z - z0).pow(2).mean()
            loss = l1 + 0.5 * edge + a.lam * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if it % 50 == 0 or it == a.steps - 1:
                hist.append([it, round(float(loss), 5), round(float(l1), 5)])
        with torch.no_grad():
            rec1 = vae.decode(z).sample.float()

        def metrics(rec):
            raw_ = ((rec[0].clamp(-1, 1) + 1) / 2).mean(0).cpu().numpy()
            ink = to_ink(raw_)
            jc = [robust_topology(ink, t) for t in THRS]
            jc = [t_['junction'] for t_ in jc if t_]
            return {'l1_vs_gt': float(np.abs(ink - to_ink(raw)).mean()),
                    'grad': float(np.hypot(*np.gradient(ink)).mean()),
                    'junc': float(np.mean(jc)) if jc else None}

        m0, m1 = metrics(rec0), metrics(rec1)
        image_dir = ROOT / a.out / 'R3_images'
        image_dir.mkdir(parents=True, exist_ok=True)
        panel = Image.new('RGB', (768,256), 'white')
        for j, tensor in enumerate((x3,rec0,rec1)):
            arr=((tensor.detach().float().clamp(-1,1)+1)/2)[0].permute(1,2,0).cpu().numpy()
            panel.paste(Image.fromarray((arr*255).round().astype(np.uint8)),(j*256,0))
        panel.save(image_dir/f'{ci:03d}.png')
        entry = {'z_enc': m0, 'z_opt': m1,
                 'l1_improvement': round(m0['l1_vs_gt'] - m1['l1_vs_gt'], 5),
                 'junc_change': (round(m1['junc'] - m0['junc'], 3)
                                 if m0['junc'] is not None and m1['junc'] is not None
                                 else None),
                 'converged': bool(abs(hist[-1][2] - hist[-2][2]) < 0.002),
                 'loss_hist': hist}
        rep['per_char'][f'{sc}|{ch}'] = entry
        for k in ('l1_vs_gt', 'grad', 'junc'):
            if m0[k] is not None:
                agg[f'z_enc_{k}'].append(m0[k])
            if m1[k] is not None:
                agg[f'z_opt_{k}'].append(m1[k])
        if entry['l1_improvement'] is not None:
            agg['l1_improvement'].append(entry['l1_improvement'])
        print(f"[R3] {sc}|{ch}: l1 {m0['l1_vs_gt']:.4f}->{m1['l1_vs_gt']:.4f} "
              f"junc {m0['junc']}->{m1['junc']} conv={entry['converged']}", flush=True)

    summary = {k: {'mean': round(float(np.mean(v)), 4), 'n': len(v)}
               for k, v in agg.items()}
    rep['summary'] = summary
    imp = summary.get('l1_improvement', {}).get('mean')
    conv = sum(1 for e in rep['per_char'].values() if e['converged'])
    rep['verdict'] = {
        'mean_l1_improvement': imp,
        'n_converged': conv, 'n_total': len(cand),
        'decoder_has_capacity': bool(imp is not None and imp > 0.001 and
                                     conv >= 0.6 * len(cand)),
    }
    print(f'\n[R3] summary: {summary}', flush=True)
    print(f"[R3] VERDICT: {rep['verdict']}", flush=True)

    p = ROOT / a.out / 'R3_inversion.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[R3] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

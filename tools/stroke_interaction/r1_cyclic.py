"""R1: does VAE error accumulate over repeated round-trips?

    x1 = D(E(x_GT)),  x2 = D(E(x1)),  x3 = D(E(x2))

If the first round-trip changes the topology count a lot but later rounds are
stable, the loss is a one-time mapping effect. If it keeps getting worse, the
round-trip map itself is unstable.

Note on polarity and scaling, because a wrong version of this produced a
nonsense result earlier: the SD VAE expects the RAW scan in [-1,1] (bright
paper), encode output is multiplied by 0.18215 and divided again before decode,
and the decoded image is returned in the same raw polarity. Metrics are then
computed in ink space (ink = 1 - raw when the raw image is mostly bright).

Everything is reported per character with clustered CIs, and the topology count
is the pruned multi-threshold version from R2, not raw skeleton junctions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stroke_interaction.r2_blur import robust_topology, THRS  # noqa: E402


def to_ink(raw):
    return 1.0 - raw if raw.mean() > 0.5 else raw


def boot(vals, n=800, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, np.float64)
    if len(v) < 3:
        return None
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='exp-std/csv/train_top10_aug_sym.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--chars', type=int, default=20)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    import csv
    from PIL import Image
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(ROOT / 'pretrained_models/sd-vae-ft-ema',
                                        torch_dtype=torch.bfloat16,
                                        local_files_only=True).cuda().eval()

    @torch.no_grad()
    def rt(raw):
        t = torch.from_numpy(raw)[None, None].cuda().float() * 2 - 1
        t = t.repeat(1, 3, 1, 1).to(torch.bfloat16)
        lat = vae.encode(t).latent_dist.mode() * 0.18215
        x = vae.decode((lat.float() / 0.18215).to(torch.bfloat16)).sample
        return ((x.float().clamp(-1, 1) + 1) / 2)[0].mean(0).cpu().numpy()

    rows = [r for r in csv.DictReader(open(ROOT / a.csv, encoding='utf-8'))
            if (ROOT / r['image_path']).exists()]
    rng = np.random.default_rng(a.seed)
    by = defaultdict(list)
    for r in rows:
        by[(r['script'], r['character'])].append(r)
    cand = [k for k, v in by.items() if len({r['calligrapher'] for r in v}) >= 3]
    rng.shuffle(cand)
    cand = cand[:a.chars]
    print(f'[R1] {len(cand)} characters', flush=True)

    rep = {'n_chars': len(cand), 'rounds': {}}
    per_char = defaultdict(list)
    lat_drift = defaultdict(list)
    for ci, (sc, ch) in enumerate(cand):
        for r in by[(sc, ch)][:4]:
            raw = np.array(Image.open(ROOT / r['image_path']).convert('L'),
                           np.float32) / 255.0
            chain = [raw]
            lats = []
            for _ in range(3):
                prev = chain[-1]
                t = torch.from_numpy(prev)[None, None].cuda().float() * 2 - 1
                t = t.repeat(1, 3, 1, 1).to(torch.bfloat16)
                with torch.no_grad():
                    lat = vae.encode(t).latent_dist.mode() * 0.18215
                    x = vae.decode((lat.float() / 0.18215).to(torch.bfloat16)).sample
                nxt = ((x.float().clamp(-1, 1) + 1) / 2)[0].mean(0).cpu().numpy()
                lats.append(lat.float().flatten().cpu().numpy())
                chain.append(nxt)
            ink0 = to_ink(chain[0])
            for rnd in range(1, 4):
                ink = to_ink(chain[rnd])
                l1 = float(np.abs(ink - ink0).mean())
                gm = float(np.hypot(*np.gradient(ink)).mean())
                jc = [robust_topology(ink, thr) for thr in THRS]
                jc = [t['junction'] for t in jc if t]
                per_char[(rnd, 'l1')].append((l1, ci))
                per_char[(rnd, 'grad')].append((gm, ci))
                per_char[(rnd, 'junc')].append((float(np.mean(jc)), ci))
                if rnd > 1:
                    d = float(np.linalg.norm(lats[rnd - 1] - lats[rnd - 2]))
                    lat_drift[rnd].append((d, ci))
        if (ci + 1) % 5 == 0:
            print(f'[R1] {ci+1}/{len(cand)}', flush=True)

    for rnd in (1, 2, 3):
        e = {}
        for key in ('l1', 'grad', 'junc'):
            vs = [v for v, _ in per_char[(rnd, key)]]
            gs = [g for _, g in per_char[(rnd, key)]]
            if not vs:
                continue
            e[key] = {'mean': round(float(np.mean(vs)), 4), 'ci95': boot(vs)}
        if rnd > 1 and lat_drift.get(rnd):
            vs = [v for v, _ in lat_drift[rnd]]
            e['latent_drift'] = {'mean': round(float(np.mean(vs)), 4), 'ci95': boot(vs)}
        rep['rounds'][f'RT{rnd}'] = e
        print(f'[R1] RT{rnd}: {e}', flush=True)

    r1 = rep['rounds']['RT1']
    r2 = rep['rounds']['RT2']
    r3 = rep['rounds']['RT3']
    rep['verdict'] = {
        'junction_rt1_vs_rt2': round(r2['junc']['mean'] - r1['junc']['mean'], 3),
        'junction_rt2_vs_rt3': round(r3['junc']['mean'] - r2['junc']['mean'], 3),
        'stable_after_first': bool(abs(r3['junc']['mean'] - r2['junc']['mean']) <
                                   max(0.15 * abs(r2['junc']['mean']), 0.5)),
        'l1_rt3_vs_rt1': round(r3['l1']['mean'] - r1['l1']['mean'], 4),
    }
    print(f"[R1] VERDICT: {rep['verdict']}", flush=True)

    p = ROOT / a.out / 'R1_cyclic.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[R1] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

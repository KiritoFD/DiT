"""C: for each sample, how much of the REAL writer difference does the model
produce -- in strength AND in the right direction?

The earlier F4 reported cos(mean GT diff, mean Gen diff) and a magnitude ratio.
The review correctly pointed out that multiplying those two population means is
not a per-sample projection, and that a low cosine means the direction itself is
wrong, not merely weak. So here, for every (character, writer-pair) sample:

    d_GT   real writer difference vector at that character
    d_Gen  generated writer difference vector for the same pair
    proj   <d_Gen, d_GT> / ||d_GT||^2      (1.0 = exactly reproduced strength)
    orth   ||d_Gen - proj*d_GT|| / ||d_Gen||  (fraction pointing elsewhere)
    cos    <d_Gen, d_GT> / (||d_Gen|| ||d_GT||)

proj is the quantity the review asked for, averaged over samples rather than
over population means. A model with correct direction but weak strength has
proj ~ 0.6 and orth ~ 0; a model with wrong direction has orth ~ 1.

Reported per event kind, and separately for the features F2 found most
discriminative (n_branches, end_area) versus the rest, so we can say whether
the miss is specific to the topology/endpoint family or global.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

E_KINDS = ['end', 'junction', 'turn']
GEO = ['asym_ud', 'asym_lr', 'grad_mag', 'turn_angle', 'end_area', 'end_elong',
       'n_branches', 'orient_peak']
ALLF = GEO + ['mass', 'ink_frac']
# the families F2 flagged as most writer-discriminative
KEY_FEATS = ['n_branches', 'end_area']


def event_means(img, rng, per_kind=20):
    from tools.stroke_interaction.features import extract_events
    a = img.astype(np.float32)
    ink = 1.0 - a if a.mean() > 0.5 else a
    b = (ink > 0.5).astype(np.uint8)
    if b.sum() < 64:
        return None
    ev, _, _ = extract_events(ink, b, rng, per_kind=per_kind)
    out = defaultdict(list)
    for e in ev:
        out[e['kind']].append([e.get(k, 0.0) for k in ALLF])
    return {k: np.array(v, np.float32).mean(0) for k, v in out.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default='exp-std/csv/train_top10_aug_sym.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--chars', type=int, default=30)
    ap.add_argument('--per-kind', type=int, default=20)
    ap.add_argument('--raw', action='store_true')
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()

    from PIL import Image
    from diffusers import AutoencoderKL
    from src.unet.io import load, diffusion
    from src.eval.sampler import sample_latents
    from tools.stroke_interaction.f4_gt_vs_gen import std_path

    rows = [r for r in csv.DictReader(open(ROOT / a.csv, encoding='utf-8'))
            if (ROOT / r['image_path']).exists()]
    ck_dir = (ROOT / a.ckpt).parents[1]
    cfg = json.loads((ck_dir / 'resolved_config.json').read_text(encoding='utf-8'))
    callig_remap = json.loads((ROOT / cfg['callig_remap_json']).read_text(encoding='utf-8'))
    font_remap = json.loads((ROOT / cfg['font_remap_json']).read_text(encoding='utf-8'))

    model, config, ck = load(a.ckpt, 'cuda', use_ema=not a.raw)
    process = diffusion(config)
    vae = AutoencoderKL.from_pretrained(config.vae_path, torch_dtype=torch.bfloat16,
                                        local_files_only=True).cuda().eval()
    rng = np.random.default_rng(a.seed)

    by_char = defaultdict(set)
    rep_row = {}
    for r in rows:
        by_char[(r['script'], r['character'])].add(r['calligrapher'])
        rep_row.setdefault((r['script'], r['character'], r['calligrapher']), r)
    cand = [(k, w) for k, w in by_char.items() if len(w) >= 4 and std_path(*k)]
    rng.shuffle(cand)
    cand = cand[:a.chars]
    print(f'[C] {len(cand)} characters', flush=True)

    per_kind = defaultdict(lambda: defaultdict(list))
    per_featgroup = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (sc, ch), writes in cand:
        # --- GT means per writer ---
        gt = {}
        for w in sorted(writes):
            rs = [r for r in rows if r['script'] == sc and r['character'] == ch
                  and r['calligrapher'] == w][:3]
            vs = []
            for r in rs:
                arr = np.array(Image.open(ROOT / r['image_path']).convert('L'),
                               np.float32) / 255.0
                m = event_means(arr, rng, a.per_kind)
                if m:
                    vs.append(m)
            if vs:
                acc = defaultdict(list)
                for m in vs:
                    for k, v in m.items():
                        acc[k].append(v)
                gt[w] = {k: np.mean(v, 0) for k, v in acc.items()}
        # --- generated means per writer ---
        gen = {}
        g = np.array(Image.open(std_path(sc, ch)).convert('L'), np.float32) / 255.0
        g64 = torch.from_numpy(g)[None, None]
        for w in sorted(writes):
            row = rep_row.get((sc, ch, w))
            if row is None:
                continue
            wid = int(callig_remap.get(str(row['calligrapher_id']), -1))
            sid = int(font_remap.get(str(row['script_id']), 0))
            if wid < 0:
                continue
            noise = torch.randn(1, 4, 32, 32, generator=torch.Generator()
                                .manual_seed(a.seed))
            with torch.no_grad():
                lat = sample_latents(model, process, noise, [(wid, 0)], 1.0, 1, 'cuda',
                                     skel=g64, seed=a.seed,
                                     hier_conds=[[wid], [0], [sid]])
                x = vae.decode((lat.cuda().float() / 0.18215).to(torch.bfloat16)).sample
            m = event_means(((x.float().clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy(),
                            rng, a.per_kind)
            if m:
                gen[w] = m
        ws = sorted(set(gt) & set(gen))
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                for k in E_KINDS:
                    g1, g2 = gt[ws[i]].get(k), gt[ws[j]].get(k)
                    n1, n2 = gen[ws[i]].get(k), gen[ws[j]].get(k)
                    if any(v is None for v in (g1, g2, n1, n2)):
                        continue
                    dG, dN = g1 - g2, n1 - n2
                    nn = float(dG @ dG)
                    if nn < 1e-12:
                        continue
                    proj = float(dN @ dG / nn)
                    resid = dN - proj * dG
                    orth = float(np.linalg.norm(resid) / max(np.linalg.norm(dN), 1e-12))
                    cos = float(dG @ dN / (np.linalg.norm(dG) * max(np.linalg.norm(dN), 1e-12)))
                    per_kind[k]['proj'].append(proj)
                    per_kind[k]['orth'].append(orth)
                    per_kind[k]['cos'].append(cos)
                    # split by feature family
                    idxk = [ALLF.index(f) for f in KEY_FEATS]
                    idxo = [i for i in range(len(ALLF)) if i not in idxk]
                    for tag, sel in (('key_feats', idxk), ('rest_feats', idxo)):
                        vg, vn = dG[sel], dN[sel]
                        vv = float(vg @ vg)
                        if vv < 1e-12:
                            continue
                        p = float(vn @ vg / vv)
                        o = float(np.linalg.norm(vn - p * vg) /
                                  max(np.linalg.norm(vn), 1e-12))
                        per_featgroup[tag][k]['proj'].append(p)
                        per_featgroup[tag][k]['orth'].append(o)

    def agg(d):
        return {k: {'n': len(v['proj']),
                    'proj_mean': round(float(np.mean(v['proj'])), 4),
                    'proj_median': round(float(np.median(v['proj'])), 4),
                    'orth_mean': round(float(np.mean(v['orth'])), 4),
                    'cos_mean': round(float(np.mean(v['cos'])), 4)}
                for k, v in d.items() if v['proj']}

    rep = {'ckpt': a.ckpt, 'n_chars': len(cand),
           'by_kind': agg(per_kind),
           'by_feature_group': {tag: agg(d) for tag, d in per_featgroup.items()},
           'key_features': KEY_FEATS}
    for k, v in rep['by_kind'].items():
        print(f'[C] {k:9s} n={v["n"]:4d} proj={v["proj_mean"]:+.3f} '
              f'orth={v["orth_mean"]:.3f} cos={v["cos_mean"]:+.3f}', flush=True)
    for tag, d in rep['by_feature_group'].items():
        for k, v in d.items():
            print(f'[C] {tag:10s} {k:9s} n={v["n"]:4d} proj={v["proj_mean"]:+.3f} '
                  f'orth={v["orth_mean"]:.3f}', flush=True)

    p = ROOT / a.out / 'C_generation_projection.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[C] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

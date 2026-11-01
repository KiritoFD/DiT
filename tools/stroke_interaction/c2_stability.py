"""C2: is the negative projection on n_branches / end_area a detector artefact?

C1 showed the VAE arm alone loses most of the topological signal, which already
suggests the miss is upstream of the U-Net. This script attacks the other
suspicion: that n_branches / end_area are simply unstable under threshold and
resolution, so a negative sign could be noise.

For each (threshold, resolution) we recompute the per-kind projection of the
GT writer difference onto the generated one, on the SAME character set, and
report the sign and the clustered CI. A sign that flips across settings means
the negative projection cannot be interpreted as a property of the model.
"""
from __future__ import annotations

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

GEO = ['asym_ud', 'asym_lr', 'grad_mag', 'turn_angle', 'end_area', 'end_elong',
       'n_branches', 'orient_peak']
ALLF = GEO + ['mass', 'ink_frac']
KEY_FEATS = ['n_branches', 'end_area']
SCRIPT_CODE = {'楷': 'kai', '行': 'xing', '隶': 'li'}


def std_png(script, character):
    p = ROOT / 'exp-std/data/std_soft_glyphs_v1' / SCRIPT_CODE[script] / \
        f'U+{ord(character):05X}.png'
    return p if p.exists() else None


def prep(arr, size, thr):
    from PIL import Image
    if arr.shape[0] != size:
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        arr = np.array(im.resize((size, size), Image.BICUBIC), np.float32) / 255.0
    ink = 1.0 - arr if arr.mean() > 0.5 else arr
    return ink, (ink > thr).astype(np.uint8)


def kind_means(ink, b, rng):
    from tools.stroke_interaction.features import extract_events
    ev, _, _ = extract_events(ink, b, rng, per_kind=60)
    out = defaultdict(list)
    for e in ev:
        out[e['kind']].append([e.get(k, 0.0) for k in ALLF])
    return {k: np.array(v, np.float32).mean(0) for k, v in out.items() if v}


def boot(vals, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, np.float64)
    if len(v) < 3:
        return None
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default='exp-std/csv/train_top10_aug_sym.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--chars', type=int, default=24)
    ap.add_argument('--raw', action='store_true')
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()

    from PIL import Image
    from diffusers import AutoencoderKL
    from src.unet.io import load, diffusion
    from src.eval.sampler import sample_latents

    rows = [r for r in csv.DictReader(open(ROOT / a.csv, encoding='utf-8'))
            if (ROOT / r['image_path']).exists()]
    ck_dir = (ROOT / a.ckpt).parents[1]
    cfg = json.loads((ck_dir / 'resolved_config.json').read_text(encoding='utf-8'))
    callig = json.loads((ROOT / cfg['callig_remap_json']).read_text(encoding='utf-8'))
    font = json.loads((ROOT / cfg['font_remap_json']).read_text(encoding='utf-8'))

    model, config, _ = load(a.ckpt, 'cuda', use_ema=not a.raw)
    process = diffusion(config)
    vae = AutoencoderKL.from_pretrained(config.vae_path, torch_dtype=torch.bfloat16,
                                        local_files_only=True).cuda().eval()
    rng = np.random.default_rng(a.seed)
    by_char = defaultdict(list)
    for r in rows:
        by_char[(r['script'], r['character'])].append(r)
    cand = [k for k, v in by_char.items()
            if std_png(*k) and len({r['calligrapher'] for r in v}) >= 4]
    rng.shuffle(cand)
    cand = cand[:a.chars]
    print(f'[C2] {len(cand)} characters', flush=True)

    # generate once per (char, writer) at native 256, then resample analytically
    gen_cache = {}
    gt_cache = {}
    styles = {}
    for ci, (sc, ch) in enumerate(cand):
        rs = by_char[(sc, ch)]
        g = np.array(Image.open(std_png(sc, ch)).convert('L'), np.float32) / 255.0
        g64 = torch.from_numpy(g)[None, None]
        writers = sorted({r['calligrapher'] for r in rs})
        for w in writers:
            wr = [r for r in rs if r['calligrapher'] == w][:3]
            gt_cache[(ci, w)] = [
                np.array(Image.open(ROOT / r['image_path']).convert('L'),
                         np.float32) / 255.0 for r in wr]
            row = wr[0]
            wid = int(callig.get(str(row['calligrapher_id']), -1))
            sid = int(font.get(str(row['script_id']), 0))
            if wid < 0:
                continue
            with torch.no_grad():
                noise = torch.randn(1, 4, 32, 32,
                                    generator=torch.Generator()
                                    .manual_seed(a.seed + ci + hash(w) % 97))
                lat = sample_latents(model, process, noise, [(wid, 0)], 1.0, 1,
                                     'cuda', skel=g64, seed=a.seed + ci,
                                     hier_conds=[[wid], [0], [sid]])
                x = vae.decode((lat.cuda().float() / 0.18215)
                               .to(torch.bfloat16)).sample
            gen_cache[(ci, w)] = ((x.float().clamp(-1, 1) + 1) / 2)[0].mean(
                0).cpu().numpy()
        styles[ci] = (sc, ch, writers)
        if (ci + 1) % 6 == 0:
            print(f'[C2] generated {ci+1}/{len(cand)}', flush=True)

    CONFIGS = [(0.35, 256), (0.50, 256), (0.65, 256),
               (0.50, 128), (0.50, 192), (0.50, 320)]
    rep = {'configs': {}, 'n_chars': len(cand)}
    for thr, size in CONFIGS:
        per = defaultdict(lambda: defaultdict(list))   # kind -> group -> [proj]
        for ci, (sc, ch, writers) in styles.items():
            gt_m, gen_m = {}, {}
            for w in writers:
                for src, store in (('gt', gt_m), ('gen', gen_m)):
                    arrs = gt_cache[(ci, w)] if src == 'gt' else \
                        ([gen_cache[(ci, w)]] if (ci, w) in gen_cache else [])
                    if not arrs:
                        continue
                    acc = defaultdict(list)
                    for arr in arrs:
                        ink, b = prep(arr, size, thr)
                        if b.sum() < 64:
                            continue
                        for k, v in kind_means(ink, b,
                                               np.random.default_rng(1)).items():
                            acc[k].append(v)
                    if acc:
                        store[w] = {k: np.mean(v, 0) for k, v in acc.items()}
            ws = sorted(set(gt_m) & set(gen_m))
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    for k in ('end', 'junction', 'turn'):
                        if k not in gt_m[ws[i]] or k not in gen_m[ws[i]] or \
                                k not in gt_m[ws[j]] or k not in gen_m[ws[j]]:
                            continue
                        dG = gt_m[ws[i]][k] - gt_m[ws[j]][k]
                        dN = gen_m[ws[i]][k] - gen_m[ws[j]][k]
                        nn = float(dG @ dG)
                        if nn < 1e-12:
                            continue
                        per[k]['all'].append(float(dN @ dG / nn))
                        ik = [t for t, f in enumerate(ALLF) if f in KEY_FEATS]
                        io = [t for t in range(len(ALLF)) if t not in ik]
                        if float(dG[ik] @ dG[ik]) > 1e-12:
                            per[k]['key'].append(float(dN[ik] @ dG[ik] /
                                                       (dG[ik] @ dG[ik])))
                        if float(dG[io] @ dG[io]) > 1e-12:
                            per[k]['rest'].append(float(dN[io] @ dG[io] /
                                                        (dG[io] @ dG[io])))
        entry = {}
        for k, g in per.items():
            entry[k] = {name: {'mean': round(float(np.mean(v)), 4),
                               'ci95': boot(v)} for name, v in g.items() if v}
        rep['configs'][f'thr{thr}_size{size}'] = entry
        print(f'[C2] thr={thr} size={size}: '
              + ' | '.join(f'{k} all={entry[k]["all"]["mean"]:+.3f} '
                           f'key={entry[k].get("key", {}).get("mean", float("nan")):+.3f}'
                           for k in entry), flush=True)

    # sign stability
    print('\n[C2] sign stability of the KEY-feature projection:', flush=True)
    stab = {}
    for k in ('end', 'junction', 'turn'):
        signs = [rep['configs'][c][k]['key']['mean']
                 for c in rep['configs'] if k in rep['configs'][c] and
                 'key' in rep['configs'][c][k]]
        if signs:
            stab[k] = {'values': signs, 'n_negative': sum(1 for s in signs if s < 0),
                       'n_configs': len(signs),
                       'stable_sign': len({s < 0 for s in signs}) == 1}
            print(f'  {k:9s} {stab[k]}', flush=True)
    rep['key_feature_sign_stability'] = stab

    p = ROOT / a.out / 'C2_detector_stability.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[C2] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""G1 second half: does the model reproduce the GT writer difference, per scale?

g1_audit.py stopped at "correct writer gives lower FM loss than a wrong writer at
t>=0.55". That only says conditioning helps. The plan asks sharper: of the real
writer differences Delta^GT_AB(c), which ones does the model actually reproduce?
So:

  1. generate the SAME noise/glyph under writer A and under writer B -> Delta^Gen_AB(c)
  2. measure Delta^GT_AB(c) on the real images of the same two writers
  3. split both into the plan's three scales and compare DIRECTIONS

Direction agreement is the point: a model that moves the right features the right
way is learning writer style; one that moves nothing, or the wrong way, is not,
regardless of FM loss.

Uses the project's own src/eval/sampler.sample_latents so the sampling path is
the same one every evaluation in this repo uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEYS = ['ink_frac', 'width_med', 'cx', 'cy',
        'edge_mean', 'orient_entropy', 'orient_peak',
        'branch_density', 'end_density', 'n_components']

GROUPS = {
    'global_mass_width_centroid': ['ink_frac', 'width_med', 'cx', 'cy'],
    'local_edge_orientation': ['edge_mean', 'orient_entropy', 'orient_peak'],
    'structure_connectivity': ['branch_density', 'end_density', 'n_components'],
}


def feats(img):
    """Three-scale descriptors on ink-in-[0,1] grayscale (1 = full ink)."""
    g = np.clip(np.asarray(img, np.float64), 0, 1)
    if g.ndim == 3:
        g = g.mean(0)
    ink = g > 0.5
    n_ink = int(ink.sum())
    if n_ink < 24:
        return None
    ys, xs = np.nonzero(ink)
    # stroke calibre proxy: ink area / number of inked columns
    cols = int((ink.sum(axis=0) > 0).sum())
    out = {
        'ink_frac': float(g.mean()),
        'width_med': float(n_ink / max(cols, 1)),
        'cx': float(ys.mean()) / g.shape[0],
        'cy': float(xs.mean()) / g.shape[1],
    }
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    out['edge_mean'] = float(mag.mean())
    ang = np.arctan2(gy, gx)
    m = mag > np.percentile(mag, 85)
    if m.sum() > 32:
        h, _ = np.histogram(ang[m] % np.pi, bins=8, range=(0, np.pi), weights=mag[m])
        h = h / (h.sum() + 1e-9)
        out['orient_entropy'] = float(-(h * np.log(h + 1e-9)).sum())
        out['orient_peak'] = float(h.max())
    else:
        out['orient_entropy'] = 0.0
        out['orient_peak'] = 0.0
    from scipy import ndimage as ndi
    sk = ink.astype(np.uint8)
    nb = ndi.convolve(sk, np.ones((3, 3), np.uint8), mode='constant') - sk
    out['branch_density'] = float(((sk == 1) & (nb >= 3)).mean())
    out['end_density'] = float(((sk == 1) & (nb == 1)).mean())
    _, n = ndi.label(sk)
    out['n_components'] = float(n)
    return out


def stack(arr):
    fs = [feats(im) for im in arr]
    keep = np.array([i for i, f in enumerate(fs) if f is not None])
    M = np.array([[fs[i][k] for k in KEYS] for i in keep], np.float64)
    return M, keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default='exp-std/csv/eval200_fixed.csv')
    ap.add_argument('--out', default='assets/style_24h_20261026/G1')
    ap.add_argument('--limit', type=int, default=32)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    out = ROOT / a.out
    from src.unet.io import load, diffusion
    from src.eval.sampler import sample_latents
    from tools.style_24h4090.g1_audit import load_inputs, writer_rows

    model, config, ck = load(a.ckpt, 'cuda', use_ema=True)
    process = diffusion(config)
    rows = load_inputs(ROOT / a.csv, a.limit, ROOT)
    wrow, _ = writer_rows(ROOT)
    swap = json.loads((out / 'swap_pairs.json').read_text())
    true_name, wrong_name, chars = swap['true'], swap['wrong'], swap['characters']
    print(f'[g1b] {len(rows)} rows, {len(set(true_name))} true / {len(set(wrong_name))} wrong writers',
          flush=True)

    from diffusers import AutoencoderKL
    import torch.nn.functional as Fn
    from PIL import Image

    def load_gt_ink(paths, size=224):
        """GT image -> ink-in-[0,1] grayscale, same crop convention as everywhere else."""
        outs = []
        for p in paths:
            p = Path(p)
            p = p if p.is_absolute() else ROOT / p
            im = Image.open(p).convert('RGBA')
            arr = np.array(im).astype(np.float32)
            al = arr[..., 3:4] / 255.0
            rgb = arr[..., :3] * al + 255.0 * (1 - al)
            g = rgb.mean(-1)
            ink = g < 190
            if ink.sum() > 16:
                ys, xs = np.nonzero(ink)
                mh = int((ys.max() - ys.min() + 1) * .08)
                mw = int((xs.max() - xs.min() + 1) * .08)
                g = g[max(0, ys.min() - mh):ys.max() + 1 + mh,
                      max(0, xs.min() - mw):xs.max() + 1 + mw]
            im2 = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).resize((size, size),
                                                                             Image.BICUBIC)
            outs.append(np.array(im2, np.float32) / 255.0)
        return np.stack(outs)

    vae = AutoencoderKL.from_pretrained(config.vae_path, torch_dtype=torch.bfloat16,
                                        local_files_only=True).cuda().eval()
    data = np.load(out / 'gt_latents.npz')
    G = Fn.interpolate(torch.from_numpy(data['g']).float().unsqueeze(1), (64, 64),
                       mode='bilinear', align_corners=False)
    # The std-soft condition must cover exactly the rows we sample, so the latent
    # cache decides the limit -- do not let --limit disagree with it.
    n_cache = int(data['z'].shape[0])
    if a.limit != n_cache:
        print(f'[g1b] --limit {a.limit} != cached latents {n_cache}; using {n_cache}', flush=True)
        a.limit = n_cache
        rows = rows[:n_cache]

    # character ids come straight from the trained char table
    char_ids = []
    for r in rows:
        cid = int(r.get('character_id', -1))
        char_ids.append(max(cid, 0))

    n = len(rows)
    noise = torch.randn(n, 4, 32, 32, generator=torch.Generator().manual_seed(a.seed))

    def run(writer_name_list, tag):
        conds = [(wrow[w], char_ids[i]) for i, w in enumerate(writer_name_list)]
        # The 114M U-Net requires the script axis explicitly; without hier_conds
        # it raises "Script condition required" (G1B died this way in 7 s).
        # P2's script table has 4 rows with num_scripts=3, so the RAW ids
        # {0,3,4} must be remapped to {0,1,2} -- passing 3 or 4 is out of range
        # and produced a device-side assert in the 48-side equivalent.
        font = json.loads((ROOT / 'assets/triple_tables_best_minimal/font_remap.json')
                          .read_text(encoding='utf-8'))
        script_ids = [int(font.get(str(r.get('script_id', 0)), 0)) for r in rows]
        z = sample_latents(model, process, noise, conds, a.cfg, a.batch, 'cuda',
                           skel=G, seed=a.seed,
                           hier_conds=[[c[0] for c in conds], char_ids, script_ids])
        imgs = []
        for s in range(0, n, a.batch):
            lat = z[s:s + a.batch].cuda().float() / 0.18215
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                x = vae.decode(lat.to(torch.bfloat16)).sample
            x = ((x.float().clamp(-1, 1) + 1) / 2)[:, :1].cpu().numpy()
            imgs.append(1.0 - x[:, 0])          # ink = 1
        arr = np.concatenate(imgs)
        print(f'[g1b] {tag}: decoded {arr.shape}', flush=True)
        return arr

    genA = run(true_name, 'writerA')
    genB = run(wrong_name, 'writerB')

    gt = load_gt_ink([r['image_path'] for r in rows])   # white=0 ink=1
    gt = gt[:, None]

    MA, kA = stack(genA)
    MB, kB = stack(genB)
    MT, kT = stack(gt)
    keep = np.intersect1d(np.intersect1d(kA, kB), kT)
    MA, MB, MT = MA[np.isin(kA, keep)], MB[np.isin(kB, keep)], MT[np.isin(kT, keep)]
    m = min(len(MA), len(MB), len(MT))
    MA, MB, MT = MA[:m], MB[:m], MT[:m]
    print(f'[g1b] usable rows {m}', flush=True)
    if m < 8:
        print('[g1b] too few usable rows, aborting rather than reporting noise', flush=True)
        return 1

    sd = MT.std(0) + 1e-9
    dGT = (MT - MT.mean(0, keepdims=True)) / sd          # GT writer-A deviation
    dGen = (MA - MB) / sd                                 # generated A minus B

    rep = {'n_rows': int(m), 'cfg': a.cfg, 'seed': a.seed, 'ckpt': str(a.ckpt),
           'note': ('Delta^GT is writer-A deviation from the population mean of the '
                    'same characters; Delta^Gen is generated A minus generated B under '
                    'identical noise and glyph. Directions compared per scale.'),
           'scales': {}}
    for gname, cols in GROUPS.items():
        idx = [KEYS.index(c) for c in cols]
        vT, vG = dGT[:, idx], dGen[:, idx]
        cos = np.array([float(vT[i] @ vG[i] / (np.linalg.norm(vT[i]) * np.linalg.norm(vG[i]) + 1e-12))
                        for i in range(m)])
        cos_mean = float(vT.mean(0) @ vG.mean(0) /
                         (np.linalg.norm(vT.mean(0)) * np.linalg.norm(vG.mean(0)) + 1e-12))
        move = float(np.linalg.norm(vG.mean(0)) / (np.linalg.norm(vT.mean(0)) + 1e-12))
        rep['scales'][gname] = {
            'features': cols,
            'cosine_mean_direction': round(cos_mean, 4),
            'cosine_per_row_median': round(float(np.median(cos)), 4),
            'cosine_positive_frac': round(float((cos > 0).mean()), 4),
            'generated_movement_ratio': round(move, 4),
        }
        print(f'[g1b] {gname:28s} cos_mean={cos_mean:+.4f} med={np.median(cos):+.4f} '
              f'pos={float((cos>0).mean()):.2f} move={move:.3f}', flush=True)

    rep['per_feature'] = {
        KEYS[i]: {'gt_delta': float(dGT[:, i].mean()),
                  'gen_delta': float(dGen[:, i].mean()),
                  'same_sign': bool(np.sign(dGT[:, i].mean()) == np.sign(dGen[:, i].mean()))}
        for i in range(len(KEYS))}
    mean_cos = float(np.mean([s['cosine_mean_direction'] for s in rep['scales'].values()]))
    rep['aggregate'] = {
        'mean_cosine_across_scales': round(mean_cos, 4),
        'scales_positive': int(sum(1 for s in rep['scales'].values()
                                   if s['cosine_mean_direction'] > 0)),
        'verdict': ('model reproduces GT writer direction' if mean_cos > 0.3 else
                    'model does NOT reproduce GT writer direction'),
    }
    (out / 'g1_gt_vs_gen.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f'[g1b] VERDICT {rep["aggregate"]["verdict"]} (mean cos {mean_cos:+.4f})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

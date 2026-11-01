"""F5 gradient pilot: can a physical stroke-statistic target actually guide the U-Net?

The plan forbids shipping a loss just because a statistic looked good. Before any
training we ask whether the gradient points somewhere useful:

  * it must move toward the REAL target statistics, and
  * an INDEPENDENT feature family must agree, and
  * ink amount / stroke width / content structure must not degrade, and
  * it must not be pure high-frequency texture.

DIFFERENTIABILITY. The measurement path in features.py uses skeletonisation and
hard thresholds and therefore has NO gradient -- a first attempt at this file
silently optimised nothing because loss.requires_grad was always False. So the
loss here is computed with SMOOTH surrogates of the same physical quantities:

    soft ink        sigma-predicted ink      I = sigmoid((v - thr) / tau)
    soft width      1 - exp(-alpha * local ink mass)   (mass grows with width)
    soft direction  structure tensor of the ink, its anisotropy and angle
    soft richness   local ink variance (proxy for branched / turned regions)

Each surrogate is validated against the hard extractor on REAL images before it
is allowed into the loss: if the soft width does not correlate with the measured
skeleton width, the surrogate is wrong and no gradient conclusion is valid.

Optimisation touches the LATENT, not the weights, with an identical step budget
for every candidate, so differences between C0/C1/C2 are differences in gradient
direction, not in capacity.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

E_KINDS = ['end', 'junction', 'turn']
# the loss sees only FAM_A; FAM_B and FAM_C are independent judges
FAM_A = ['soft_width', 'soft_aniso', 'soft_richness', 'soft_ink']
FAM_B = ['width_p50', 'ink_frac', 'grad_mag']
FAM_C = ['n_branches', 'orient_entropy', 'end_area']

# Measured on real images (30 samples) so judge changes can be quoted in units
# of the natural spread instead of raw feature units, which differ by ~100x.
NATURAL_SD = {'width_p50': 1.064, 'ink_frac': 0.067, 'grad_mag': 0.028,
              'n_branches': 0.113, 'orient_entropy': 0.204, 'end_area': 15.028}

TAU = 0.06          # soft-threshold temperature
ALPHA = 2.5         # width saturation
# 3x3 Sobel, shape (out=1, in=1, kH, kW)
KX = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])
KY = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])


def blur(x, k=5, sigma=1.2):
    if k <= 1:
        return x
    g = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    w = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    w = w / w.sum()
    ker = (w[:, None] * w[None, :]).expand(x.shape[1], 1, k, k).contiguous()
    return Fn.conv2d(x, ker, padding=k // 2, groups=x.shape[1])


def soft_stats(img):
    """Smooth, differentiable physical statistics of one (H,W) ink image."""
    x = img[None, None]
    ink = torch.sigmoid((x - 0.5) / TAU)                  # soft ink mask
    mass = blur(ink, 7, 2.0)                              # local ink amount
    width = 1.0 - torch.exp(-ALPHA * mass)                # grows with stroke width
    gx = Fn.conv2d(ink, KX.to(ink.device, ink.dtype), padding=1)
    gy = Fn.conv2d(ink, KY.to(ink.device, ink.dtype), padding=1)
    gxv = blur(gx, 7, 2.0)
    gyv = blur(gy, 7, 2.0)
    Jxx = (gxv * gxv).mean()
    Jyy = (gyv * gyv).mean()
    Jxy = (gxv * gyv).mean()
    tr = Jxx + Jyy + 1e-8
    aniso = torch.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / tr     # direction coherence
    richness = blur(ink, 7, 2.0).var()                           # branched/turned areas
    return {
        'soft_ink': ink.mean(),
        'soft_width': width.mean(),
        'soft_aniso': aniso.mean(),
        'soft_richness': richness,
    }


def hard_stats(img01, rng, per_kind=16):
    """Non-differentiable measurement, used only for validation and judging."""
    from tools.stroke_interaction.features import extract_events, glyph_features
    ink = img01.astype(np.float32)
    binary = (ink > 0.5).astype(np.uint8)
    if binary.sum() < 64:
        return None
    G = glyph_features(ink, binary)
    if G is None:
        return None
    events, _, _ = extract_events(ink, binary, rng, per_kind=per_kind)
    out = {'width_p50': G['width_p50'], 'ink_frac': G['ink_frac'],
           'grad_mag': 0.0, 'n_branches': 0.0, 'orient_entropy': 0.0, 'end_area': 0.0}
    if events:
        gm = [e.get('grad_mag', 0) for e in events]
        out['grad_mag'] = float(np.mean(gm))
        br = [e.get('n_branches', 0) for e in events if e['kind'] == 'junction']
        out['n_branches'] = float(np.mean(br)) if br else 0.0
        oe = [e.get('orient_entropy', 0) for e in events if e['kind'] in ('turn', 'junction')]
        out['orient_entropy'] = float(np.mean(oe)) if oe else 0.0
        ea = [e.get('end_area', 0) for e in events if e['kind'] == 'end']
        out['end_area'] = float(np.mean(ea)) if ea else 0.0
    return out


def validate_surrogates(rows, rng, n=40):
    """Soft width must track the measured skeleton width on REAL images."""
    from tools.stroke_interaction.features import load_ink
    from PIL import Image
    pairs = []
    for r in rows[:n]:
        arr = np.array(Image.open(ROOT / r['image_path']).convert('L'), np.float32) / 255.0
        ink = 1.0 - arr if arr.mean() > 0.5 else arr
        s = hard_stats(ink, rng)
        if s is None:
            continue
        t = torch.from_numpy(ink.astype(np.float32)).cuda()
        with torch.no_grad():
            ss = soft_stats(t)
        pairs.append((float(ss['soft_width']), s['width_p50'],
                      float(ss['soft_ink']), s['ink_frac']))
    P = np.array(pairs)
    if len(P) < 8:
        return {'ok': False, 'n': len(P)}
    cw = float(np.corrcoef(P[:, 0], P[:, 1])[0, 1])
    ci = float(np.corrcoef(P[:, 2], P[:, 3])[0, 1])
    return {'ok': cw > 0.5 and ci > 0.8, 'n': len(P),
            'corr_soft_width_vs_measured': round(cw, 3),
            'corr_soft_ink_vs_measured': round(ci, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default='exp-std/csv/train_top10_aug_sym.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--n-sample', type=int, default=12)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--raw', action='store_true')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    import csv
    from PIL import Image
    from diffusers import AutoencoderKL
    from src.unet.io import load, diffusion
    from src.eval.sampler import sample_latents
    from tools.stroke_interaction.f4_gt_vs_gen import std_path

    rows = [r for r in csv.DictReader(open(ROOT / a.csv, encoding='utf-8'))
            if (ROOT / r['image_path']).exists()]
    rng = np.random.default_rng(a.seed)

    surf = validate_surrogates(rows, rng)
    print(f'[f5] surrogate validation: {surf}', flush=True)
    if not surf.get('ok'):
        print('[f5] ABORT: smooth surrogates do not track the measured physics; '
              'any gradient conclusion would be meaningless', flush=True)
        return 1

    model, config, ck = load(a.ckpt, 'cuda', use_ema=not a.raw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    process = diffusion(config)
    vae = AutoencoderKL.from_pretrained(config.vae_path, torch_dtype=torch.bfloat16,
                                        local_files_only=True).cuda().eval()
    ck_dir = (ROOT / a.ckpt).parents[1]
    cfg = json.loads((ck_dir / 'resolved_config.json').read_text(encoding='utf-8'))
    callig_remap = json.loads((ROOT / cfg['callig_remap_json']).read_text(encoding='utf-8'))
    font_remap = json.loads((ROOT / cfg['font_remap_json']).read_text(encoding='utf-8'))

    by_char = defaultdict(set)
    for r in rows:
        by_char[(r['script'], r['character'])].add(r['calligrapher'])
    cand = [(k, w) for k, w in by_char.items() if len(w) >= 4 and std_path(*k)]
    rng.shuffle(cand)
    pairs = []
    for (sc, ch), writes in cand[:a.n_sample]:
        w = sorted(writes)[0]
        rs = [r for r in rows if r['script'] == sc and r['character'] == ch
              and r['calligrapher'] == w][:3]
        if not rs:
            continue
        acc = defaultdict(list)
        for r in rs:
            arr = np.array(Image.open(ROOT / r['image_path']).convert('L'),
                           np.float32) / 255.0
            ink = 1.0 - arr if arr.mean() > 0.5 else arr
            t = torch.from_numpy(ink.astype(np.float32)).cuda()
            with torch.no_grad():
                ss = soft_stats(t)
            for k_, val in ss.items():
                acc[k_].append(float(val))
        if acc:
            pairs.append(((sc, ch, w, rs[0]), {k_: float(np.mean(v_)) for k_, v_ in acc.items()}))
    print(f'[f5] {len(pairs)} (char, writer) pairs, targets from REAL images', flush=True)

    results = {}
    for cand_name in ('C0', 'C1', 'C2'):
        dA, dB, dC, shift, n_used = [], [], [], [], 0
        content = []          # content-preservation judge, never seen by the loss
        per_feat = {}
        closer = []           # relative reduction in distance-to-target
        for (sc, ch, w, row), tgt in pairs:
            wid = int(callig_remap[str(row['calligrapher_id'])])
            sid = int(font_remap[str(row['script_id'])])
            g = np.array(Image.open(std_path(sc, ch)).convert('L'), np.float32) / 255.0
            g64 = torch.from_numpy(g)[None, None]
            with torch.no_grad():
                noise = torch.randn(1, 4, 32, 32, generator=torch.Generator()
                                    .manual_seed(a.seed))
                z0 = sample_latents(model, process, noise, [(wid, 0)], 1.0, 1, 'cuda',
                                    skel=g64, seed=a.seed,
                                    hier_conds=[[wid], [0], [sid]])
            z = z0.clone().cuda().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=a.lr)

            def img_of(zz):
                x = vae.decode((zz.float() / 0.18215).to(torch.bfloat16)).sample
                return ((x.float().clamp(-1, 1) + 1) / 2)[0, 0]

            def loss_of(img):
                if cand_name == 'C1':
                    return sum((soft_stats(blur(img[None, None], 5, 1.2)[0, 0])[k_] -
                                tgt[k_]) ** 2 for k_ in FAM_A)
                if cand_name == 'C2':
                    tot = 0.0
                    for sig in (0.8, 1.2, 1.8):
                        st = soft_stats(blur(img[None, None], 5, sig)[0, 0])
                        tot = tot + sum((st[k_] - tgt[k_]) ** 2 for k_ in FAM_A)
                    return tot / 3
                st = soft_stats(img)
                return sum((st[k_] - tgt[k_]) ** 2 for k_ in FAM_A)

            with torch.no_grad():
                ref = img_of(z0.cuda()).detach().cpu().numpy()
                h0 = hard_stats(ref, rng)
                st0 = soft_stats(torch.from_numpy(ref).cuda())
            for _ in range(a.steps):
                loss = loss_of(img_of(z))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                cur = img_of(z).detach().cpu().numpy()
                h1 = hard_stats(cur, rng)
                st1 = soft_stats(torch.from_numpy(cur).cuda())
            if h0 is None or h1 is None:
                continue
            # THE point of the pilot: did we actually get closer to the real
            # target, in units of the target? A loss value alone cannot answer
            # this, so we measure distance-to-target before vs after.
            d0 = float(sum((st0[k_] - tgt[k_]) ** 2 for k_ in FAM_A).item())
            d1 = float(sum((st1[k_] - tgt[k_]) ** 2 for k_ in FAM_A).item())
            closer.append((d0 - d1) / max(d0, 1e-9))
            # content judge: the image must still look like the same glyph.
            # Correlation with the starting image plus total ink mass.
            if ref.shape == cur.shape:
                cc = float(np.corrcoef(ref.ravel(), cur.ravel())[0, 1])
            else:
                cc = 0.0
            content.append((cc, float(cur.mean()) / (float(ref.mean()) + 1e-6)))
            n_used += 1
            # Judge changes in units of the feature's natural spread on real
            # images, not in raw units. end_area has mean 27 / std 15 while
            # n_branches has mean 3.1 / std 0.11, so a raw-unit average says
            # nothing about which feature actually degraded.
            for k_ in FAM_B:
                dB.append((h1[k_] - h0[k_]) / (NATURAL_SD.get(k_, 1.0) or 1.0))
            for k_ in FAM_C:
                dC.append((h1[k_] - h0[k_]) / (NATURAL_SD.get(k_, 1.0) or 1.0))
            per_feat.setdefault('B', []).append({k_: (h1[k_] - h0[k_]) / NATURAL_SD.get(k_, 1)
                                                 for k_ in FAM_B})
            per_feat.setdefault('C', []).append({k_: (h1[k_] - h0[k_]) / NATURAL_SD.get(k_, 1)
                                                 for k_ in FAM_C})
            dA.append(np.mean([(soft_stats(img_of(z))[k_] - tgt[k_]).item() ** 2
                               for k_ in FAM_A]))
            shift.append(float((z.detach() - z0.cuda()).norm()))
        results[cand_name] = {
            'n': n_used,
            'distance_to_target_reduction': round(float(np.mean(closer)), 4)
            if closer else None,
            'loss_final_mse': round(float(np.mean(dA)), 5) if dA else None,
            'independent_B_change': round(float(np.mean(dB)), 5) if dB else None,
            'independent_C_change': round(float(np.mean(dC)), 5) if dC else None,
            'latent_shift': round(float(np.mean(shift)), 4) if shift else None,
            'content_corr_with_start': round(float(np.mean([c[0] for c in content])), 4)
            if content else None,
            'per_feature_sd_change': {
                fam: {k_: round(float(np.mean([r[k_] for r in v])), 2) for k_ in v[0]}
                for fam, v in per_feat.items() if v},
            'ink_mass_ratio': round(float(np.mean([c[1] for c in content])), 4)
            if content else None,
        }
        print(f'[f5] {cand_name}: {results[cand_name]}', flush=True)

    # verdict, in the plan's own terms. A candidate ships only if ALL hold:
    #   * distance to the REAL target actually shrinks, by a non-trivial margin
    #   * an INDEPENDENT family moves, in units of its natural spread
    #   * the glyph survives (correlation with the start, ink mass sane)
    #   * NO judged feature is pushed more than 0.5 sd -- a 1+ sd shift on
    #     orient_entropy means strokes are being smoothed together, which is
    #     content erosion wearing the costume of a style gain.
    # A candidate that only raises its own score is rejected, as the plan says.
    for name, r in results.items():
        red = r['distance_to_target_reduction']
        closes = red is not None and red > 0.02
        indep = abs(r['independent_B_change'] or 0) > 0.05
        intact = (r['content_corr_with_start'] or 0) > 0.6 and \
                 0.7 < (r['ink_mass_ratio'] or 0) < 1.4
        worst, worst_name = 0.0, ''
        for fam in ('B', 'C'):
            for k_, val in (r.get('per_feature_sd_change', {}).get(fam) or {}).items():
                if abs(val) > worst:
                    worst, worst_name = abs(val), f'{fam}:{k_}'
        no_erosion = worst < 0.5
        r['pass'] = bool(closes and indep and intact and no_erosion)
        r['gate'] = {'closer_to_target': closes, 'independent_agrees': indep,
                     'content_intact': intact,
                     'worst_judged_feature': f'{worst_name}={worst:.2f}sd',
                     'no_structure_erosion': no_erosion}
    for cand_key, cand_res in results.items():
        print(f'[f5] {cand_key}: pass={cand_res["pass"]} gate={cand_res["gate"]}',
              flush=True)

    (ROOT / a.out).mkdir(parents=True, exist_ok=True)
    p = ROOT / a.out / 'gradient_pilot.json'
    p.write_text(json.dumps({'surrogates': surf, 'candidates': results},
                            ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[f5] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

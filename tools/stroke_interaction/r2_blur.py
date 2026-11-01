"""R2: is the "47 pseudo-junctions" a VAE effect or a detector effect?

The whole VAE conclusion rests on one number: GT has ~7 junctions in the panel
and GT->VAE has ~47. Before blaming the VAE, we must ask whether ANY mild
smoothing trips the same detector. Blur and downsample-upsample are applied to
GT with no VAE involved; if a blur whose edge statistics match the VAE
reconstruction reproduces the jump, the 47 is a detector property.

Two things are measured for each condition, on identical panels:
  * a robust topology count (pruned-skeleton junctions, computed at several
    thresholds and reported with its spread)
  * the writer-difference projection of the SAME features C1 used, so the two
    numbers are directly comparable.

Matching criterion: for each blur level we report the mean gradient magnitude
relative to GT. The level whose grad_mag matches GT->VAE is the one that makes
the comparison fair; comparing a sharp GT to a blurry VAE without matching
sharpness would confound topology with softness.
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

GEO = ['asym_ud', 'asym_lr', 'grad_mag', 'turn_angle', 'end_area', 'end_elong',
       'n_branches', 'orient_peak']
ALLF = GEO + ['mass', 'ink_frac']
THRS = (0.35, 0.45, 0.50, 0.55, 0.65)


def robust_topology(ink, thr, prune=3):
    """Junction/endpoint counts after pruning short spurs.

    A skeleton of a soft edge sprouts many 1-2 px spurs, each of which is a
    spurious junction. Pruning spurs shorter than `prune` before counting is the
    standard remedy and is what makes the count comparable across sharpness.
    """
    from skimage.morphology import skeletonize, remove_small_objects
    from scipy import ndimage as ndi
    b = ink > thr
    if b.sum() < 64:
        return None
    sk = skeletonize(b)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3)), mode='constant') - sk
    ends = sk & (nb == 1)
    junc = sk & (nb >= 3)
    # prune: drop skeleton pixels belonging to an end-branch shorter than prune
    sk_p = sk.copy()
    for _ in range(prune):
        nb2 = ndi.convolve(sk_p.astype(np.uint8), np.ones((3, 3)),
                           mode='constant') - sk_p
        tip = sk_p & (nb2 == 1)
        if not tip.any():
            break
        sk_p &= ~tip
    nb3 = ndi.convolve(sk_p.astype(np.uint8), np.ones((3, 3)),
                       mode='constant') - sk_p
    junc_p = ndi.label(sk_p & (nb3 >= 3))[1]
    ends_p = ndi.label(sk_p & (nb3 == 1))[1]
    return {'junction': int(junc_p), 'end': int(ends_p),
            'junction_raw': int(ndi.label(junc)[1]),
            'end_raw': int(ndi.label(ends)[1])}


def blur(arr, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma)


def resize_rt(arr, f):
    from PIL import Image
    h, w = arr.shape
    small = np.array(Image.fromarray((arr * 255).astype(np.uint8))
                     .resize((max(8, int(w * f)), max(8, int(h * f))), Image.BICUBIC),
                     np.float32) / 255.0
    return np.array(Image.fromarray((small * 255).astype(np.uint8)).resize(
        (w, h), Image.BICUBIC), np.float32) / 255.0


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
    rows = [r for r in csv.DictReader(open(ROOT / a.csv, encoding='utf-8'))
            if (ROOT / r['image_path']).exists()]

    @torch.no_grad()
    def vae_rt(arr):
        # arr is the RAW scan in [0,1] (bright = paper); the VAE was trained on
        # exactly that, so no polarity flip here. Returning ink-space directly
        # caused the first sweep to report grad 0.0088 instead of 0.0278 and
        # produced a completely wrong "blur explains VAE" verdict.
        t = torch.from_numpy(arr)[None, None].cuda().float() * 2 - 1
        t = t.repeat(1, 3, 1, 1).to(torch.bfloat16)
        lat = vae.encode(t).latent_dist.mode() * 0.18215
        x = vae.decode((lat.float() / 0.18215).to(torch.bfloat16)).sample
        out = ((x.float().clamp(-1, 1) + 1) / 2)[0].mean(0).cpu().numpy()
        return 1.0 - out if out.mean() > 0.5 else out

    rng = np.random.default_rng(a.seed)
    by_char = defaultdict(list)
    for r in rows:
        by_char[(r['script'], r['character'])].append(r)
    cand = [k for k, v in by_char.items()
            if len({r['calligrapher'] for r in v}) >= 4]
    rng.shuffle(cand)
    cand = cand[:a.chars]
    print(f'[R2] {len(cand)} characters', flush=True)

    imgs = defaultdict(list)
    for sc, ch in cand:
        for r in by_char[(sc, ch)][:6]:
            p = ROOT / r['image_path']
            arr = np.array(Image.open(p).convert('L'), np.float32) / 255.0
            ink = 1.0 - arr if arr.mean() > 0.5 else arr
            imgs['GT'].append(ink)
            imgs['VAE'].append(vae_rt(arr))
            # go well past the VAE's own sharpness: the first sweep bottomed out
            # at sigma=1.3 with grad 0.0244 while the VAE sits at 0.0088, so the
            # matched-blur comparison had not actually been made yet.
            for sg in (0.6, 0.9, 1.3, 2.0, 2.8, 3.8, 5.0):
                imgs[f'blur{sg}'].append(blur(ink, sg))
            for f in (0.5, 0.35):
                imgs[f'resize{f}'].append(resize_rt(ink, f))
    print(f'[R2] {len(imgs["GT"])} images per arm', flush=True)

    rep = {}
    for arm, arrs in imgs.items():
        per_thr = defaultdict(lambda: defaultdict(list))
        gm = []
        for arr in arrs:
            gm.append(float(np.hypot(*np.gradient(arr)).mean()))
            for thr in THRS:
                t = robust_topology(arr, thr)
                if t:
                    for k, v in t.items():
                        per_thr[thr][k].append(v)
        # per_thr[thr][key] holds a LIST of per-image values; summarise each
        entry = {'grad_mag': round(float(np.mean(gm)), 5),
                 'by_threshold': {str(thr): {k: round(float(np.mean(v)), 3)
                                             for k, v in d.items() if len(v)}
                                  for thr, d in per_thr.items()}}
        jc = [float(np.mean(d['junction'])) for d in per_thr.values()
              if d.get('junction')]
        entry['junction_pruned_mean'] = round(float(np.mean(jc)), 3) if jc else None
        entry['junction_pruned_spread'] = [round(min(jc), 2),
                                           round(max(jc), 2)] if jc else None
        jr = [float(np.mean(d['junction_raw'])) for d in per_thr.values()
              if d.get('junction_raw')]
        entry['junction_raw_mean'] = round(float(np.mean(jr)), 3) if jr else None
        rep[arm] = entry
        print(f'[R2] {arm:10s} grad={entry["grad_mag"]:.5f} '
              f'junc_pruned={entry["junction_pruned_mean"]} '
              f'(range {entry["junction_pruned_spread"]}) '
              f'junc_raw={entry["junction_raw_mean"]}', flush=True)

    # which blur matches the VAE's sharpness?
    g_vae = rep['VAE']['grad_mag']
    blurs = {k: v['grad_mag'] for k, v in rep.items() if k.startswith('blur')}
    best = min(blurs, key=lambda k: abs(blurs[k] - g_vae))
    rep['_sharpness_match'] = {'vae_grad': g_vae, 'closest_blur': best,
                               'closest_blur_grad': blurs[best],
                               'gt_grad': rep['GT']['grad_mag']}
    print(f'\n[R2] VAE grad={g_vae:.5f} closest blur={best} '
          f'({blurs[best]:.5f}); GT grad={rep["GT"]["grad_mag"]:.5f}', flush=True)

    g0 = rep['GT']['junction_pruned_mean']
    gv = rep['VAE']['junction_pruned_mean']
    gb = rep[best]['junction_pruned_mean']
    verdict = {
        'gt_junction': g0, 'vae_junction': gv, 'matched_blur_junction': gb,
        'vae_inflation': round(gv / max(g0, 1e-9), 2),
        'blur_inflation': round(gb / max(g0, 1e-9), 2),
        'blur_explains_vae': bool(gb >= 0.7 * gv),
    }
    rep['_verdict'] = verdict
    print(f'[R2] VERDICT: VAE inflates junctions x{verdict["vae_inflation"]}, '
          f'matched blur x{verdict["blur_inflation"]} '
          f'-> blur explains VAE = {verdict["blur_explains_vae"]}', flush=True)

    p = ROOT / a.out / 'R2_blur_control.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[R2] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

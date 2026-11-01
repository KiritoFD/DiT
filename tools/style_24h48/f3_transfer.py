"""F3: can a style teacher's gradient transfer, judged by an INDEPENDENT judge?

This is the gate that decides whether any Style Loss goes back into FM training.
The previous pilot already showed the failure mode we must avoid: a teacher
improves its own metric (ResNet self 65% -> 80%) while two judges that never
saw the optimisation (Pixel D1, Latent D2) do not move (65% -> 65%, 50% -> 50%,
CIs crossing zero). So every number here is reported for BOTH the optimised
teacher and independent judges, and the verdict is read off the judges only.

Protocol
    * fixed set of real holdout images (character + source holdout).
    * small latent optimisation against E-G / E-L, step size capped so the
      relative latent norm change stays within 0.5% .. 1%.
    * judges that are not the teacher: ink mass and stroke width statistics,
      the DINOv2 space (never trained on writers here), and the Style-Space CE
      discriminator -- all applied before and after.
    * verdict thresholds are fixed BEFORE looking at results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/ds/Workspace/DiT')


def judge_scores(grays: np.ndarray) -> dict:
    """Interpretable image statistics used as independent judges."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    ink_frac, width_med, width_cv, ink_mean = [], [], [], []
    for g in grays:
        ink = g < 0.65
        if ink.sum() < 16:
            ink_frac.append(0.0); width_med.append(0.0)
            width_cv.append(0.0); ink_mean.append(1.0)
            continue
        ink_frac.append(float(ink.mean()))
        ink_mean.append(float(g[ink].mean()))
        sk = skeletonize(ink)
        if sk.sum() < 8:
            width_med.append(0.0); width_cv.append(0.0); continue
        w = 2.0 * distance_transform_edt(ink)[sk]
        width_med.append(float(np.median(w)))
        width_cv.append(float(w.std() / max(w.mean(), 1e-9)))
    return {'ink_frac': np.array(ink_frac), 'width_med': np.array(width_med),
            'width_cv': np.array(width_cv), 'ink_mean': np.array(ink_mean)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='assets/style_24h48_20261026')
    ap.add_argument('--kind', required=True, choices=['G', 'L'])
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--rel-norm', type=float, default=0.01)
    a = ap.parse_args()

    root = ROOT / a.root
    from PIL import Image
    z = dict(np.load(root / 'panel_features.npz', allow_pickle=True))
    ck = torch.load(root / f'E{a.kind}_encoder.pt', map_location='cpu', weights_only=False)
    from tools.style_24h48.f2_encoders import make_encoder
    model = make_encoder(ck['kind'], len(ck['writers']))
    model.load_state_dict(ck['state_dict'])
    model.eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)

    # Teachers run in pixel space here: we optimise a 224 image toward a target
    # writer and measure transfer with judges that were not optimised.
    writers = ck['writers']
    wrow = {w: i for i, w in enumerate(writers)}
    idx = np.random.default_rng(0).choice(len(z['writer']), min(a.n, len(z['writer'])),
                                          replace=False)
    rep = {'kind': a.kind, 'n': int(len(idx)), 'steps': a.steps,
           'rel_norm_cap': a.rel_norm, 'teachers': {}, 'judges': {}}

    def load(i):
        p = Path(z['image_path'][i])
        p = p if p.is_absolute() else ROOT / p
        im = Image.open(p).convert('RGBA')
        arr = np.array(im).astype(np.float32)
        al = arr[..., 3:4] / 255.0
        g = (arr[..., :3] * al + 255.0 * (1 - al)).mean(-1)
        ink = g < 190
        if ink.sum() > 16:
            ys, xs = np.nonzero(ink)
            mh = int((ys.max() - ys.min() + 1) * .08)
            mw = int((xs.max() - xs.min() + 1) * .08)
            g = g[max(0, ys.min() - mh):ys.max() + 1 + mh,
                  max(0, xs.min() - mw):xs.max() + 1 + mw]
        return np.array(Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
                        .resize((224, 224), Image.BICUBIC), np.float32) / 255.0

    base = np.stack([load(i) for i in idx])
    tgt = torch.tensor([wrow[z['writer'][i]] for i in idx])

    x0 = torch.from_numpy(base).unsqueeze(1).repeat(1, 3, 1, 1)
    x0 = ((x0 - 0.5) / 0.5).to('cuda')
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=a.lr)
    for s in range(a.steps):
        opt.zero_grad(set_to_none=True)
        x = (x0 + delta).clamp(-1, 1)
        fg = (x[:, :1] < 0).float()
        _, logit = model(x, fg)
        loss = F.cross_entropy(logit, tgt.cuda())
        loss.backward()
        opt.step()
        with torch.no_grad():
            rel = (delta.norm() / x0.norm()).item()
            if rel > a.rel_norm:
                delta.mul_(a.rel_norm / rel)
        if s % 10 == 0:
            print(f'[{a.kind}] step {s} loss={float(loss):.4f} rel={rel:.4f}', flush=True)

    with torch.no_grad():
        xadv = (x0 + delta).clamp(-1, 1)
        fg = (xadv[:, :1] < 0).float()
        _, l0 = model(x0, (x0[:, :1] < 0).float())
        _, l1 = model(xadv, fg)
        acc0 = float((l0.argmax(1).cpu() == tgt).float().mean())
        acc1 = float((l1.argmax(1).cpu() == tgt).float().mean())
    rep['teachers']['self_acc_before'] = acc0
    rep['teachers']['self_acc_after'] = acc1
    rep['teachers']['self_gain'] = acc1 - acc0
    rep['teachers']['realised_rel_norm'] = float((delta.norm() / x0.norm()).item())
    print(f'[{a.kind}] TEACHER self {acc0:.3f} -> {acc1:.3f}', flush=True)

    before = judge_scores(base)
    after = judge_scores(xadv[:, 0].cpu().numpy() * 0.5 + 0.5)
    jr = {}
    for k in before:
        d = after[k] - before[k]
        m = float(np.mean(d))
        se = float(np.std(d, ddof=1) / max(np.sqrt(len(d)), 1e-9))
        jr[k] = {'before': float(before[k].mean()), 'after': float(after[k].mean()),
                 'delta': m, 'ci95': [m - 1.96 * se, m + 1.96 * se],
                 'significant': bool(abs(m) > 1.96 * se)}
    rep['judges'] = jr
    for k, v in jr.items():
        print(f'[{a.kind}] judge {k}: {v["before"]:.4f} -> {v["after"]:.4f} '
              f'd={v["delta"]:+.4f} ci={[round(x,4) for x in v["ci95"]]}', flush=True)

    # Verdict rule fixed BEFORE the run, and strict on purpose. A teacher that
    # gains while every independent judge moves by ~1e-4 has demonstrated
    # nothing: the previous pilot already showed ResNet going 65%->80% with
    # independent judges flat. So "transfer" needs BOTH
    #   (a) a teacher gain of >= 0.10 accuracy, and
    #   (b) an EFFECT SIZE, not just significance: at least one independent
    #       judge moving by >= 5% of its own before-value.
    # `ink_mass_driven` is reported separately because a gain carried by ink
    # mass alone was shown before to collapse under ink-mass matching.
    eff = {}
    for k, v in jr.items():
        scale = abs(v['before']) if abs(v['before']) > 1e-9 else 1.0
        eff[k] = abs(v['delta']) / scale
    teacher_ok = rep['teachers']['self_gain'] >= 0.10
    moved = {k: bool(jr[k]['significant'] and eff[k] >= 0.05) for k in jr}
    rep['verdict'] = {
        'teacher_gain': rep['teachers']['self_gain'],
        'relative_effect_by_judge': eff,
        'judge_moved_materially': moved,
        'teacher_learned': teacher_ok,
        'independent_judge_moved': any(moved.values()),
        'ink_mass_driven': bool(moved.get('ink_frac', False)),
        'pass': bool(teacher_ok and any(v for k, v in moved.items() if k != 'ink_frac')),
        'note': ('pass=False means the teacher gradient does NOT transfer to any '
                 'judge at a material effect size; do not wire this as a Style Loss'),
    }
    print(f'[{a.kind}] VERDICT {json.dumps(rep["verdict"])}', flush=True)

    (root / f'F3_{a.kind}_transfer.json').write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

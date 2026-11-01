"""C1-C3: attribute the negative projection on n_branches / end_area.

C1  VAE DECOMPOSITION. Same 30 characters, three arms:
        GT  ->  VAE(encode/decode)  ->  Deep114
    If GT->VAE already loses the topology features, the miss is the VAE's, not
    the generator's. This is the control that could overturn the C conclusion,
    because a VAE trained for reconstruction need not preserve skeleton
    branch counts even when its style classification features survive.

C2  DETECTOR STABILITY. Recompute the sign of the projection under several ink
    thresholds and 128/256 resolutions. A negative projection that flips sign
    with the threshold is a detector artefact.

C3  FIXED-REGION PANEL. The earlier comparison let each image find its OWN
    events, then compared. That silently drops the most important style
    difference: one writer joins two strokes and another does not. So here the
    region comes from STD alone (shared by all arms), event COUNTS are compared
    inside each region, and shape statistics use only that region's pixels.

Statistics are clustered: the resampling unit is the CHARACTER, not the
324 (character, writer-pair) rows, because those rows share 30 characters and
are not independent experiments.
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


def ink_of(arr):
    a = arr.astype(np.float32)
    return 1.0 - a if a.mean() > 0.5 else a


def panel_features(ink, b, regions, rng):
    """Per-region event counts + shape stats, on ONE shared region set.

    regions: list of (y0, y1, x0, x1) in normalised coords, from STD only.
    For each region we record the number of end/junction events whose centre
    falls inside, plus the mean of every feature. Counting per region is what
    C3 asks for: a writer who joins two strokes where another does not changes
    the COUNT, and that variation must not be filtered out.
    """
    from tools.stroke_interaction.features import extract_events, skeleton_graph
    ev, _, _ = extract_events(ink, b, rng, per_kind=400)
    if not ev:
        return None
    H, W = ink.shape
    pos = np.array([[e['y'], e['x']] for e in ev], np.float32)
    kinds = np.array([e['kind'] for e in ev])
    feats = np.array([[e.get(k, 0.0) for k in ALLF] for e in ev], np.float32)
    out = {}
    for ri, (y0, y1, x0, x1) in enumerate(regions):
        m = (pos[:, 0] >= y0) & (pos[:, 0] < y1) & \
            (pos[:, 1] >= x0) & (pos[:, 1] < x1)
        for kind in ('end', 'junction', 'turn'):
            out[f'r{ri}_{kind}_count'] = float(np.sum(m & (kinds == kind)))
            # shape stats of THAT kind's events inside the region, so each
            # kind carries its own vector instead of sharing one pooled set
            mk = m & (kinds == kind)
            if mk.any():
                for fi, f in enumerate(ALLF):
                    out[f'r{ri}_{kind}_{f}'] = float(feats[mk, fi].mean())
            else:
                for f in ALLF:
                    out[f'r{ri}_{kind}_{f}'] = 0.0
        # local ink inside the region, independent of event detection
        yy0, yy1 = int(y0 * H), int(y1 * H)
        xx0, xx1 = int(x0 * W), int(x1 * W)
        sub = ink[max(0, yy0):min(H, yy1), max(0, xx0):min(W, xx1)]
        out[f'r{ri}_region_ink'] = float(sub.mean()) if sub.size else 0.0
    return out


def build_regions(ink_std, gi, gx, n=8):
    """Regions defined by the STD alone, so every arm sees the same boxes."""
    H, W = ink_std.shape
    mass = ink_std
    ys, xs = np.mgrid[0:H, 0:W]
    # 8 regions = a 3x3-ish split biased to where the glyph actually is
    regs = []
    stats = []
    for i in range(3):
        for j in range(3):
            y0, y1 = i / 3, (i + 1) / 3
            x0, x1 = j / 3, (j + 1) / 3
            m = mass[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
            stats.append((float(m.sum()), (y0, y1, x0, x1)))
    stats.sort(key=lambda t: -t[0])
    for _, r in stats[:n]:
        regs.append(r)
    return regs


def cluster_boot(vals, groups, n=1000, seed=0):
    """Bootstrap over GROUPS (characters), not rows."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, np.float64)
    uniq = np.unique(groups)
    idx = {g: np.nonzero(np.asarray(groups) == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        out.append(float(np.mean(vals[sel])))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default='exp-std/csv/train_top10_aug_sym.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--chars', type=int, default=30)
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
    callig_remap = json.loads((ROOT / cfg['callig_remap_json']).read_text(encoding='utf-8'))
    font_remap = json.loads((ROOT / cfg['font_remap_json']).read_text(encoding='utf-8'))

    model, config, _ = load(a.ckpt, 'cuda', use_ema=not a.raw)
    process = diffusion(config)
    vae = AutoencoderKL.from_pretrained(config.vae_path, torch_dtype=torch.bfloat16,
                                        local_files_only=True).cuda().eval()
    rng = np.random.default_rng(a.seed)

    by_char = defaultdict(list)
    for r in rows:
        by_char[(r['script'], r['character'])].append(r)
    cand = [(k, v) for k, v in by_char.items() if std_png(*k) and
            len({r['calligrapher'] for r in v}) >= 4]
    rng.shuffle(cand)
    cand = cand[:a.chars]
    print(f'[C1] {len(cand)} characters', flush=True)

    @torch.no_grad()
    def vae_rt(arr):
        # the SD VAE is a 3-channel model; a grey calligraphy scan is replicated
        # across channels, which is exactly how the training data was fed.
        t = torch.from_numpy(arr)[None, None].cuda().float() * 2 - 1
        t = t.repeat(1, 3, 1, 1)
        lat = vae.encode(t.to(torch.bfloat16)).latent_dist.mode() * 0.18215
        x = vae.decode(lat.to(torch.bfloat16)).sample
        return ((x.float().clamp(-1, 1) + 1) / 2)[0].mean(0).cpu().numpy()

    @torch.no_grad()
    def gen_img(wid, sid, g64, seed):
        noise = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(seed))
        lat = sample_latents(model, process, noise, [(wid, 0)], 1.0, 1, 'cuda',
                             skel=g64, seed=seed, hier_conds=[[wid], [0], [sid]])
        x = vae.decode((lat.cuda().float() / 0.18215).to(torch.bfloat16)).sample
        return ((x.float().clamp(-1, 1) + 1) / 2)[0].mean(0).cpu().numpy()

    # accumulate per-arm per-character per-writer-pair projections
    arms = ('GT', 'VAE', 'GEN')
    per_arm = {arm: defaultdict(list) for arm in arms}   # arm -> kind -> proj
    per_arm_feat = {arm: defaultdict(list) for arm in arms}
    grp = {arm: defaultdict(list) for arm in arms}
    per_char_means_kind = {arm: {} for arm in arms}
    counts = defaultdict(lambda: defaultdict(dict))     # kind -> arm -> char -> count

    for ci, ((sc, ch), rs) in enumerate(cand):
        sp = std_png(sc, ch)
        g = np.array(Image.open(sp).convert('L'), np.float32) / 255.0
        g64 = torch.from_numpy(g)[None, None]
        regions = build_regions(ink_of(g), 0, 0)
        writers = sorted({r['calligrapher'] for r in rs})
        # arm features per writer: average over up to 3 GT images each
        feats = {arm: {} for arm in arms}
        for w in writers:
            wr = [r for r in rs if r['calligrapher'] == w][:3]
            if not wr:
                continue
            per_arm_imgs = {arm: [] for arm in arms}
            for r in wr:
                arr = np.array(Image.open(ROOT / r['image_path']).convert('L'),
                               np.float32) / 255.0
                for arm, im in (('GT', arr), ('VAE', vae_rt(arr))):
                    ink = ink_of(im)
                    f = panel_features(ink, (ink > 0.5).astype(np.uint8),
                                       regions, np.random.default_rng(1))
                    if f:
                        per_arm_imgs[arm].append(f)
            row = next((r for r in wr), None)
            if row is not None:
                wid = int(callig_remap.get(str(row['calligrapher_id']), -1))
                sid = int(font_remap.get(str(row['script_id']), 0))
                if wid >= 0:
                    im = gen_img(wid, sid, g64, a.seed + ci)
                    ink = ink_of(im)
                    f = panel_features(ink, (ink > 0.5).astype(np.uint8),
                                       regions, np.random.default_rng(1))
                    if f:
                        per_arm_imgs['GEN'].append(f)
            for arm in arms:
                if per_arm_imgs[arm]:
                    keys = set.intersection(*[set(d) for d in per_arm_imgs[arm]])
                    feats[arm][w] = {k: float(np.mean([d[k] for d in per_arm_imgs[arm]]))
                                     for k in keys}
        # pair differences, kept per event kind. Shape features are stored per
        # (region, kind) by the panel, so each kind's vector is self-contained
        # and end/junction/turn give different numbers.
        ws = [w for w in writers if all(w in feats[arm] for arm in arms)]
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                for arm in arms:
                    f1, f2 = feats[arm][ws[i]], feats[arm][ws[j]]
                    keys = sorted(set(f1) & set(f2))
                    if not keys:
                        continue
                    d = np.array([f1[k] - f2[k] for k in keys], np.float32)
                    for kind in ('end', 'junction', 'turn'):
                        use = [t for t, kk in enumerate(keys)
                               if f'_{kind}_' in kk]
                        if not use:
                            continue
                        per_char_means_kind[arm][(ws[i], ws[j], ci, kind)] = (
                            [keys[t] for t in use], d[use])
        for kind in ('end', 'junction', 'turn'):
            cnt = {}
            for arm in arms:
                vals = [feats[arm][w].get(f'r{r}_{kind}_count', 0.0)
                        for r in range(8) for w in ws if w in feats[arm]]
                cnt[arm] = float(np.mean(vals)) if vals else None
            for arm in arms:
                if cnt[arm] is not None:
                    counts[kind][arm][ci] = cnt[arm]
        if (ci + 1) % 5 == 0:
            print(f'[C1] {ci+1}/{len(cand)} chars', flush=True)

    # projections: for each character, aggregate its pair rows first (cluster),
    # then bootstrap over characters
    rep = {'n_chars': len(cand), 'ckpt': a.ckpt, 'analyzed_vs': ['GT', 'VAE', 'GEN']}
    for kind in ('end', 'junction', 'turn'):
        entry = {}
        for arm in ('VAE', 'GEN'):
            projs, grps, kp, rest = [], [], [], []
            for ci in range(len(cand)):
                # keys are (w_i, w_j, char_index); unpack all three or the
                # comparison k[2] == ci silently reads the wrong element
                rows_ = [v for k, v in per_char_means_kind['GT'].items()
                         if k[2] == ci and k[3] == kind]
                if not rows_:
                    continue
                ar = [v for k, v in per_char_means_kind[arm].items()
                      if k[2] == ci and k[3] == kind]
                if not ar:
                    continue
                # keys come from set intersections, so their ORDER differs
                # between pairs; intersect across both arms and index by name
                common = set(rows_[0][0])
                for ks, _ in rows_:
                    common &= set(ks)
                for ks, _ in ar:
                    common &= set(ks)
                keys = sorted(common)
                if not keys:
                    continue

                def vec(dd):
                    pos = {k: i for i, k in enumerate(keys)}
                    return np.array([np.mean([d[pos[k]] for ks, d in dd if k in pos])
                                     for k in keys], np.float32)
                gt = vec(rows_)
                al = vec(ar)
                nn = float(gt @ gt)
                if nn < 1e-12:
                    continue
                p = float(al @ gt / nn)
                projs.append(p)
                grps.append(ci)
                # key format is r{region}_{kind}_{feature} or r{region}_{kind}_count
                in_k = [i for i, kk in enumerate(keys)
                        if any(kk.endswith(f'_{f}') for f in KEY_FEATS)]
                in_o = [i for i in range(len(keys)) if i not in in_k]
                if in_k:
                    vg = gt[in_k]
                    vv = float(vg @ vg)
                    if vv > 1e-12:
                        kp.append(float(al[in_k] @ vg / vv))
                if in_o:
                    vg = gt[in_o]
                    vv = float(vg @ vg)
                    if vv > 1e-12:
                        rest.append(float(al[in_o] @ vg / vv))
            if projs:
                lo, hi = cluster_boot(projs, grps)
                entry[arm] = {
                    'proj_mean': round(float(np.mean(projs)), 4),
                    'proj_ci95_clustered': [round(lo, 4), round(hi, 4)],
                    'n_chars': len(projs),
                    'key_feats_proj': round(float(np.mean(kp)), 4) if kp else None,
                    'rest_feats_proj': round(float(np.mean(rest)), 4) if rest else None,
                }
        rep[f'projection_{kind}'] = entry
        print(f'[C1] {kind:9s} {entry}', flush=True)

    rep['event_counts_by_arm'] = {
        kind: {arm: round(float(np.mean(list(d.values()))), 3) if d else None
               for arm, d in v.items()} for kind, v in counts.items()}
    print(f"[C1] event counts: {rep['event_counts_by_arm']}", flush=True)

    p = ROOT / a.out / 'C1_vae_decomposition.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[C1] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""F2 ablations the plan asks for, run on the ALREADY TRAINED encoders.

The plan requires four measurements for each encoder:
  1. writer macro accuracy under character holdout and source holdout
  2. same-character different-writer pair accuracy
  3. performance after removing the background, and after ink-mass matching
  4. stability on VAE round-tripped images (stand-in for generated images)

f2_encoders.py only ever implemented (1) and (2). This file adds (3) and (4)
plus the per-Writer x Script work count the plan explicitly demands, without
retraining anything: it loads EG_encoder.pt / EL_encoder.pt and re-evaluates.

Ink-mass matching is the important control. A writer classifier can score well
by reading how much ink a page carries rather than how the strokes are shaped,
and the same shortcut is exactly what poisoned the earlier critic. So we
re-normalise every image to a COMMON ink mass before encoding, which destroys
ink-quantity information and keeps stroke geometry.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tools.style_24h48.f2_encoders import make_encoder

ROOT = Path(__file__).resolve().parents[2]


def load_meta():
    d = np.load(ROOT / 'assets/style_24h48_20261026/panel_features.npz', allow_pickle=True)
    return d


def load_gray(paths, R, size=224):
    from PIL import Image
    out = []
    for p in paths:
        p = Path(p)
        p = p if p.is_absolute() else R / p
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
        out.append(np.array(im2, np.float32) / 255.0)
    return np.stack(out)[:, None]


def to_tensor(grays):
    x = torch.from_numpy(grays).float()
    return ((x - 0.5) / 0.5).repeat(1, 3, 1, 1)


def strip_background(grays):
    """Keep only ink pixels over white: removes paper tone, scan borders, stains.

    This is the 'remove background' control. Anything the encoder gains from
    paper colour, foxing, or vignetting disappears, while stroke geometry is
    preserved exactly.
    """
    out = grays.copy()
    for i in range(len(out)):
        g = out[i, 0]
        # paper level per image, then push everything lighter than a margin to white
        paper = np.percentile(g, 90)
        m = g > (paper - 0.08)
        out[i, 0][m] = 1.0
    return out


def match_ink_mass(grays, iters=32):
    """Equalise per-image mean ink mass WITHOUT destroying stroke shape.

    Ink mass is defined on the same [0,1] ink convention the encoder sees.
    For each image we find the gamma that maps its ink mass onto the dataset
    median, applied on the ink channel (1 - gray). Gamma only re-weights how
    dark each stroke pixel is; it cannot move a stroke or erase it. If a
    classifier dies under this transform, its writer cue was ink QUANTITY,
    not stroke geometry - which is exactly the shortcut we need to expose.
    """
    ink = np.clip(1.0 - grays[:, 0], 0, 1)
    target = float(np.median(ink.mean(axis=(1, 2))))
    out = grays.copy()
    for i in range(len(out)):
        x = ink[i]
        lo, hi = 0.05, 8.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if float((x ** mid).mean()) > target:
                hi = mid
            else:
                lo = mid
        g = 0.5 * (lo + hi)
        new_ink = np.clip(x ** g, 0, 1)
        # never let the transform blank a page: keep the strongest 1% of ink
        if new_ink.max() > 1e-6:
            new_ink *= min(1.0, 1.0 / (np.percentile(new_ink, 99.0) + 1e-9))
        out[i, 0] = np.clip(1.0 - new_ink, 0, 1)
    return out


def vae_roundtrip(grays, vae_dir, dev, bs=48):
    """Encode/decode through the project VAE: the plan's 'generated image' proxy.

    Generated calligraphy always passes through this VAE, so its reconstruction
    is the cheapest honest stand-in for 'does the cue survive generation?'.
    Batched: 3746 images at once would need ~90 GiB of activations.
    """
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=torch.float32).to(dev).eval()
    outs = []
    for s in range(0, len(grays), bs):
        chunk = grays[s:s + bs]
        x = torch.from_numpy(chunk).float().to(dev) * 2 - 1
        x = x.repeat(1, 3, 1, 1)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode() * 0.18215
            y = vae.decode(z / 0.18215).sample
        y = ((y.clamp(-1, 1) + 1) / 2)[:, :1].cpu().numpy()
        outs.append(y)
        del x, z, y
        torch.cuda.empty_cache()
    del vae
    torch.cuda.empty_cache()
    return np.concatenate(outs)


@torch.no_grad()
def encode_all(model, grays, dev, bs=64):
    embs, logits = [], []
    for s in range(0, len(grays), bs):
        X = to_tensor(grays[s:s + bs]).to(dev)
        feat, lg = model(X, fg=(X[:, :1] < 0.0).float())
        embs.append(feat.cpu().numpy())
        logits.append(lg.cpu().numpy())
    return np.concatenate(embs), np.concatenate(logits)


def macro_acc(pred, y, writers):
    per = [float((pred == y)[y == w].mean()) for w in writers if (y == w).sum() > 0]
    return float(np.mean(per)), len(per)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--vae-dir', default='pretrained_models/sd-vae-ft-ema')
    ap.add_argument('--char-test-frac', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=90)
    ap.add_argument('--out', default='assets/style_24h48_20261026/F2_ablations.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    R = ROOT

    z = load_meta()
    paths = z['image_path']
    writer = z['writer']
    char = z['character']
    split = z['split']
    script = z['script']

    writers = sorted(set(writer.tolist()))
    # Reproduce EXACTLY the split f2_encoders.py used, so these ablations are a
    # controlled re-measurement of the same model on the same images. Guessing a
    # different holdout (e.g. "character absent from any train row") shrinks the
    # eval set to 168 images over 10 writers and inflates accuracy to ~0.94.
    chars = np.array(sorted(set(char.tolist())))
    rng0 = np.random.default_rng(a.seed)
    perm = rng0.permutation(len(chars))
    n_te = int(round(len(chars) * a.char_test_frac))
    test_chars = set(chars[perm[:n_te]].tolist())

    te_char = np.array([c in test_chars for c in char])
    te_both = te_char & (split == 'test')

    # The plan asks for BOTH holdouts: character holdout, and source holdout.
    # Source holdout here is the stricter "held-out characters AND held-out
    # works" set -- a writer can only be scored on it if some of their works
    # landed in the test split, which is exactly the cross-source test.
    HOLDOUTS = {'char': te_char, 'char_src': te_both}

    grays_all = {}
    for hname, mask in HOLDOUTS.items():
        ids = np.nonzero(mask)[0]
        if len(ids) == 0:
            continue
        print(f'[abl] loading {hname}: {len(ids)} images', flush=True)
        grays_all[hname] = (ids, load_gray([paths[i] for i in ids], R))

    out = {'n_writers': len(writers),
           'protocol': {'character_holdout': True,
                        'source_holdout': 'held-out characters AND held-out works',
                        'holdouts': {k: int(len(np.nonzero(m)[0])) for k, m in HOLDOUTS.items()}},
           'encoders': {}}

    for hname, (ids, grays) in grays_all.items():
        print(f'\n[abl] ===== holdout {hname}: {len(ids)} images =====', flush=True)
        y = np.array([writers.index(w) for w in writer[ids]])
        variants = {'original': grays}
        variants['no_background'] = strip_background(grays)
        variants['ink_matched'] = match_ink_mass(grays)
        try:
            variants['vae_roundtrip'] = vae_roundtrip(grays, a.vae_dir, dev)
            print('[abl] VAE round-trip ok', flush=True)
        except Exception as e:                  # noqa: BLE001 - report, never silently skip
            print(f'[abl] VAE round-trip SKIPPED: {type(e).__name__}: {e}', flush=True)

        for kind in ('G', 'L'):
            ck = torch.load(R / f'assets/style_24h48_20261026/E{kind}_encoder.pt',
                            map_location='cpu', weights_only=False)
            model = make_encoder(ck['kind'], len(ck['writers']))
            model.load_state_dict(ck['state_dict'])
            model.eval().to(dev)
            res = {}
            for name, g in variants.items():
                emb, lg = encode_all(model, g, dev)
                pred = lg.argmax(1)
                mac, nw = macro_acc(pred, y, list(range(len(writers))))
                # pair accuracy: same character, different writer.
                # A pair is correct when each image is closer to its OWN writer's
                # centroid than to the other writer's centroid. This is the plan's
                # "same-character different-writer identification" test, done on
                # cross-character evidence rather than on the classifier head.
                rng = np.random.default_rng(7)
                by_char = defaultdict(list)
                for j, c in enumerate(char[ids]):
                    by_char[c].append(j)
                embn = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
                centroid = {}
                for w in np.unique(y):
                    m = y == w
                    centroid[w] = embn[m].mean(0)
                correct = total = 0
                for c, js in by_char.items():
                    if len(js) < 2:
                        continue
                    for _ in range(min(len(js) * 2, 40)):
                        i1, i2 = rng.choice(js, 2, replace=False)
                        if y[i1] == y[i2]:
                            continue
                        own1 = embn[i1] @ centroid[y[i1]]
                        oth1 = embn[i1] @ centroid[y[i2]]
                        own2 = embn[i2] @ centroid[y[i2]]
                        oth2 = embn[i2] @ centroid[y[i1]]
                        correct += int(own1 > oth1) + int(own2 > oth2)
                        total += 2
                res[name] = {'macro_acc': round(mac, 4), 'n_writers_scored': nw,
                             'pair_acc': round(correct / max(total, 1), 4),
                             'n_pairs': total}
                print(f'[abl] {hname:8s} E-{kind} {name:14s} macro={mac:.4f} '
                      f'pair={correct/max(total,1):.4f}', flush=True)
            out['encoders'].setdefault(f'E-{kind}', {})[hname] = res

    # per Writer x Script work counts: the honest denominator (plan requirement)
    work = z['work']
    tally = defaultdict(lambda: {'images': 0, 'works': set()})
    for i in range(len(writer)):
        k = f'{writer[i]}|{script[i]}'
        tally[k]['images'] += 1
        tally[k]['works'].add(work[i])
    out['writer_script_works'] = {
        k: {'images': v['images'], 'works': len(v['works'])} for k, v in sorted(tally.items())
    }
    weak = {k: v for k, v in out['writer_script_works'].items() if v['works'] < 10}
    out['weak_strata_under_10_works'] = weak
    print(f'[abl] strata={len(tally)}  under 10 works: {len(weak)}', flush=True)

    p = R / a.out
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'[abl] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

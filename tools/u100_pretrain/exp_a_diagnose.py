"""Experiment A: is the writer transformation global or character-dependent?

Uses the cached DINOv2-B CLS features of Top10 training images
(style_bank_top10.npz). Per (writer, char) the images are averaged, which
removes the single-writing noise eps and leaves the writer's character-level
style point f[w,c]. For every writer pair (A,B) and shared char c:

    delta_AB(c) = f[A,c] - f[B,c]

Decomposition: delta_AB(c) = mu_AB + r(c), where mu_AB is the writer-pair
mean (the GLOBAL style-difference hypothesis: one vector per pair) and r(c)
is the character-dependent residual. Reported:
  1. global_energy: ||mu||^2 / E||delta||^2  (how much a single vector explains)
  2. residual predictability, leave-one-char-out within each pair:
       - pred_mean:   delta_hat = mu_AB                      (global baseline)
       - pred_knn:    delta_hat = structure-weighted mean of other chars'
                        deltas (structure = DINO CLS of the FONT STD render,
                        softmax over -distance / tau)
       - pred_1nn:    delta of the structurally nearest other char
     scored by cosine(pred, delta) and relative L2 error ||pred-delta||/||delta||.
  3. cross-pair consistency of the residual: is r_AB(c) for the same char
     correlated across writer pairs (char-driven) or independent (noise)?
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--min-writers', type=int, default=4)
    parser.add_argument('--min-imgs', type=int, default=2)
    parser.add_argument('--tau', type=float, default=0.15)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_a_diagnosis.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    d = np.load(args.bank)
    feats, writers, chars = d['features'], d['writers'], d['chars']
    scripts = d['scripts']

    # per (writer, char) mean feature (averages away single-writing noise)
    wc = defaultdict(list)
    wscript = {}
    for i in range(len(chars)):
        wc[(int(writers[i]), int(chars[i]))].append(feats[i])
        wscript[int(chars[i])] = int(scripts[i])
    wc_mean = {k: np.mean(v, axis=0) for k, v in wc.items() if len(v) >= args.min_imgs}
    by_char = defaultdict(list)
    for (w, c) in wc_mean: by_char[c].append(w)
    eligible = sorted(c for c, ws in by_char.items() if len(set(ws)) >= args.min_writers)
    if len(eligible) < 10: raise ValueError('too few shared characters')

    # structure embedding: DINO CLS of the font STD render per eligible char
    import torch
    from transformers import AutoModel
    from PIL import Image
    model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None].cuda()
    std = torch.tensor([.229, .224, .225])[None, :, None, None].cuda()
    import csv
    std_img = {}
    for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig')):
        std_img[(int(r['script_id']), r['character'])] = r['path']
    remap = json.loads((Path(args.std_root).parent / 'tables' / 'font_remap.json').read_text()) \
        if (Path(args.std_root).parent / 'tables' / 'font_remap.json').exists() else {'0': 0, '3': 1, '4': 2}
    embeds = {}
    with torch.inference_mode():
        for i in range(0, len(eligible), 32):
            chunk = eligible[i:i + 32]
            imgs = []
            for c in chunk:
                p = std_img.get((wscript[c], chr(c))) or std_img.get((0, chr(c)))
                g = np.asarray(Image.open(Path(args.std_root) / p).convert('L').resize((224, 224), Image.BICUBIC), dtype=np.float32)[None, None] / 255
                imgs.append(torch.from_numpy(g))
            x = torch.cat(imgs).cuda()
            x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1).expand(-1, 3, -1, -1)
            x = (x - mean) / std
            with torch.autocast('cuda', dtype=torch.bfloat16):
                e = model(pixel_values=x).last_hidden_state[:, 0]
            e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
            for c, e_ in zip(chunk, e): embeds[c] = e_
    E = np.stack([embeds[c] for c in eligible])
    S = E @ E.T  # structure cosine similarity among eligible chars
    S = (S - S.mean()) / S.std()

    pair_stats = []
    global_energy_all, cos_mean_all, cos_knn_all, cos_1nn_all = [], [], [], []
    relerr_mean_all, relerr_knn_all, relerr_1nn_all = [], [], []
    pairs_used = 0
    for a in range(10):
        for b in range(a + 1, 10):
            shared = [c for c in eligible if a in by_char[c] and b in by_char[c]]
            if len(shared) < 12: continue
            deltas = np.stack([wc_mean[(a, c)] - wc_mean[(b, c)] for c in shared])
            mu = deltas.mean(0, keepdims=True)
            resid = deltas - mu
            g_energy = float((mu ** 2).sum() / (deltas ** 2).sum(1, keepdims=True).mean())
            idx = {c: i for i, c in enumerate(shared)}
            cos_m, cos_k, cos_1, re_m, re_k, re_1 = [], [], [], [], [], []
            for j, c in enumerate(shared):
                others = [i for i in range(len(shared)) if i != j]
                sim = S[j, [idx[x] for x in [shared[i] for i in others]]]
                w = np.exp(sim / args.tau); w /= w.sum()
                pred_knn = (w[:, None] * deltas[others]).sum(0)
                nn = others[int(np.argmax(sim))]
                for pred, cs, re in ((mu[0], cos_m, re_m), (pred_knn, cos_k, re_k), (deltas[nn], cos_1, re_1)):
                    cs.append(float(pred @ deltas[j] / (np.linalg.norm(pred) * np.linalg.norm(deltas[j]) + 1e-9)))
                    re.append(float(np.linalg.norm(pred - deltas[j]) / np.linalg.norm(deltas[j])))
            global_energy_all.append(g_energy)
            cos_mean_all.append(np.mean(cos_m)); cos_knn_all.append(np.mean(cos_k)); cos_1nn_all.append(np.mean(cos_1))
            relerr_mean_all.append(np.mean(re_m)); relerr_knn_all.append(np.mean(re_k)); relerr_1nn_all.append(np.mean(re_1))
            pair_stats.append(dict(pair=f'{a}-{b}', n_chars=len(shared), global_energy=g_energy,
                                   cos_mean=float(np.mean(cos_m)), cos_knn=float(np.mean(cos_k)),
                                   cos_1nn=float(np.mean(cos_1))))
            pairs_used += 1

    # cross-pair residual consistency for chars shared by >=2 pairs
    resid_by_char = defaultdict(list)
    for a in range(10):
        for b in range(a + 1, 10):
            shared = [c for c in eligible if a in by_char[c] and b in by_char[c]]
            if len(shared) < 12: continue
            deltas = np.stack([wc_mean[(a, c)] - wc_mean[(b, c)] for c in shared])
            mu = deltas.mean(0, keepdims=True)
            for i, c in enumerate(shared): resid_by_char[c].append((f'{a}-{b}', deltas[i] - mu[0]))
    resid_by_pair = defaultdict(dict)
    for c, v in resid_by_char.items():
        for p, r in v: resid_by_pair[p][c] = r
    pair_ids = sorted(resid_by_pair)
    cors = []
    for i in range(len(pair_ids)):
        for j in range(i + 1, len(pair_ids)):
            common = sorted(set(resid_by_pair[pair_ids[i]]) & set(resid_by_pair[pair_ids[j]]))
            if len(common) < 4: continue
            x = np.concatenate([resid_by_pair[pair_ids[i]][c] for c in common])
            y = np.concatenate([resid_by_pair[pair_ids[j]][c] for c in common])
            if x.std() > 1e-9 and y.std() > 1e-9:
                cors.append(dict(pair=f'{pair_ids[i]}~{pair_ids[j]}', n_chars=len(common),
                                 corr=float(np.corrcoef(x, y)[0, 1])))
    summary = dict(
        eligible_chars=len(eligible), writer_pairs=pairs_used,
        min_imgs_per_wc=args.min_imgs,
        global_energy=dict(mean=float(np.mean(global_energy_all)),
                           per_pair={p['pair']: round(p['global_energy'], 3) for p in pair_stats}),
        loo_predictability=dict(
            cos_global_mean=float(np.mean(cos_mean_all)), cos_structknn_mean=float(np.mean(cos_knn_all)),
            cos_1nn_mean=float(np.mean(cos_1nn_all)),
            relerr_global_mean=float(np.mean(relerr_mean_all)),
            relerr_structknn_mean=float(np.mean(relerr_knn_all)),
            relerr_1nn_mean=float(np.mean(relerr_1nn_all))),
        residual_cross_pair_corr=dict(mean=float(np.mean([c['corr'] for c in cors])),
                                      median=float(np.median([c['corr'] for c in cors])),
                                      n_pair_pairs=len(cors),
                                      top=[{**c, 'corr': round(c['corr'], 3)} for c in sorted(cors, key=lambda x: -abs(x['corr']))[:5]]),
        note='global_energy=1.0 means a single per-pair vector explains all char-level style difference; '
             'residual cross-pair corr>0 means the char-dependent part is char-driven rather than noise')
    out.write_text(json.dumps(dict(summary=summary, per_pair=pair_stats), indent=1))
    print('EXP_A_DONE', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()

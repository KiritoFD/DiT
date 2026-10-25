"""Round-1 addition metrics: stroke-width distribution and writer-identity judge.

Width (per character, gen and GT identically): binarize ink, crop bbox, scale
normalize bbox height to 128, EDT ridge sampling (local maxima of the distance
transform) -> P10/P50/P90 in bbox-height units; Width-W1 = Wasserstein-1 between
gen and GT ridge-width samples. Deterministic; no learned component.

Writer judge (independent, frozen): DINOv2-B CLS features of generations scored
against per-writer centroids from the Top10 training feature bank; Macro Top-1
and Margin = cos(target) - max_{other} cos(target). The judge is never trained
on generations.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BBOX_NORM = 128.0


def ridge_widths(gray):
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    mask = arr < 0.5
    if not mask.any(): return None
    ys, xs = np.where(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = BBOX_NORM / h
    from PIL import Image
    img = Image.fromarray(crop.astype(np.uint8) * 255)
    nw = max(1, int(round(w * scale)))
    resized = np.asarray(img.resize((nw, int(BBOX_NORM)), Image.LANCZOS), dtype=np.float32) / 255.0
    mask = resized >= 0.5
    if not mask.any(): return None
    edt = distance_transform_edt(mask)
    ridge = mask & (edt >= maximum_filter(edt, size=5))
    samples = 2.0 * edt[ridge] - 1.0 if ridge.any() else 2.0 * edt[mask] - 1.0
    samples = samples[samples > 0]
    if not len(samples): return None
    return samples / BBOX_NORM


def percentiles(samples):
    return tuple(float(np.percentile(samples, p)) for p in (10, 50, 90))


class DinoJudge:
    def __init__(self, feature_bank):
        data = np.load(feature_bank)
        feats = data['features'] / np.linalg.norm(data['features'], axis=1, keepdims=True)
        writers = data['writers']
        self.centroids = np.stack([feats[writers == w].mean(0) for w in range(10)])
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406])[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225])[None, :, None, None]

    @torch.inference_mode()
    def features(self, gray_batches):
        out = []
        for batch in gray_batches:
            x = torch.from_numpy(batch).cuda().float() / 255
            x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic',
                                                align_corners=False, antialias=True).clamp_(0, 1)
            x = x.expand(-1, 3, -1, -1)
            x = (x - self.mean.cuda()) / self.std.cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feat = self.model(pixel_values=x).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(feat.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @staticmethod
    def score(feats, targets, centroids):
        sims = feats @ centroids.T
        pred = sims.argmax(1)
        correct = pred == targets
        margin = np.empty(len(targets), dtype=np.float32)
        for i, t in enumerate(targets):
            others = [j for j in range(len(centroids)) if j != t]
            margin[i] = sims[i, t] - sims[i, others].max()
        macro = float(np.mean([correct[targets == w].mean() for w in np.unique(targets)]))
        return dict(top1_micro=float(correct.mean()), top1_macro=macro,
                    margin_mean=float(margin.mean()),
                    margin_by_correct=dict(correct=float(margin[correct].mean()),
                                           wrong=float(margin[~correct].mean())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True, help='dir with g{i}.png and gt{i}.png')
    parser.add_argument('--n', type=int, default=187)
    parser.add_argument('--features', required=True, help='Top10 training feature bank npz')
    parser.add_argument('--batch-csv', required=True, help='eval_stdskel_batch.csv for char/writer per idx')
    parser.add_argument('--meta-csv', default='', help='rewritten eval CSV (row order == idx) with character/calligrapher_id')
    parser.add_argument('--judge-target', choices=('gen', 'gt'), default='gen',
                        help='score writer judge on generations or on GT (sanity check)')
    parser.add_argument('--probe', default='', help='writer_probe.pt from build_probe; overrides centroids')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    samples = Path(args.samples)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    meta = {}
    if args.meta_csv:
        with open(args.meta_csv, encoding='utf-8-sig') as f:
            for idx, row in enumerate(csv.DictReader(f)):
                meta[idx] = (row['character'], int(row['calligrapher_id']))
        meta = {k: meta[k] for k in sorted(meta)}
    else:
        with open(args.batch_csv, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                meta[int(row['idx'])] = (row['char'], int(row['calligrapher']))
    judge = DinoJudge(args.features)

    rows = []
    gray_gens = []
    for i in range(args.n):
        gen = Image.open(samples / f'g{i}.png').convert('L')
        gt = Image.open(samples / f'gt{i}.png').convert('L')
        wg, wt = ridge_widths(gen), ridge_widths(gt)
        char, callig = meta[i]
        row = dict(idx=i, char=char, calligrapher=callig)
        if wg is not None and wt is not None:
            p10, p50, p90 = percentiles(wg)
            q10, q50, q90 = percentiles(wt)
            row.update(gen_w_p10=p10, gen_w_p50=p50, gen_w_p90=p90,
                       gt_w_p10=q10, gt_w_p50=q50, gt_w_p90=q90,
                       width_w1=float(wasserstein_distance(wg, wt)),
                       heavy_p90_excess=p90 - q90,
                       heavy_flag=bool(p90 > q90 * 1.15))
        else:
            row.update(gen_w_p10=None, width_w1=None, heavy_flag=None)
        rows.append(row)
        judge_img = gt if args.judge_target == 'gt' else gen
        gray_gens.append(np.asarray(judge_img, dtype=np.float32)[None, None])

    feats = judge.features(gray_gens)
    targets = np.array([meta[i][1] for i in range(args.n)])
    if args.probe:
        probe = torch.load(args.probe, map_location='cpu', weights_only=False)
        logits = feats @ probe['coef'].T + probe['intercept']
        logits = logits - logits.max(1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        pred = probs.argmax(1)
        correct = pred == targets
        margin = np.empty(len(targets), dtype=np.float32)
        for i, t in enumerate(targets):
            others = [j for j in range(10) if j != t]
            margin[i] = probs[i, t] - probs[i, others].max()
        writer = dict(kind='probe', val_top1=float(probe['val_top1']),
                      top1_micro=float(correct.mean()),
                      top1_macro=float(np.mean([correct[targets == w].mean() for w in np.unique(targets)])),
                      margin_mean=float(margin.mean()),
                      margin_by_correct=dict(correct=float(margin[correct].mean()),
                                             wrong=float(margin[~correct].mean())))
    else:
        writer = DinoJudge.score(feats, targets, judge.centroids)
        writer['kind'] = 'centroid'
    preds = (feats @ judge.centroids.T).argmax(1) if not args.probe else pred
    for i in range(args.n):
        rows[i]['writer_pred'] = int(preds[i])

    widths = [r for r in rows if r.get('width_w1') is not None]
    heavy = [r for r in rows if r.get('heavy_flag')]
    summary = dict(n=args.n, writer=writer,
                   width=dict(
                       gen_p50_mean=float(np.mean([r['gen_w_p50'] for r in widths])),
                       gen_p90_mean=float(np.mean([r['gen_w_p90'] for r in widths])),
                       gt_p50_mean=float(np.mean([r['gt_w_p50'] for r in widths])),
                       gt_p90_mean=float(np.mean([r['gt_w_p90'] for r in widths])),
                       w1_mean=float(np.mean([r['width_w1'] for r in widths])),
                       w1_median=float(np.median([r['width_w1'] for r in widths])),
                       heavy_p90_excess_mean=float(np.mean([r['heavy_p90_excess'] for r in widths])),
                       heavy_flag_rate=len(heavy) / len(widths)))
    Path(str(out) + '.json').write_text(json.dumps(summary, indent=2))
    fields = list(rows[0].keys())
    with open(str(out) + '.csv', 'w', newline='', encoding='utf-8') as f:
        writer_csv = csv.DictWriter(f, fieldnames=fields)
        writer_csv.writeheader(); writer_csv.writerows(rows)
    print('ROUND1_DONE', json.dumps(summary))


if __name__ == '__main__':
    main()

"""Independent writer probe trained ONLY on VAE-decoded training images.

The generator's latent domain (SD VAE decodes) differs from raw scans, so the
probe is built from posterior-mean latents of Top10 training rows decoded by
the same VAE, then DINOv2-B CLS features and a logistic head with a stratified
10% validation split. No generation output is ever seen by this judge.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}
VAE_SCALE = 0.18215


def dino_features(gray_batch, model, mean, std):
    x = gray_batch.cuda().float() / 255
    x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic',
                                        align_corners=False, antialias=True).clamp_(0, 1)
    x = x.expand(-1, 3, -1, -1)
    x = (x - mean) / std
    with torch.autocast('cuda', dtype=torch.bfloat16), torch.inference_mode():
        feat = model(pixel_values=x).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(feat.float(), dim=-1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='assets/reference_top10_20261024/train.csv')
    parser.add_argument('--latents', default='assets/generic48_kxl_top10_20261022/latents')
    parser.add_argument('--vae', default='pretrained_models/sd-vae-ft-ema')
    parser.add_argument('--per-writer', type=int, default=600)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/writer_probe.pt')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)

    rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8-sig')) if r['generic'] == '0']
    by_writer = {w: [] for w in range(10)}
    for r in rows:
        by_writer[TOP10_RAW[int(r['calligrapher_id'])]].append(int(r['img_id']))
    rng = np.random.default_rng(90)
    picked = {}
    for w, ids in by_writer.items():
        if len(ids) < args.per_writer: raise ValueError(f'writer {w} has {len(ids)} rows')
        picked[w] = rng.choice(ids, size=args.per_writer, replace=False)
    want = {int(v): w for w, ids in picked.items() for v in ids}

    latents = np.empty((len(want), 4, 32, 32), dtype=np.float16)
    labels = np.empty(len(want), dtype=np.int64)
    filled = np.zeros(len(want), dtype=bool)
    order = list(want.keys())
    shard_paths = sorted(Path(args.latents).glob('shard_*.npz'))
    for p in shard_paths:
        data = np.load(p)
        pos = {int(v): i for i, v in enumerate(data['img_ids'])}
        for j, v in enumerate(order):
            if filled[j] or v not in pos: continue
            latents[j] = data['latents'][pos[v]]
            labels[j] = want[v]
            filled[j] = True
        if filled.all(): break
    if not filled.all(): raise ValueError(f'missing latents for {int((~filled).sum())} ids')

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, local_files_only=True).cuda().eval().requires_grad_(False)
    from transformers import AutoModel
    dino = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None].cuda()
    std = torch.tensor([.229, .224, .225])[None, :, None, None].cuda()

    feats = np.empty((len(want), 768), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(want), 32):
            z = torch.from_numpy(latents[s:s + 32].astype(np.float32)).cuda() / VAE_SCALE
            img = vae.decode(z).sample.clamp_(-1, 1)
            img = (img / 2 + 0.5).clamp_(0, 1)
            gray = (img[:, 0] * .299 + img[:, 1] * .587 + img[:, 2] * .114)[:, None]
            feats[s:s + 32] = dino_features(gray.cpu().numpy(), dino, mean, std)
            if (s // 32) % 20 == 0: print('DECODE', s, len(want), flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(np.arange(len(want)), test_size=.1, stratify=labels, random_state=90)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(feats[tr], labels[tr])
    val_pred = clf.predict(feats[va])
    val_top1 = float((val_pred == labels[va]).mean())
    per_writer = {int(w): float((val_pred[labels[va] == w] == w).mean()) for w in range(10)}
    torch.save(dict(coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
                    val_top1=val_top1, per_writer_val=per_writer, n_train=len(tr),
                    protocol='logistic on DINOv2-B CLS of VAE-decoded Top10 train latents; no generations'), out)
    print('PROBE_READY', json.dumps(dict(val_top1=round(val_top1, 4), per_writer=per_writer)))


if __name__ == '__main__':
    main()

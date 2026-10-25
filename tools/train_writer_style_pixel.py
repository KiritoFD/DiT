"""Controlled Multi-loss Siamese CNN Pixel Writer Encoders (M0-M4 Ablation Matrix).

Supports:
1. Feature Architectures: Pure D1 ('none'), 1D Haar DWT ('1d'), 2D Spatial Haar DWT ('2d').
2. Metric Losses: D1 Relative Ranking ('ranking'), CJIG 2023 Euclidean Contrastive ('contrastive').
3. Same-Script Cosine CE Masking.
4. 384 Quartet + 128 Class-Balanced Mixed Microbatch.
5. CPU uint8 Data Caching with asynchronous GPU batch transfer.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.writer_style_pixel import PixelWriterV2, WriterStylePixel, augment_gray
from src.utils.writer_quartets import WriterQuartets, build_same_script_mask
from src.loss.metric_learning import compute_metric_loss, same_script_ce_loss
from tools.analysis.style_fingerprint_features import matched_pairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out, splitroot):
    """Prepare offline uint8 CPU dataset cache with strict test leakage guards."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Preparation target directory is not empty')
    
    d = np.load('scratch/disc_lat_full.npz', allow_pickle=True)
    w, s, c, source = [d[k].astype(str) for k in ('writer', 'script', 'char', 'work')]
    rows = [r for r in csv.DictReader(open('exp-std/csv/train.csv', encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    assert len(rows) == len(w) and all((r['calligrapher'], r['script'], r['character']) == (ww, ss, cc) for r, ww, ss, cc in zip(rows, w, s, c))

    split = np.load(Path(splitroot) / 'split_indices.npz')
    splits = {k: split[k] for k in ('train', 'val_char', 'test_char', 'test_source')}
    allrows = np.concatenate(list(splits.values()))
    assert len(allrows) == len(set(allrows.tolist()))

    train = splits['train']
    assert not set(c[train]) & set(c[splits['val_char']])
    assert not set(c[train]) & set(c[splits['test_char']])
    assert not set(zip(w[train], s[train], source[train])) & set(zip(w[splits['test_source']], s[splits['test_source']], source[splits['test_source']]))

    # Exclude official eval200 characters from train
    eval200_chars = set()
    if os.path.exists('exp-std/csv/eval200_fixed.csv'):
        erows = list(csv.DictReader(open('exp-std/csv/eval200_fixed.csv', encoding='utf-8-sig')))[:187]
        eval200_chars = {r['character'] for r in erows if r.get('character')}
        assert not set(c[train]) & eval200_chars, "Leakage detected: eval200 characters present in training set!"

    slots = sorted(set(zip(w, s)))
    scripts = sorted(set(s))
    labels = np.array([slots.index(x) for x in zip(w, s)])
    si = np.array([scripts.index(x) for x in s])
    slot_scripts = [scripts.index(s_name) for _, s_name in slots]

    sampler = WriterQuartets(train, labels, si, c, source)
    assert not sampler.coverage['slots_without_full_rectangles'], sampler.coverage

    statsfile = Path(splitroot) / 'ink_width_stats.npz'
    stats = np.load(statsfile)
    digest = sha('scratch/disc_lat_full.npz')
    assert str(stats['data_sha256']) == digest

    def read_img(row):
        with Image.open(row['image_path']) as im:
            return np.asarray(im.convert('L').resize((128, 128), Image.Resampling.BICUBIC), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=16) as pool:
        gray = np.stack(list(pool.map(read_img, rows)))[:, None].copy()

    matched = {key: matched_pairs(splits[key], train, w, s, c, source, stats['stats'])[0] for key in ('val_char', 'test_char', 'test_source')}
    assert len(matched['test_char']) == 847 and len(matched['test_source']) == 196

    payload = {
        'gray': torch.from_numpy(gray),  # Stays uint8 on CPU
        'writer': w, 'script': s, 'char': c, 'source': source,
        'labels': labels, 'si': si, 'slots': slots, 'scripts': scripts,
        'slot_scripts': slot_scripts,
        'splits': splits, 'stats': stats['stats'], 'matched_pairs': matched
    }
    torch.save(payload, out / 'data.pt')

    manifest = {
        'data_sha256': digest,
        'csv_sha256': sha('exp-std/csv/train.csv'),
        'split_sha256': sha(Path(splitroot) / 'split_indices.npz'),
        'stats_sha256': sha(statsfile),
        'slots': slots,
        'scripts': scripts,
        'split_sizes': {k: len(v) for k, v in splits.items()},
        'matched_counts': {k: len(v) for k, v in matched.items()},
        'quartet_coverage': sampler.coverage,
        'data_cache_sha256': sha(out / 'data.pt')
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'split_indices.npz', **splits)
    print('PIXEL_DATA_PREPARED', json.dumps(manifest, ensure_ascii=False), flush=True)


@torch.no_grad()
def embed(model, gray, indices, batch=256, perturb=None):
    """Extract features in batches. gray is CPU uint8 tensor."""
    features = []
    pred = []
    for start in range(0, len(indices), batch):
        ids = indices[start:start + batch]
        x = gray[torch.as_tensor(ids, dtype=torch.long)].cuda(non_blocking=True).float() / 255.0
        if perturb == 'blur':
            x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        elif perturb == 'ink_gain':
            x = (1.0 - (1.0 - x) * 0.85).clamp(0, 1)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(x)
        features.append(o['embedding'].float().cpu().numpy())
        pred.extend(o['slot_logits'].argmax(1).cpu().tolist())
    return np.concatenate(features), np.array(pred)


def evaluate_features(features, pred, data, query, reference):
    labels, si, chars, source = [data[k] for k in ('labels', 'si', 'char', 'source')]
    nslots = len(data['slots'])
    proto = []
    for slot in range(nslots):
        ix = reference[labels[reference] == slot]
        means = []
        for src in sorted(set(source[ix])):
            e = features[ix[source[ix] == src]].mean(0)
            means.append(e / max(np.linalg.norm(e), 1e-12))
        e = np.mean(means, axis=0) if means else np.zeros(features.shape[1])
        proto.append(e / max(np.linalg.norm(e), 1e-12))
    proto = np.array(proto)
    slot_si = np.array([data['scripts'].index(x[1]) for x in data['slots']])

    hits = []
    aps = []
    pair_acc = []
    rng = np.random.default_rng(89)

    for start in range(0, len(query), 128):
        ids = query[start:start + 128]
        scores = features[ids] @ features[reference].T
        ps = features[ids] @ proto.T
        for j, i in enumerate(ids):
            candidates = (si[reference] == si[i]) & (chars[reference] != chars[i])
            relevant = labels[reference[candidates]] == labels[i]
            if relevant.any() and not relevant.all():
                ranked = relevant[np.argsort(-scores[j, candidates], kind='stable')]
                aps.append(float(((np.cumsum(ranked) / (np.arange(len(ranked)) + 1)) * ranked).sum() / ranked.sum()))
            allowed = np.flatnonzero(slot_si == si[i])
            hits.append(allowed[ps[j, allowed].argmax()] == labels[i])
            neg = query[(si[query] == si[i]) & (chars[query] == chars[i]) & (labels[query] != labels[i])]
            if len(neg):
                k = int(rng.choice(neg))
                m = float((features[i] - features[k]) @ proto[labels[i]])
                pair_acc.append(float(m > 0) + 0.5 * float(m == 0))

    return {
        'crosschar_map': float(np.mean(aps)) if aps else 0.0,
        'prototype_top1': float(np.mean(hits)) if hits else 0.0,
        'slot_ce_top1': float(np.mean(pred[query] == labels[query])),
        'prototype_targeted_samechar_2afc': float(np.mean(pair_acc)) if pair_acc else None
    }


def matched_score(features, pairs):
    m = (features[pairs[:, 0]] * features[pairs[:, 1]]).sum(1) - (features[pairs[:, 0]] * features[pairs[:, 2]]).sum(1)
    return float(((m > 0).astype(float) + 0.5 * (m == 0)).mean())


def resolve_arm_config(arm):
    """Map M0-M4 arm to (haar_mode, metric_loss)."""
    mapping = {
        'M0': ('none', 'ranking'),
        'M1': ('none', 'contrastive'),
        'M2': ('1d', 'ranking'),
        'M3': ('2d', 'ranking'),
        'M4': ('2d', 'contrastive'),
        # Backward compatibility for D0/D1
        'D0': ('none', 'none'),
        'D1': ('none', 'ranking')
    }
    if arm not in mapping:
        raise ValueError(f"Unknown arm: {arm}, choices: {list(mapping.keys())}")
    return mapping[arm]


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cudnn.benchmark = True

    data = torch.load(Path(a.data_root) / 'data.pt', map_location='cpu', weights_only=False)
    gray = data['gray']  # uint8 on CPU
    tr = data['splits']['train']
    val = data['splits']['val_char']
    labels = torch.from_numpy(data['labels'])
    slot_scripts = data.get('slot_scripts', [data['scripts'].index(x[1]) for x in data['slots']])

    # Resolve feature structure and metric loss
    def_haar, def_metric = resolve_arm_config(a.arm)
    haar_mode = a.haar_mode or def_haar
    metric_loss_type = a.metric_loss or def_metric

    model = PixelWriterV2(n_slots=len(data['slots']), haar_mode=haar_mode, haar_dropout=0.1).cuda()
    params = sum(p.numel() for p in model.parameters())

    sampler = WriterQuartets(tr, data['labels'], data['si'], data['char'], data['source'], a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, fused=True)

    initial_hash = hashlib.sha256()
    for parameter in model.parameters():
        initial_hash.update(parameter.detach().cpu().numpy().tobytes())

    config = {
        **vars(a),
        'parameters': params,
        'haar_mode': haar_mode,
        'metric_loss': metric_loss_type,
        'data_manifest_sha256': sha(Path(a.data_root) / 'manifest.json'),
        'initial_parameter_sha256': initial_hash.hexdigest(),
        'warmup_ce_steps': max(1, math.ceil(3 * len(tr) / (a.batch * a.grad_accum))),
        'cross_ramp_steps': max(1, math.ceil(10 * len(tr) / (a.batch * a.grad_accum)))
    }

    best = (-1, -1)
    start = time.time()
    last = start
    beststep = 0
    first = 1

    if a.resume:
        checkpoint = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        opt.load_state_dict(checkpoint['optimizer'])
        best = tuple(checkpoint['best_validation'])
        beststep = checkpoint['best_step']
        first = checkpoint['step'] + 1
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        sampler.rng.bit_generator.state = checkpoint['sampler_rng']
    else:
        (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    warmup = config['warmup_ce_steps']
    cross_ramp = config['cross_ramp_steps']

    def save_ckpt(name, step):
        torch.save({
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'step': step,
            'config': config,
            'best_validation': best,
            'best_step': beststep,
            'slots': data['slots'],
            'scripts': data['scripts'],
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
            'sampler_rng': sampler.rng.bit_generator.state
        }, out / name)

    print(f"PIXEL_TRAIN_START arm={a.arm} haar={haar_mode} metric={metric_loss_type} params={params} batch={a.batch} accum={a.grad_accum} steps={a.steps}", flush=True)

    ce_sum = 0.0
    metric_sum = 0.0

    for step in range(first, a.steps + 1):
        # Cosine LR schedule with linear warmup
        lr = a.lr * min(step / warmup, 1.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / a.steps)))
        for group in opt.param_groups:
            group['lr'] = lr

        ramp = min(max((step - warmup) / cross_ramp, 0.0), 1.0)
        weight = 0.5 * ramp if metric_loss_type != 'none' else 0.0

        model.train()
        opt.zero_grad(set_to_none=True)

        for accum_idx in range(a.grad_accum):
            ids, sampling = sampler.sample_mixed(total_batch=a.batch, quartet_samples=a.quartet_samples)
            ids_cpu = torch.as_tensor(ids, dtype=torch.long)
            batch_targets = labels[ids_cpu].cuda(non_blocking=True)

            # Async fetch from CPU uint8 array and augment
            raw = gray[ids_cpu].cuda(non_blocking=True).float() / 255.0
            x = augment_gray(raw)

            # Build same-script competition mask
            script_mask = build_same_script_mask(batch_targets, slot_scripts, len(data['slots']))

            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(x, script_mask=script_mask)
                ce = F.cross_entropy(o['masked_logits'], batch_targets)

                if metric_loss_type != 'none':
                    m_loss, lossmetrics = compute_metric_loss(
                        o['embedding'].float(), data['char'][ids],
                        loss_type=metric_loss_type,
                        n_quartet_samples=a.quartet_samples,
                        contrastive_margin=a.contrastive_margin
                    )
                else:
                    m_loss = ce.new_zeros(())
                    lossmetrics = {}

                step_loss = (ce + weight * m_loss) / a.grad_accum

            step_loss.backward()
            ce_sum += float(ce.detach()) / a.grad_accum
            metric_sum += float(m_loss.detach()) / a.grad_accum

        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if not torch.isfinite(step_loss):
            raise RuntimeError('Nonfinite training loss detected')

        if step % 20 == 0 or step == a.steps:
            now = time.time()
            n_report = 20 if step % 20 == 0 else (step % 20)
            entry = {
                'step': step,
                'ce': ce_sum / n_report,
                'metric_loss': metric_sum / n_report,
                'metric_weight': weight,
                'lr': lr,
                'grad_norm': float(grad),
                'steps_per_sec': n_report / (now - last),
                'elapsed_sec': now - start,
                'gpu_mem_gb': torch.cuda.max_memory_allocated() / (1024**3),
                **sampling,
                **lossmetrics
            }
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(entry) + '\n')
            print('TRAIN', json.dumps(entry), flush=True)
            last = now
            ce_sum = 0.0
            metric_sum = 0.0

        # Periodic Validation on held-out characters
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            indices = np.r_[tr, val]
            emb, pr = embed(model, gray, indices)
            allfeatures = np.zeros((len(gray), 256), dtype=np.float32)
            allpred = np.zeros(len(gray), dtype=int)
            allfeatures[indices] = emb
            allpred[indices] = pr

            result = evaluate_features(allfeatures, allpred, data, val, tr)
            result['matched_2afc'] = matched_score(allfeatures, data['matched_pairs']['val_char'])
            result['step'] = step

            score = (result['matched_2afc'], result['crosschar_map'])
            if score > best:
                best = score
                beststep = step
                save_ckpt('best.pt', step)
            save_ckpt('last.pt', step)

            with (out / 'validation.jsonl').open('a') as f:
                f.write(json.dumps(result) + '\n')
            if step in (1000, 3000, 5000, 10000) or step % 2000 == 0 or step == a.steps:
                save_ckpt(f'step{step:07d}.pt', step)
            print('VALIDATION', json.dumps(result), flush=True)
            last = time.time()

    print(f"PIXEL_TRAIN_COMPLETE arm={a.arm} best_step={beststep} best_score={best}", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true', help='Prepare dataset cache')
    p.add_argument('--data-root', default='assets/data')
    p.add_argument('--split-root', default='scratch/v89_teacher_v3_20261016_01/assets/encoder')
    p.add_argument('--out', required=True)
    p.add_argument('--arm', choices=['M0', 'M1', 'M2', 'M3', 'M4', 'D0', 'D1'], default='M3')
    p.add_argument('--haar-mode', choices=['none', '1d', '2d'], default=None)
    p.add_argument('--metric-loss', choices=['ranking', 'contrastive', 'none'], default=None)
    p.add_argument('--contrastive-margin', type=float, default=1.0)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--quartet-samples', type=int, default=384)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=89)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--resume', default='')
    a = p.parse_args()

    if a.prepare:
        prepare(a.out, a.split_root)
    else:
        train(a)

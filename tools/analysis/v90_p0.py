"""Read-only v90 gates: exact pair coverage and fixed-input writer-swap FM.

No optimizer, checkpoint updates, or implicit permission to launch P1/P2.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def pair_cells(rows):
    """Unique (script, character, unordered writer pair), not augmented row pairs."""
    grouped = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        if not (row.get('aug') or '').strip():
            grouped[row['script'], row['character']][row['calligrapher']].append(i)
    return [(s, c, a, b, writers[a], writers[b])
            for (s, c), writers in sorted(grouped.items())
            for a, b in itertools.combinations(sorted(writers), 2)]


def distribution_shift(rows, probability, key, baseline_rows=None):
    baseline_rows = rows if baseline_rows is None else baseline_rows
    original = Counter(r[key] for r in baseline_rows)
    pair = defaultdict(float)
    for r, p in zip(rows, probability):
        pair[r[key]] += float(p)
    keys = sorted(set(original) | set(pair))
    p = np.array([original[k] / len(baseline_rows) for k in keys])
    q = np.array([pair[k] for k in keys])
    assert np.isclose(q.sum(), 1)
    mixed = .75 * p + .25 * q
    middle = (p + mixed) / 2
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    records = [{'key': k, 'original': float(x), 'pair_only': float(y),
                'mixed_25pct': float(z), 'mixed_ratio': float(z / x) if x else None}
               for k, x, y, z in zip(keys, p, q, mixed)]
    return {'js_nats': .5 * (kl(p, middle) + kl(mixed, middle)),
            'pair_top10_mass': float(np.sort(q)[-10:].sum()),
            'pair_effective_categories': float(1 / (q @ q)),
            'largest_upweight': sorted(records, key=lambda r: -r['mixed_ratio'])[:20],
            'largest_downweight': sorted(records, key=lambda r: r['mixed_ratio'])[:20]}


def coverage(rows, baseline_rows=None):
    assert rows and all(not (r.get('aug') or '').strip() for r in rows)
    cells = pair_cells(rows)
    if not cells:
        return {'rows': len(rows), 'legal_cells': 0}
    by_pair = defaultdict(list)
    sources = Counter()
    pair_sources = defaultdict(Counter)
    for cell in cells:
        s, c, a, b, ia, ib = cell
        by_pair[s, a, b].append(cell)
        for i in ia + ib:
            sources[rows[i].get('source', 'unknown')] += 1
            pair_sources[s, a, b][rows[i].get('source', 'unknown')] += 1
    # Exact endpoint probabilities; no Monte Carlo error or materialized row-pair explosion.
    distributions = {}
    for mode in ('uniform_cell', 'uniform_writer_pair_then_character'):
        prob = np.zeros(len(rows))
        for s, c, a, b, ia, ib in cells:
            mass = (1 / len(cells) if mode == 'uniform_cell'
                    else 1 / len(by_pair) / len(by_pair[s, a, b]))
            for ids in (ia, ib):
                prob[ids] += mass / 2 / len(ids)
        distributions[mode] = {key: distribution_shift(rows, prob, key, baseline_rows)
                               for key in ('calligrapher', 'script', 'character', 'source')}
    pairs = []
    for (s, a, b), cc in sorted(by_pair.items()):
        pairs.append({'script': s, 'writers': [a, b], 'unique_characters': len(cc),
                      'distinct_endpoint_rows': len({i for x in cc for i in x[4] + x[5]}),
                      'row_pair_products': sum(len(x[4]) * len(x[5]) for x in cc),
                      'source_endpoint_cell_counts': dict(pair_sources[s, a, b])})
    return {'rows': len(rows), 'writers': len({r['calligrapher'] for r in rows}),
            'characters': len({r['character'] for r in rows}), 'legal_cells': len(cells),
            'writer_script_pairs': len(pairs),
            'baseline_sampling_rows': len(rows if baseline_rows is None else baseline_rows),
            'pairs_with_fewer_than_32_characters': sum(x['unique_characters'] < 32 for x in pairs),
            'pair_eligible_row_fraction': len({i for x in cells for i in x[4]+x[5]}) / len(rows),
            'source_endpoint_cell_counts': dict(sources), 'pairs': pairs,
            'expected_25pct_pair_distribution': distributions,
            'limitations': 'source is a dataset/source label, not a verified artwork ID. '
                            'Natural non-augmented rows only; augmented variants must never be counted as new characters.'}


def legal_wrong_slots(rows, train):
    reps = {}
    for r in train:
        reps.setdefault((r['calligrapher'], r['script']), r)
    options = []
    for r in rows:
        wrong = [key for key in sorted(reps) if key[1] == r['script'] and key[0] != r['calligrapher']]
        if not wrong:
            raise ValueError(f'No legal same-script writer for {r}')
        options.append(wrong)
    return reps, options


def fm_metrics(v_true, v_wrong, target):
    def mse(x):
        return (x - target).float().flatten(1).square().mean(1)
    good, wrong = mse(v_true), mse(v_wrong)
    return {'correct_mse': good, 'wrong_mse': wrong, 'delta_fm': wrong-good,
            'relative_gap': (wrong-good) / good.clamp_min(1e-12),
            'relative_velocity_change': (v_wrong-v_true).float().flatten(1).norm(dim=1)
            / v_true.float().flatten(1).norm(dim=1).clamp_min(1e-12)}


def fixed_input_forward(model, z, eps, t, true_kwargs, wrong_kwargs):
    import torch
    # Fail closed: every non-writer condition must remain exactly unchanged.
    writer_keys = {'y_callig', 'y_callig_raw', 'y_pair'}
    if set(true_kwargs) != set(wrong_kwargs):
        raise ValueError('Condition key mismatch')
    for k in set(true_kwargs) - writer_keys:
        if not torch.equal(true_kwargs[k], wrong_kwargs[k]):
            raise ValueError(f'Non-writer condition changed: {k}')
    if torch.equal(true_kwargs['y_callig'], wrong_kwargs['y_callig']):
        raise ValueError('Writer was not changed')
    xt = (1-t[:, None, None, None])*z + t[:, None, None, None]*eps
    target = eps-z
    def predict(kw):
        value = model(xt, t*1000., **kw)
        if isinstance(value, tuple):
            value = value[0]
        return value[:, :z.shape[1]]
    return fm_metrics(predict(true_kwargs), predict(wrong_kwargs), target)


def summarize(records, group='character', repetitions=1000):
    """Cluster bootstrap: repeated seeds/times/wrong writers are NOT independent rows."""
    grouped = defaultdict(list)
    for r in records:
        grouped[r[group]].append(r)
    rng = np.random.default_rng(90)
    units = list(grouped.values())
    result = {'n_observations': len(records), 'n_clusters': len(units), 'cluster': group}
    for key in ('delta_fm', 'relative_gap', 'relative_velocity_change'):
        values = np.array([r[key] for r in records])
        sums = np.array([sum(r[key] for r in rr) for rr in units])
        counts = np.array([len(rr) for rr in units])
        ix = rng.integers(0, len(units), (repetitions, len(units)))
        means = sums[ix].sum(1) / counts[ix].sum(1)
        positive_sums = np.array([sum(r[key] > 0 for r in rr) for rr in units])
        positive_means = positive_sums[ix].sum(1) / counts[ix].sum(1)
        result[key] = {'mean': float(values.mean()), 'median': float(np.median(values)),
                       'mean_ci95': np.quantile(means, [.025, .975]).tolist(),
                       'positive_fraction': float(np.mean(values > 0)),
                       'positive_fraction_ci95': np.quantile(positive_means, [.025, .975]).tolist()}
    return result


def assemble_report(root, out):
    """Validate the actual evidence, never infer that an unfinished gate passed."""
    result = {'stage': 'P0', 'P1_started': False, 'P2_started': False,
              'decision': 'HOLD: high-noise writer sensitivity is clear; D2/D3 transfer is still a separate gate.'}
    train = read_rows(root/'exp-std/csv/train.csv')
    evaluation = read_rows(root/'exp-std/csv/eval200_fixed.csv')
    image_overlap = set(r['image_path'] for r in train) & set(r['image_path'] for r in evaluation)
    result['train_eval_image_path_overlap'] = sorted(image_overlap)
    if image_overlap:
        raise ValueError('Generation evaluation contains training images')
    result['eval_csv_sha256'] = sha(root/'exp-std/csv/eval200_fixed.csv')
    result['independent_judge_sha256'] = sha(root/'assets/style_disc_ce_holdout.pt')
    result['generation_evidence'] = {}
    runs = {'v84A200k': ('v84A_std_cond/20261013-072215-v84A-std-cond', '0200000'),
            'C10k': ('v86_full_fixed_10k_C/20261015-112206-v86-full-fixed-10k-C', '0010000')}
    for name, (run, step) in runs.items():
        run_root = root/'assets/runs'/run
        image_dir = run_root/f'eval_samples_ctrl/step{step}/eval200fix'
        expected = {f'g{i}.png' for i in range(len(evaluation))}
        actual = {p.name for p in image_dir.glob('g*.png') if p.stem[1:].isdigit()}
        assert actual == expected, ('Incomplete or misnumbered generation set', name)
        result['generation_evidence'][name] = {
            'run': str(run_root), 'checkpoint_sha256': sha(run_root/f'checkpoints/{step}.pt'),
            'per_image_sha256': [sha(image_dir/f'g{i}.png') for i in range(len(evaluation))]}
    fm_path = out/'fm.json'
    if fm_path.exists():
        fm = json.loads(fm_path.read_text(encoding='utf-8'))
        records = fm['per_image']
        assert len(records) == len(evaluation)*5*fm['seeds']
        supported = {(x['calligrapher'], x['script']) for x in train}
        for r in records:
            assert r['writer'] != r['wrong_writer']
            assert (r['wrong_writer'], r['script']) in supported
        for key in ('all',):
            fm[key] = summarize(records)
        for field in ('bin', 'writer', 'script', 'source', 'character'):
            fm['by_'+field] = {str(k): summarize([r for r in records if r[field]==k])
                               for k in sorted({r[field] for r in records})}
        result['fm'] = {k: v for k, v in fm.items() if k != 'per_image'}
        result['fm']['all_by_source_cluster'] = summarize(records, 'source')
        result['fm']['all_by_writer_cluster'] = summarize(records, 'writer')
        result['fm_sha256'] = sha(fm_path)
    paired_path = out/'C10k_per_image.json'
    if paired_path.exists():
        paired = json.loads(paired_path.read_text(encoding='utf-8'))
        a, b = paired['v84A200k'], paired['C10k']
        assert len(a) == len(b) == len(evaluation)
        deltas = []
        for old, new in zip(a, b):
            for key in ('row', 'writer', 'script', 'character', 'source'):
                assert old[key] == new[key]
            deltas.append({'character': old['character'],
                           'delta_fm': float(new['top1'])-float(old['top1']),
                           'relative_gap': new['writer_margin']-old['writer_margin'],
                           'relative_velocity_change': new['std_margin']-old['std_margin']})
        # Names in the generic bootstrap helper are mapped explicitly, never presented as FM.
        summary = summarize(deltas)
        result['C10k_paired'] = {'n': len(a),
            'top1_difference': summary['delta_fm'], 'writer_margin_difference': summary['relative_gap'],
            'std_margin_difference': summary['relative_velocity_change'],
            'improved': sum(not x['top1'] and y['top1'] for x, y in zip(a, b)),
            'worsened': sum(x['top1'] and not y['top1'] for x, y in zip(a, b)),
            'limitations': 'Same fixed generation seed; bootstrap by character, not an independent multi-seed validation.'}
        result['C10k_per_image_sha256'] = sha(paired_path)
    result['coverage_sha256'] = sha(out/'coverage.json')
    result['code_sha256'] = sha(Path(__file__))
    result['missing_P0_evidence'] = ['D2/D3 final GT acceptance and multi-seed independent gradient transfer',
                                    'expanded fixed-STD/noise writer-switch generation beyond the existing 8-character pilot']
    (out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('P0_REPORT_COMPLETE', json.dumps(result.get('C10k_paired', {})), flush=True)


def run_fm(a, root, out, train):
    import torch
    from src.eval.model_io import load_model_from_ckpt
    from src.eval.in_mem_eval import _get_cache
    from src.eval.sampler import model_kwargs_for_slice
    from diffusers import AutoencoderKL
    from PIL import Image
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    rows = read_rows(root/'exp-std/csv/eval200_fixed.csv')
    reps, options = legal_wrong_slots(rows, train)
    rep_keys = sorted(reps)
    combined = rows + [reps[k] for k in rep_keys]
    csv_path = out/'conditions.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in combined))))
        w.writeheader(); w.writerows(combined)
    # Encode real HELDOUT GT, not classifier-heldout samples seen by DiT.
    # A fixed posterior sample follows the training posterior convention, no domain substitution.
    vae_path = root/'data/pretrained/sd-vae-ft-ema'
    vae = AutoencoderKL.from_pretrained(str(vae_path)).to(a.device).eval()
    vae.requires_grad_(False)
    rng = torch.Generator(device=a.device).manual_seed(9001)
    zz = []
    for i, r in enumerate(rows):
        im = Image.open(root/r['image_path']).convert('RGB').resize((256, 256), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.array(im).copy()).permute(2, 0, 1).unsqueeze(0).float().to(a.device)/127.5-1
        posterior = vae.encode(x).latent_dist
        zz.append((posterior.sample(generator=rng)*.18215).cpu())
    z = torch.cat(zz)
    torch.save(z, out/'gt_latents.pt')
    del vae, x, posterior
    if a.device == 'cuda':
        torch.cuda.empty_cache()
    ckpt = root/'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/checkpoints/0200000.pt'
    model, args = load_model_from_ckpt(str(ckpt), device=a.device, use_ema=True)
    model.eval().requires_grad_(False)
    assert model.std_soft_cond and model.content_cond == 'std'
    args.std_soft_root = str(root/'exp-std/data/std_soft_glyphs_v1')
    args.std_soft_eval_sigma = .65
    cache = _get_cache(str(csv_path), len(combined), None, '', args)
    def kwargs(indices, glyph_indices):
        cond = [cache['conds'][i] for i in indices]
        return model_kwargs_for_slice(model,
            yc=torch.tensor([c[0] for c in cond], device=a.device),
            yh=torch.tensor([cache['conds'][i][1] for i in glyph_indices], device=a.device),
            hier_slice=tuple([[axis[i] for i in indices] for axis in cache['hier_conds']]),
            skel_slice=cache['skels_latent'][glyph_indices], device=a.device)
    records = []
    bins = [(.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95)]
    for seed in range(a.seeds):
        generator = torch.Generator().manual_seed(90+seed)
        noise = torch.randn(z.shape, generator=generator)
        for bi, (lo, hi) in enumerate(bins):
            times = lo+(hi-lo)*torch.rand(len(rows), generator=generator)
            for start in range(0, len(rows), a.batch):
                ids = list(range(start, min(start+a.batch, len(rows))))
                # Round-robin legal writers per seed, deterministic, not chosen for a large gap.
                swapped = [options[i][(seed+bi+i) % len(options[i])] for i in ids]
                other = [len(rows)+rep_keys.index(k) for k in swapped]
                goodkw, badkw = kwargs(ids, ids), kwargs(other, ids)
                with torch.autocast(a.device, dtype=torch.bfloat16, enabled=a.device=='cuda'):
                    metrics = fixed_input_forward(model, z[ids].to(a.device), noise[ids].to(a.device),
                                                   times[ids].to(a.device), goodkw, badkw)
                for j, i in enumerate(ids):
                    records.append({'row': i, 'character': rows[i]['character'], 'writer': rows[i]['calligrapher'],
                        'script': rows[i]['script'], 'source': rows[i].get('source', 'unknown'),
                        'wrong_writer': swapped[j][0], 'seed': seed, 'bin': bi,
                        't': float(times[i]), 'true_y_callig': int(goodkw['y_callig'][j]),
                        'wrong_y_callig': int(badkw['y_callig'][j]),
                        'noise_sha256': hashlib.sha256(noise[i].numpy().tobytes()).hexdigest(),
                        **{k: float(v[j]) for k, v in metrics.items()}})
            print('FM_BIN_COMPLETE', seed, bi, len(records), flush=True)
            (out/'fm_partial.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    result = {'checkpoint_sha256': sha(ckpt), 'weights': 'ema', 'n_gt': len(rows),
        'vae_path': str(vae_path), 'sf': .18215, 'shift': 0., 'posterior': 'sample, fp32, generator seed9001',
        'gt_latents_sha256': sha(out/'gt_latents.pt'), 'std_sigma': .65, 'cfg': 'none, conditional forward',
        'time_bins': bins, 'seeds': a.seeds, 'all': summarize(records),
        'by_bin': {str(b): summarize([r for r in records if r['bin']==b]) for b in range(5)},
        'by_writer': {w: summarize([r for r in records if r['writer']==w]) for w in sorted({r['writer'] for r in records})},
        'by_character': {c: summarize([r for r in records if r['character']==c]) for c in sorted({r['character'] for r in records})},
        'by_script': {s: summarize([r for r in records if r['script']==s]) for s in sorted({r['script'] for r in records})},
        'by_source': {s: summarize([r for r in records if r['source']==s]) for s in sorted({r['source'] for r in records})},
        'per_image': records,
        'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30 if a.device=='cuda' else None,
        'limitations': 'Teacher-forced FM conditional sensitivity, not full generation controllability. '
                        'Source labels are not verified artwork IDs. 3 noise seeds, one fixed posterior sample per GT.'}
    (out/'fm.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FM_AUDIT_COMPLETE', json.dumps(result['all']), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['coverage', 'fm', 'report'], required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--memory-fraction', type=float, default=.035)
    p.add_argument('--batch', type=int, default=2); p.add_argument('--seeds', type=int, default=3)
    a = p.parse_args(); root = Path(a.root).resolve(); out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True); os.chdir(root)
    train_csv = root/'exp-std/csv/train.csv'
    train = [r for r in read_rows(train_csv) if not (r.get('aug') or '').strip()]
    if a.mode == 'coverage':
        aug_csv = root/'exp-std/csv/train_top10_aug_sym.csv'
        # Report both natural-image distribution and the actual augmented DiT row sampler.
        augmented = read_rows(aug_csv)
        result = {'train_csv_sha256': sha(train_csv), 'augmented_train_csv_sha256': sha(aug_csv),
                  'primary': coverage(train, augmented), 'natural_image_reference': coverage(train),
                  'gate': 'P1/P2 NOT started; pair coverage alone cannot authorize training.'}
        (out/'coverage.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('COVERAGE_COMPLETE', {k: v for k, v in result['primary'].items() if not isinstance(v, (dict, list))}, flush=True)
    elif a.mode == 'report':
        assemble_report(root, out)
    else:
        if (out/'fm.json').exists():
            raise ValueError('Refusing to overwrite completed audit')
        run_fm(a, root, out, train)


if __name__ == '__main__':
    main()

"""Paired holdout deltas: average seeds per image before bootstrapping images.

No treating different seeds of the same GT image as independent samples.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

METRICS = ['ssim', 'lpips_alex', 'lpips_vgg', 'ink_ssim', 'ink_iou',
           'frag', 'frag_error', 'hole', 'ink_area_ratio']


def summarize(directory, subset, reference='baseline'):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    wanted = manifest['splits'][subset]
    results = {}
    raw_csv = root / 'paired_samples.csv'
    csv_rows = list(csv.DictReader(raw_csv.open(encoding='utf-8'))) if raw_csv.exists() else []
    for recipe in manifest['recipes']:
        name = recipe['name']
        values = []
        for seed in manifest['seeds']:
            result_path = root / f'{name}_seed{seed}.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
            else:
                selected = [r for r in csv_rows if r['name'] == name and int(r['seed']) == seed]
                if not selected:
                    raise FileNotFoundError(result_path)
                result = dict(indices=[int(r['idx']) for r in selected],
                              per_image=[{m: float(r[m]) for m in METRICS} for r in selected])
            positions = {image_id: i for i, image_id in enumerate(result['indices'])}
            values.append([[result['per_image'][positions[i]][m] for m in METRICS] for i in wanted])
        results[name] = np.asarray(values)
    baseline = results[reference]
    rng = np.random.default_rng(20261012)
    resamples = rng.integers(0, len(wanted), (5000, len(wanted)))
    stats = dict(n_images=len(wanted), seeds=manifest['seeds'], subset=subset, reference=reference,
                 note='Paired CI resamples images after averaging seeds; conditional on these seed realizations.',
                 checkpoint_sha256=manifest['checkpoint_sha256'], comparisons={})
    for name, values in results.items():
        delta = values - baseline
        across_seed_image = delta.mean(axis=0)
        boot = across_seed_image[resamples].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        stats['comparisons'][name] = {
            m: dict(mean=float(values[:, :, j].mean()), baseline=float(baseline[:, :, j].mean()),
                    delta=float(delta[:, :, j].mean()), ci95=[float(lo[j]), float(hi[j])],
                    seed_deltas=[float(d) for d in delta[:, :, j].mean(axis=1)])
            for j, m in enumerate(METRICS)}
    return stats


def export_evidence(directory, destination):
    """Compact tracked numeric evidence; no checkpoints or generated image blobs."""
    source, dest = Path(directory), Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    for path in [source / 'manifest.json', source / 'summary.csv', *source.glob('paired_*.json')]:
        shutil.copyfile(path, dest / path.name)
    with (dest / 'paired_samples.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'seed', 'idx', *METRICS])
        writer.writeheader()
        for recipe in manifest['recipes']:
            for seed in manifest['seeds']:
                result = json.loads((source / f"{recipe['name']}_seed{seed}.json").read_text(encoding='utf-8'))
                for idx, metrics in zip(result['indices'], result['per_image']):
                    writer.writerow(dict(name=recipe['name'], seed=seed, idx=idx,
                                         **{m: metrics[m] for m in METRICS}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--subset', choices=['tune', 'holdout', 'all'], default='holdout')
    parser.add_argument('--reference', default='baseline')
    parser.add_argument('--evidence-out')
    args = parser.parse_args()
    stats = summarize(args.directory, args.subset, args.reference)
    suffix = '' if args.reference == 'baseline' else '_vs_' + args.reference
    path = Path(args.directory) / f'paired_{args.subset}{suffix}.json'
    path.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    for name, metrics in stats['comparisons'].items():
        print(name, ' '.join(f'{m}={metrics[m]["mean"]:.6f} delta={metrics[m]["delta"]:+.6f}'
                            for m in ['ssim', 'lpips_alex', 'lpips_vgg', 'ink_iou', 'frag_error']))
    if args.evidence_out:
        export_evidence(args.directory, args.evidence_out)


if __name__ == '__main__':
    main()

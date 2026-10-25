"""CLI: fit a training reference, or calibrate existing PNGs without a model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import scipy
import skimage
from skimage.metrics import structural_similarity

from .geometry import Settings, calibrate, gray, measure


SCHEMA = 'glyph-width-reference-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    for key in ('image_path', 'calligrapher', 'script'):
        if key not in rows[0]:
            raise ValueError(f'Missing {key}: {path}')
    return rows


def sample_key(row):
    return row['image_path'].strip().replace('\\', '/')


def index_rows(rows):
    mapping = {sample_key(row): i for i, row in enumerate(rows)}
    if len(mapping) != len(rows):
        raise ValueError('Duplicate image_path keys; supply an unambiguous evaluation CSV')
    return mapping


def read_image(path, size):
    with Image.open(path) as im:
        im = im.convert('RGB')
        if im.size != (size, size):
            raise ValueError(f'{path}: {im.size}, expected {(size,size)}; no implicit resize')
        return np.asarray(im)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def protocol(settings, size):
    return dict(settings=asdict(settings), image_size=size, width='2*EDT on medial axis',
                foreground_connectivity=8, background_connectivity=4,
                numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__)


def build_reference(args):
    settings = Settings(threshold=args.threshold, seed=args.seed)
    rows = read_rows(args.csv)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)

    def extract(row):
        path = Path(args.image_root) / row['image_path']
        stats, *_ = measure(read_image(path, args.size), settings)
        return row, digest(path), stats

    slots, hashes, invalid = {}, set(), 0
    # Ordered map keeps floating-point aggregation deterministic even with threads.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, sha, stats in pool.map(extract, rows):
            hashes.add(sha)
            if not stats['valid']:
                invalid += 1
                continue
            slot = row['calligrapher']+'_'+row['script']
            slots.setdefault(slot, []).append(stats)
    if not slots:
        raise ValueError('No valid training images')
    table = {slot: dict(n=len(items), **{k: float(np.mean([x[k] for x in items]))
                                       for k in ('p10', 'p50', 'p90')})
             for slot, items in sorted(slots.items())}
    result = dict(schema=SCHEMA, role=args.role, aggregation='mean_of_per_image_quantiles',
                  csv=str(Path(args.csv).resolve()), csv_sha256=digest(args.csv),
                  source_image_sha256=sorted(hashes), n_rows=len(rows), n_invalid=invalid,
                  protocol=protocol(settings, args.size), slots=table)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    print(f'Reference: {len(table)} slots, {len(rows)-invalid}/{len(rows)} valid -> {destination}')


def resolve_targets(args, settings):
    if args.mode != 'reference':
        return None
    table = json.loads(Path(args.reference).read_text(encoding='utf-8'))
    if table.get('schema') != SCHEMA or table.get('role') != 'train':
        raise ValueError('Reference must be fitted with this tool and --role train; legacy tables are not certified')
    p = table['protocol']
    if (p['image_size'], p['settings']['threshold'], p['settings']['seed']) != (args.size, settings.threshold, settings.seed):
        raise ValueError('Reference size/threshold/seed differs from current measurement')
    if p['skimage'] != skimage.__version__ or p['scipy'] != scipy.__version__:
        raise ValueError('Reference SciPy/skimage versions differ; refit in the current environment')
    return table


def image_metrics(image, gt, threshold=.5):
    a, b = gray(image), gray(gt)
    ma, mb = (1-a) >= threshold, (1-b) >= threshold
    return dict(ssim_gray=float(structural_similarity(a, b, data_range=1.)),
                ink_iou=float((ma & mb).sum()/max(1, (ma | mb).sum())))


def apply(args):
    settings = Settings(args.threshold, args.seed, args.min_area, args.max_shift, args.strength)
    rows = read_rows(args.csv)
    index_rows(rows)
    indices = [int(i) for i in args.indices.split(',')] if args.indices else list(range(len(rows)))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(rows) for i in indices):
        raise ValueError('Invalid or duplicate indices')
    reference = resolve_targets(args, settings)
    gt_rows = read_rows(args.gt_csv or args.csv) if args.gt_dir else None
    gt_index = index_rows(gt_rows) if gt_rows else None
    if args.mode == 'oracle' and not args.gt_dir:
        raise ValueError('Oracle requires --gt-dir; never used implicitly')
    source_dir, output = Path(args.input), Path(args.output)
    if output.resolve() == source_dir.resolve():
        raise ValueError('Output must not overwrite original images')
    if output.exists():
        raise FileExistsError(output)
    # Check target provenance before creating artifacts, including GT leakage when GT is available.
    gt_paths = {}
    for idx in indices:
        if not (source_dir/f'g{idx}.png').exists():
            raise FileNotFoundError(source_dir/f'g{idx}.png')
        if gt_index is not None:
            p = Path(args.gt_dir)/f'g{gt_index[sample_key(rows[idx])]}.png'
            if not p.exists():
                raise FileNotFoundError(p)
            gt_paths[idx] = p
    reference_hashes = set(reference['source_image_sha256']) if reference else set()
    if any(digest(p) in reference_hashes for p in gt_paths.values()):
        raise ValueError('Training reference overlaps evaluation GT by file content; use explicit oracle mode for diagnosis')
    lpips_fn = None
    if args.lpips:
        import torch
        import lpips
        lpips_fn = lpips.LPIPS(net=args.lpips, verbose=False).to(args.device).eval()
        lpips_fn.requires_grad_(False)

    output.mkdir(parents=True)
    manifest = dict(schema='glyph-calibration-v1', status='incomplete', mode=args.mode,
                    oracle=args.mode == 'oracle', method=args.method, protocol=protocol(settings, args.size),
                    input=str(source_dir.resolve()), csv=str(Path(args.csv).resolve()), csv_sha256=digest(args.csv),
                    reference=str(Path(args.reference).resolve()) if args.reference else None,
                    reference_sha256=digest(args.reference) if args.reference else None,
                    gt_csv_sha256=digest(args.gt_csv or args.csv) if args.gt_dir else None,
                    lpips=args.lpips, lpips_device=args.device if args.lpips else None, indices=indices)
    write_json(output/'run.json', manifest)
    records = []
    for idx in indices:
        row = rows[idx]
        path = source_dir/f'g{idx}.png'
        image = read_image(path, args.size)
        gt = read_image(gt_paths[idx], args.size) if idx in gt_paths else None
        target, delta = None, None
        if args.mode == 'reference':
            slot = row['calligrapher']+'_'+row['script']
            target = {k: reference['slots'][slot][k] for k in ('p10','p50','p90')}
        elif args.mode == 'oracle':
            gt_stats, *_ = measure(gt, settings)
            if not gt_stats['valid']:
                raise ValueError(f'Empty GT at {idx}: cannot define an oracle target')
            target = {k: gt_stats[k] for k in ('p10','p50','p90')}
        elif args.delta is not None:
            delta = args.delta
        else:
            target = {'p50': args.p50}
        adjusted, result = calibrate(image, target=target, delta=delta, method=args.method, settings=settings)
        dest = output/path.name
        if result['changed']:
            Image.fromarray(adjusted).save(dest)
        else:
            shutil.copy2(path, dest)  # Identity also preserves PNG bytes and metadata.
        record = dict(idx=idx, sample_key=sample_key(row), character=row.get('character',''),
                      calligrapher=row['calligrapher'], script=row['script'], **result,
                      input_sha256=digest(path), output_sha256=digest(dest))
        if gt is not None:
            record['gt_sha256'] = digest(gt_paths[idx])
            record['gt'] = measure(gt, settings)[0]
            record['quality_before'] = image_metrics(image, gt, settings.threshold)
            record['quality_after'] = image_metrics(adjusted, gt, settings.threshold)
            if lpips_fn is not None:
                def tensor(im):
                    return torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0).float().to(args.device)/127.5-1
                with torch.inference_mode():
                    t = tensor(gt)
                    record['quality_before']['lpips'] = float(lpips_fn(tensor(image), t).item())
                    record['quality_after']['lpips'] = float(lpips_fn(tensor(adjusted), t).item())
        records.append(record)
        print(f'{idx}: {result["before"]["p50"]:.2f} -> {result["after"]["p50"]:.2f} {result["status"]}', flush=True)
    write_json(output/'records.json', records)
    flat = []
    for r in records:
        entry = {k:v for k,v in r.items() if not isinstance(v, dict)}
        for group in ('before','after','target','gt','quality_before','quality_after'):
            entry.update({group+'_'+k:v for k,v in (r.get(group) or {}).items()})
        flat.append(entry)
    with (output/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader()
        writer.writerows(flat)
    summary = {}
    if gt_paths:
        for phase in ('before', 'after'):
            errors = [abs(r[phase]['p50']/r['gt']['p50']-1) for r in records if r['gt']['valid']]
            summary[phase] = dict(
                p50_mean_absolute_relative_error=float(np.mean(errors)) if errors else None,
                n_valid_gt=len(errors),
                **{key: float(np.mean([r['quality_'+phase][key] for r in records]))
                   for key in records[0]['quality_'+phase]})
    manifest.update(status='complete', n=len(records), n_changed=sum(r['changed'] for r in records),
                    n_unchanged=sum(not r['changed'] for r in records), summary=summary)
    write_json(output/'run.json', manifest)
    print(f'Complete: {len(records)} images -> {output}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('reference', help='Fit per-slot quantiles on explicitly declared training images')
    fit.add_argument('--csv', required=True)
    fit.add_argument('--image-root', default='.')
    fit.add_argument('--role', choices=['train'], required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--workers', type=int, default=4)
    run = sub.add_parser('apply', help='Calibrate existing PNGs; originals are never overwritten')
    for name in ('csv','input','output'):
        run.add_argument('--'+name, required=True)
    run.add_argument('--mode', choices=['manual','reference','oracle'], required=True)
    run.add_argument('--reference')
    controls = run.add_mutually_exclusive_group()
    controls.add_argument('--delta', type=float, help='Positive thickens, negative thins; pixels')
    controls.add_argument('--p50', type=float, help='Manual absolute target width in source pixels')
    run.add_argument('--method', choices=['uniform','profile'], default='uniform')
    run.add_argument('--max-shift', type=float, default=2.)
    run.add_argument('--strength', type=float, default=1.)
    run.add_argument('--min-area', type=int, default=1, help='1 checks all components/holes; larger ignores declared tiny debris')
    run.add_argument('--gt-dir')
    run.add_argument('--gt-csv', help='GT directory indexing CSV; defaults to --csv')
    run.add_argument('--indices', help='Comma-separated indices; default is every CSV row')
    run.add_argument('--lpips', choices=['alex','vgg'], help='Optional; no silent zero fallback')
    run.add_argument('--device', default='cpu')
    for p in (fit, run):
        p.add_argument('--size', type=int, default=256)
        p.add_argument('--threshold', type=float, default=.5)
        p.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'reference':
        if args.workers < 1:
            parser.error('--workers must be positive')
        return build_reference(args)
    if args.mode == 'reference' and not args.reference:
        parser.error('--mode reference requires --reference')
    if args.mode == 'manual' and args.delta is None and args.p50 is None:
        parser.error('--mode manual requires --delta or --p50')
    if args.mode != 'manual' and (args.delta is not None or args.p50 is not None):
        parser.error('--delta/--p50 belong to manual mode only')
    if args.mode != 'reference' and args.reference:
        parser.error('--reference belongs to reference mode only')
    if args.gt_csv and not args.gt_dir:
        parser.error('--gt-csv requires --gt-dir')
    return apply(args)


if __name__ == '__main__':
    main()

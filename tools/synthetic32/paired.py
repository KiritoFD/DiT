"""Create a GT-held-out style-only pilot or score its outputs with repo metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, root):
    from PIL import Image
    original = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / 'inputs', root / 'inputs')
    (root / 'gt').mkdir()
    test = Path('assets/data_cohorts/top20_20261021_v1/datasets_final/top20/test.csv')
    with test.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    jobs = []
    for wi, writer in enumerate(original['writers']):
        refs = original['reference_records'][writer]
        excluded = {r['character'] for r in refs}
        candidates = [r for r in rows if r['calligrapher'] == writer
                      and r['script'] == '楷' and r['character'] not in excluded]
        seen = set()
        for row in candidates:
            char = row['character']
            if char in seen:
                continue
            seen.add(char)
            name = f'U{ord(char):04X}_w{wi}_s20261022'
            gt = f'gt/{name}.png'
            im = Image.open(row['image_path']).convert('RGB')
            if im.size != (256, 256):
                raise ValueError(f'Unexpected native GT size: {im.size}')
            im.save(root / gt)
            jobs.append(dict(id=name, character=char, writer=writer, seed=20261022,
                             mode='style_only', images=[r['file'] for r in refs],
                             gt_file=gt, gt_sha256=sha(root / gt), gt_record=row))
            if len(seen) == 2:
                break
        if len(seen) != 2:
            raise ValueError(f'Insufficient paired test characters: {writer}')
    original.update(jobs=jobs, n=len(jobs), seeds=[20261022], targets=list(dict.fromkeys(j['character'] for j in jobs)),
                    evaluation='Frozen test GT; target GT never enters image conditions',
                    test_csv_sha256=sha(test))
    original['input_sha256'] = {str(p.relative_to(root)): sha(p)
                                for p in (root / 'inputs').glob('*.png')}
    (root / 'manifest.json').write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding='utf-8')
    with tarfile.open(root / 'paired.tar.gz', 'w:gz') as tar:
        for name in ('inputs', 'gt', 'manifest.json'):
            tar.add(root / name, arcname=name)
    print('PAIRED', [(j['writer'], j['character']) for j in jobs], flush=True)


def score(root):
    import numpy as np
    import torch
    from PIL import Image
    from src.eval.metrics import ssim, mse, frag_ratio, get_lpips
    from src.eval.metrics_ink import ink_iou, ink_ssim
    torch.set_num_threads(4)
    lp, error = get_lpips('cuda', net='alex')
    if lp is None:
        raise RuntimeError(error)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    results = []
    def canonical(a):
        ys, xs = np.where(a.mean(-1) < .5)
        if not len(xs):
            return a, 0.
        crop = Image.fromarray(np.uint8(np.clip(a, 0, 1)*255)).crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
        span = max(crop.size)/256
        scale = 256/max(crop.size)
        size = tuple(max(1, min(256, round(v*scale))) for v in crop.size)
        image = Image.new('RGB', (256, 256), 'white')
        image.paste(crop.resize(size, Image.Resampling.LANCZOS), ((256-size[0])//2, (256-size[1])//2))
        return np.asarray(image, dtype=np.float32)/255, span
    for job in manifest['jobs']:
        gt = root / job['gt_file']
        if sha(gt) != job['gt_sha256']:
            raise ValueError(f'GT changed: {gt}')
        pred_path = root / 'outputs' / (job['id'] + '_256.png')
        pred = np.asarray(Image.open(pred_path).convert('RGB'), dtype=np.float32) / 255
        target = np.asarray(Image.open(gt).convert('RGB'), dtype=np.float32) / 255
        tensor = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
        cp, spanp = canonical(pred)
        cg, spang = canonical(target)
        with torch.inference_mode():
            distance = float(lp(tensor(pred), tensor(target)).item())
            bbox_distance = float(lp(tensor(cp), tensor(cg)).item())
        results.append(dict(id=job['id'], writer=job['writer'], character=job['character'],
                            ssim=ssim(pred, target), lpips=distance, mse=mse(pred, target),
                            ink_iou=ink_iou(pred, target), ink_ssim=ink_ssim(pred, target),
                            frag_ratio=frag_ratio(pred, target),
                            soft_ink_ratio=float((1-pred).mean()/max((1-target).mean(), 1e-8)),
                            pred_bbox_span=spanp, gt_bbox_span=spang,
                            diagnostic_bbox_ssim=ssim(cp, cg), diagnostic_bbox_lpips=bbox_distance,
                            pred_sha256=sha(pred_path)))
    with (root / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    keys = ('ssim', 'lpips', 'mse', 'ink_iou', 'ink_ssim', 'frag_ratio', 'soft_ink_ratio',
            'pred_bbox_span', 'gt_bbox_span', 'diagnostic_bbox_ssim', 'diagnostic_bbox_lpips')
    summary = dict(n=len(results), protocol='Native GT 256; generated LANCZOS 256; no alignment/width correction',
                   means={k: float(np.mean([r[k] for r in results])) for k in keys},
                   per_writer={w: {k: float(np.mean([r[k] for r in results if r['writer']==w]))
                                   for k in keys} for w in manifest['writers']})
    (root / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['prepare', 'score'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path)
    a = p.parse_args()
    prepare(a.source, a.root) if a.mode == 'prepare' else score(a.root)

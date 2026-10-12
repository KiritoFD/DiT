"""Measure existing poster PNG batches with one metric contract; no generation.

CPU LPIPS-alex + Gaussian SSIM11, ink-IoU, frag; fixed eval200fix N=187.
Each input pixel file is hashed. Rows use the actual displayed batch, not a
different checkpoint's training log. --only can resume a missing row safely.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from render_v68_v70_sde_poster import ROOT, OUT, image_rows, sha

import sys
sys.path.insert(0, str(ROOT))
from src.eval.inference import _ssim
from src.eval.metrics_ink import ink_iou
from src.eval.metrics import frag_ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    import lpips
    perceptual = lpips.LPIPS(net='alex', verbose=False).eval()
    rows = image_rows() + [('v80_best', 'v80-best · ODE', '190k · SSIM最佳 / 用户选定综合折中',
        'assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/'
        'eval_samples_ctrl/step0190000/eval200fix/g{idx}.png')]
    result_path = OUT / 'row_metrics.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
    for key, title, desc, pattern in rows:
        if key == 'gt' or (args.only and key not in args.only):
            continue
        paths = [ROOT / pattern.format(idx=i) for i in range(187)]
        if not all(p.exists() for p in paths):
            print(f'MISSING {key}: {sum(p.exists() for p in paths)}/187', flush=True)
            continue
        hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
        gt_paths = [ROOT / f'eval/gt/{i}.png' for i in range(187)]
        gt_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in gt_paths}
        if key in result and result[key]['image_sha256'] == hashes and result[key]['gt_sha256'] == gt_hashes:
            print(f'SKIP {key}', flush=True)
            continue
        per_image = []
        for i, path in enumerate(paths):
            pred = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
            gt = np.asarray(Image.open(gt_paths[i]).convert('RGB'), dtype=np.float32) / 255
            assert pred.shape == gt.shape == (256, 256, 3)
            p = torch.from_numpy(pred.transpose(2, 0, 1).copy())[None] * 2 - 1
            g = torch.from_numpy(gt.transpose(2, 0, 1).copy())[None] * 2 - 1
            per_image.append(dict(idx=i, ssim=_ssim(pred, gt), lpips_alex=float(perceptual(p, g).mean()),
                                  ink_iou=ink_iou(pred, gt), frag=frag_ratio(pred, gt)))
        result[key] = dict(title=title, description=desc, n=187, source_pattern=pattern,
                           protocol='PNG RGB float01; SSIM Gaussian win11 sigma1.5; LPIPS-alex; ink thresh0.5; frag8conn',
                           image_sha256=hashes, gt_sha256=gt_hashes, per_image=per_image,
                           mean={m: float(np.mean([r[m] for r in per_image]))
                                 for m in ['ssim', 'lpips_alex', 'ink_iou', 'frag']})
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'DONE {key}: {result[key]["mean"]}', flush=True)
    with (OUT / 'row_metrics_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['row', 'n', 'ssim', 'lpips_alex', 'ink_iou', 'frag'])
        writer.writeheader()
        writer.writerows(dict(row=key, n=r['n'], **r['mean']) for key, r in result.items())


if __name__ == '__main__':
    main()

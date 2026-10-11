"""Paired, multi-metric inference sweep; loads EMA/VAE once, never trains.

python tools/analysis/tune_current_checkpoint.py --ckpt ... --recipes ...
  --out ... --subset tune --seeds 0 --vae-shift -0.03341800242750449
Use the same manifest and disjoint holdout to confirm selected recipes.
The explicit VAE shift must come from shard provenance, not a tuned value.
"""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from src.eval.inference import make_eval_cache, sample_latents, _ssim, _mse
from src.eval.in_mem_eval import _triple_remap_cache, _get_vae
from src.eval.model_io import build_model_from_args, apply_post_construction
from src.eval.metrics_ink import ink_ssim, ink_iou
from src.eval.metrics import frag_ratio, hole_ratio
from src.loss.flow_matching import FlowMatching


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def split_indices(rows, tune_n=80):
    """Fixed shuffled round-robin across calligrapher x script; no GT metrics."""
    rng = np.random.default_rng(20261012)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row['calligrapher_id'], row['script_id']), []).append(i)
    queues = [list(rng.permutation(groups[k])) for k in sorted(groups)]
    order = []
    while any(queues):
        for group in queues:
            if group:
                order.append(int(group.pop()))
    tune = sorted(order[:tune_n])
    holdout = sorted(order[tune_n:])
    return dict(tune=tune, holdout=holdout, all=list(range(len(rows))))


def sliced(cache, indices, seed):
    # Generate noise for the FULL CSV before selecting rows: per-image pairing
    # survives different subset sizes. Seed controls both x_T and Brownian path.
    noise = torch.randn(cache['noise'].shape, generator=torch.Generator().manual_seed(seed))
    return dict(gts=cache['gts'][indices], noise=noise[indices],
                conds=[cache['conds'][i] for i in indices],
                hier_conds=tuple([[axis[i] for i in indices] for axis in cache['hier_conds']]))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, vae, lpips_fns, cache, recipe, args, sf, device):
    fm = FlowMatching(num_steps=recipe.get('steps', 50), sampler='heun',
                      shift=recipe.get('shift', 1.), sde_gamma=recipe.get('gamma', 0.),
                      sde_tmin=recipe.get('tmin', 0.), sde_tmax=recipe.get('tmax', 1.),
                      sde_ramp=recipe.get('ramp', 0.))
    lat = sample_latents(model, fm, cache['noise'], cache['conds'], recipe.get('cfg', 1.),
                         args.batch, device, seed=args.current_seed,
                         hier_conds=cache['hier_conds'])
    preds = torch.empty_like(cache['gts'])
    torch.cuda.empty_cache()
    for i in range(len(lat)):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dec = vae.decode(lat[i:i+1, :4].to(device) / sf).sample
        preds[i:i+1] = ((dec.clamp(-1, 1) + 1) / 2).float().cpu()
    del lat, dec
    p_np = preds.numpy().transpose(0, 2, 3, 1)
    gt = (cache['gts'] + 1) / 2
    gt_np = gt.numpy().transpose(0, 2, 3, 1)
    per_image = []
    for i in range(len(preds)):
        p, g = p_np[i], gt_np[i]
        metrics = dict(ssim=_ssim(p, g), mse=_mse(p, g), ink_ssim=ink_ssim(p, g),
                       ink_iou=ink_iou(p, g), frag=frag_ratio(p, g), hole=hole_ratio(p),
                       frag_error=abs(frag_ratio(p, g) - 1),
                       ink_area_ratio=float((p.mean(2) < .5).sum() / max(1, (g.mean(2) < .5).sum())))
        for name, fn in lpips_fns.items():
            metrics['lpips_' + name] = float(fn((preds[i:i+1] * 2 - 1).to(device),
                                                (gt[i:i+1] * 2 - 1).to(device)).mean())
        per_image.append(metrics)
    return per_image, p_np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--recipes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--subset', choices=['tune', 'holdout', 'all'], default='tune')
    ap.add_argument('--csv', default='exp-std/csv/eval200_fixed.csv')
    ap.add_argument('--vae', default='experiments/calli_vae_physics/calli_vae_final')
    ap.add_argument('--vae-shift', type=float, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--lpips', nargs='+', choices=['alex', 'vgg'], default=['alex', 'vgg'])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    device = torch.device('cuda')
    if not (Path(args.vae) / 'config.json').exists():
        raise FileNotFoundError(args.vae)  # never silently fall back to SD VAE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recipes = json.loads(Path(args.recipes).read_text(encoding='utf-8'))
    if args.only:
        recipes = [r for r in recipes if r['name'] in args.only]
        if set(args.only) != {r['name'] for r in recipes}:
            raise ValueError('Unknown recipe in --only')
    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))
    splits = split_indices(rows)
    indices = splits[args.subset]
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False, mmap=True)
    raw_args = ckpt['args']
    a = AttrDict(vars(raw_args) if hasattr(raw_args, '__dict__') else raw_args)
    if a.get('use_ot') or a.get('use_c2ot'):
        raise ValueError('This corrected SDE score identity requires independent Gaussian pairing')
    if a.get('skel_as_glyph_cond') or a.get('use_glyph_cond') or a.get('w_glyph_cond'):
        raise ValueError('This runner currently supports no-skeleton v80 checkpoints only')
    for key in ['triple_table_prefix', 'callig_remap_json', 'char_remap_json', 'font_remap_json']:
        if a.get(key) and not Path(a[key]).exists():
            raise FileNotFoundError(f'{key}: {a[key]}')
    model = build_model_from_args(a, device)
    apply_post_construction(model, a, verbose=False)
    state = ckpt.get('ema') or ckpt.get('model')
    if state is None:
        raise ValueError('Missing EMA/model state')
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in state.items()}, strict=True)
    model.eval()
    del ckpt, state, raw_args
    gc.collect()
    sf = float(a['vae_scaling_factor'])
    vae = _get_vae(device, args.vae, args.vae_shift)
    vae.eval()
    import lpips
    lpips_fns = {name: lpips.LPIPS(net=name, verbose=False).to(device).eval() for name in args.lpips}
    cache = make_eval_cache(args.csv, a['img_root'], None, 256, len(rows), 8, 4, sf)
    _triple_remap_cache(cache, args.csv, len(rows), a)
    manifest = dict(checkpoint=str(Path(args.ckpt).resolve()), checkpoint_sha256=sha256(args.ckpt),
                    vae_weights_sha256=sha256(Path(args.vae) / 'diffusion_pytorch_model.safetensors'),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [Path(__file__), ROOT / 'src/loss/flow_matching.py',
                                    ROOT / 'src/eval/in_mem_eval.py', ROOT / 'src/eval/inference.py']},
                    csv=str(Path(args.csv).resolve()), csv_sha256=sha256(args.csv), splits=splits,
                    subset=args.subset, seeds=args.seeds, recipes=recipes, batch=args.batch,
                    vae=str(Path(args.vae).resolve()), vae_scaling_factor=sf,
                    checkpoint_shift=a.get('vae_shift_factor'), decode_shift=args.vae_shift,
                    lpips_backbones=args.lpips, noise_seed='seed regenerates full CSV noise before slicing',
                    torch_version=torch.__version__, gpu=torch.cuda.get_device_name())
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Output directory already belongs to a different experiment; use a new --out')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'READY n={len(indices)} subset={args.subset} sf={sf} decode_shift={args.vae_shift}')
    summary = []
    for seed in args.seeds:
        args.current_seed = seed
        selected = sliced(cache, indices, seed)
        for recipe in recipes:
            name = recipe['name']
            path = out / f'{name}_seed{seed}.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                print(f'SKIP {name} seed={seed}')
            else:
                start = time.time()
                print(f'START {name} seed={seed} {recipe}')
                # Historical decode-only audit overrides the wrapper, never the weights.
                decode_vae = (_get_vae(device, args.vae, recipe['decode_shift'])
                              if 'decode_shift' in recipe else vae)
                metrics, images = evaluate(model, decode_vae, lpips_fns, selected, recipe, args, sf, device)
                result = dict(name=name, seed=seed, recipe=recipe, indices=indices,
                              mean={k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
                              per_image=metrics, seconds=time.time() - start)
                path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
                sample_dir = out / 'samples' / f'{name}_seed{seed}'
                sample_dir.mkdir(parents=True, exist_ok=True)
                for i, image in zip(indices, images):
                    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(sample_dir / f'{i:03d}.png')
                del images
                torch.cuda.empty_cache()
            summary.append(dict(name=name, seed=seed, **result['mean'], seconds=result['seconds']))
            print(f"DONE {name} seed={seed} {json.dumps(result['mean'])} seconds={result['seconds']:.1f}")
            with open(out / 'summary.csv', 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
    print('SWEEP_DONE')


if __name__ == '__main__':
    main()

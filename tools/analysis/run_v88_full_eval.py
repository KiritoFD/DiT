"""Read-only, same-pass style evaluation and fixed-input causal activation probes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    import torch
    from src.eval.model_io import load_model_from_ckpt
    from src.eval.in_mem_eval import _get_cache
    # Older remote config adapters predate writer_glyph. Preserve strict loading;
    # forward the checkpoint's actual branch arguments, never discard its keys.
    import src.model.build as model_build
    original_kwargs = model_build.dit_kwargs
    def complete_kwargs(args, **kwargs):
        result = original_kwargs(args, **kwargs)
        result.setdefault('writer_glyph_rank', getattr(args, 'writer_glyph_rank', 0))
        result.setdefault('writer_glyph_at', getattr(args, 'writer_glyph_at', '3,7'))
        return result
    model_build.dit_kwargs = complete_kwargs

    torch.set_num_threads(4)
    out = Path('assets/v88_full_eval_20261016')
    out.mkdir(parents=True, exist_ok=True)
    v88 = Path('assets/results/v88_writer_glyph_base/20261015-131555-v88-writer-glyph-base')
    base = Path('assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond')
    sets = {'v88_165k': v88 / 'eval_samples_ctrl/step0165000/eval200fix',
            'v88_170k': v88 / 'eval_samples_ctrl/step0170000/eval200fix',
            'v84A_170k': base / 'eval_samples_ctrl/step0170000/eval200fix',
            'v84A_200k': base / 'eval_samples_ctrl/step0200000/eval200fix'}
    for directory in sets.values():
        assert all((directory / f'g{i}.png').exists() for i in range(187)), directory
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1')
    gen = [arg for name, directory in sets.items() for arg in ('--gen', f'{name}={directory}')]
    if '--probe-only' not in sys.argv:
        subprocess.run([sys.executable, 'tools/eval_gen_with_disc.py', '--device', 'cuda',
                    '--root', '.', '--root', 'data', *gen,
                    '--out', str(out / 'independent_writer.json')], check=True, env=env)
    style = [arg for name, directory in sets.items() for arg in ('--set', f'{name}={directory}')]
    if '--probe-only' not in sys.argv:
        subprocess.run([sys.executable, 'tools/v85_compare.py', *style,
                    '--out', str(out / 'style')], check=True, env=env)

    results = []
    trajectories = []
    provenance = []
    for step in (165000, 170000):
        ckpt = v88 / f'checkpoints/{step:07d}.pt'
        sha = hashlib.sha256()
        with ckpt.open('rb') as source:
            for chunk in iter(lambda: source.read(8*1024*1024), b''):
                sha.update(chunk)
        provenance.append({'step': step, 'checkpoint': str(ckpt.resolve()),
                           'sha256': sha.hexdigest(), 'weights': 'ema'})
        model, args = load_model_from_ckpt(str(ckpt), device='cuda', use_ema=True)
        model.eval()
        args.std_soft_root = 'exp-std/data/std_soft_glyphs_v1'
        args.std_soft_eval_sigma = .65
        cache = _get_cache('exp-std/csv/eval200_fixed.csv', 187, None, '', args)
        indices = [0, 23, 62, 82, 8, 31, 75, 88]
        z = cache['noise'][indices].cuda()
        g = cache['skels_latent'][indices].cuda()
        conds = cache['conds']
        yc = torch.tensor([conds[i][0] for i in indices], device='cuda')
        ychar = torch.tensor([conds[i][1] for i in indices], device='cuda')
        h = cache['hier_conds']
        ys = torch.tensor([h[2][i] for i in indices], device='cuda')
        raw = torch.tensor([h[0][i] for i in indices], device='cuda')
        pair = torch.tensor([h[1][i] for i in indices], device='cuda')
        branch = model.writer_glyph
        import inspect
        if 'cache_styled' not in inspect.signature(branch.film.forward).parameters:
            old_film = branch.film.forward
            def compatible_film(*args, cache_styled=True, **kwargs):
                return old_film(*args, **kwargs)
            branch.film.forward = compatible_film
        original = branch.inject
        records = []
        def measured(x, delta, block):
            y = original(x, delta, block)
            rms = lambda a: a.float().square().mean().sqrt().item()
            records.append({'block': block, 'gain': branch.gains[str(block)].item(),
                            'hidden_rms': rms(x), 'residual_rms': rms(y-x),
                            'residual_hidden_ratio': rms(y-x)/max(rms(x), 1e-12)})
            return y
        def predict(t, writer=yc):
            return model(z, torch.full((len(indices),), t*1000, device='cuda'),
                         writer, ychar, g=g, y_script=ys, y_callig_raw=raw, y_pair=pair)
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for t in (.05, .15, .3, .5, .8, 1.):
                    records.clear()
                    branch.inject = measured
                    full = predict(t).float()
                    activation = list(records)
                    branch.inject = lambda x, delta, block: x
                    off = predict(t).float()
                    branch.inject = original
                    swapped = predict(t, (yc + 1) % 10).float()
                    norm = full.square().mean().sqrt().clamp_min(1e-12)
                    results.append({'step': step, 't': t, 'activation': activation,
                        'branch_off_relative_velocity_rms': ((full-off).square().mean().sqrt()/norm).item(),
                        'writer_swap_relative_velocity_rms': ((full-swapped).square().mean().sqrt()/norm).item()})
        finally:
            branch.inject = original
        from src.eval.sampler import sample_latents, build_diffusion
        diffusion = build_diffusion(50, diffusion_type='flow',
                                   flow_kwargs={'sampler': 'heun', 'sde_gamma': 0.})
        sampler_args = dict(conds=[conds[i] for i in indices], cfg_scale=1., batch=4,
                            device=torch.device('cuda'), skel=g.cpu(), seed=0,
                            hier_conds=tuple([[axis[i] for i in indices] for axis in h]))
        # Canonical trajectory, not the off-distribution fixed-noise time sweep above.
        records.clear()
        try:
            branch.inject = measured
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                full_latent = sample_latents(model, diffusion, z.cpu(), **sampler_args)
            trajectory_records = list(records)
            branch.inject = lambda x, delta, block: x
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                off_latent = sample_latents(model, diffusion, z.cpu(), **sampler_args)
        finally:
            branch.inject = original
        trajectories.append({'step': step,
            'relative_final_latent_rms': ((full_latent-off_latent).square().mean().sqrt()/full_latent.square().mean().sqrt()).item(),
            'activation': trajectory_records})
        del model, branch, original
        torch.cuda.empty_cache()
    (out / 'internal_probes.json').write_text(json.dumps({
        'checkpoints': provenance,
        'canonical_trajectory': trajectories,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path(__file__), Path('src/model/dit.py'), Path('src/model/dit_conditioning.py'),
                          Path('src/model/components.py'), Path('src/model/writer_glyph.py')]},
        'indices': indices, 'input': 'fixed seed0 Gaussian noise at all t; NOT on-trajectory states',
        'limitations': 'writer+1 includes unsupported writer/script pairs; sensitivity only, not quality. No image-level causal claims.',
        'results': results}, indent=2))
    print('V88_FULL_EVAL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

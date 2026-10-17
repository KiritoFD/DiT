"""Strict raw U-Net evaluation, including legacy checkpoint identity audit."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import src.eval.in_mem_eval as evaluator
    from src.eval.protocol import sha256_file
    from src.unet.io import ARCHITECTURE, Config, build, eval_args, diffusion

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', nargs='+', type=int, required=True)
    cli = parser.parse_args()
    torch.set_num_threads(4)
    for step in cli.steps:
        path = Path(cli.run) / 'checkpoints' / f'{step:07d}.pt'
        manifest = json.loads(Path(str(path) + '.done').read_text())
        assert manifest['step'] == step and manifest['size'] == path.stat().st_size
        checkpoint_sha = sha256_file(str(path))
        assert checkpoint_sha == manifest['sha256']
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        saved = ck['args']
        assert saved.architecture == ARCHITECTURE
        config = Config(**saved.architecture_config)
        assert ck['train_steps'] == step
        model = build(config).cuda().eval()
        model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck['delta'].items()}, strict=True)
        args = eval_args(config)
        args.eval_use_ema = False
        args.eval_vae_path = config.vae_path  # Explicit decoder AND identity path.
        args.in_mem_eval_poster = False
        directory = Path(cli.out) / f'unet_raw{step}'
        directory.mkdir(parents=True, exist_ok=True)
        provenance = dict(checkpoint=str(path.resolve()), checkpoint_sha256=checkpoint_sha,
                          step=step, batch=config.batch, images_seen=step * config.batch,
                          state='raw', architecture=ARCHITECTURE, config=config.__dict__,
                          strict_load=True, legacy_identity_source='checkpoint.args',
                          evaluator_sha256=sha256_file(evaluator.__file__))
        (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        with torch.no_grad():
            result = evaluator.run_in_mem_eval(model, args, step, 'cuda', results_dir=str(directory),
                                               diffusion_override=diffusion(config))
        samples = directory / f'eval_samples_ctrl/step{step:07d}/eval200fix'
        assert (samples / 'DONE.json').exists(), 'Evaluation identity/completion failed'
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        print('RESULT', step, json.dumps(result), flush=True)
        del model, ck
        gc.collect()


if __name__ == '__main__':
    main()

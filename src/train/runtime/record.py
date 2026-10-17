"""Immutable launch identity and sampled execution evidence, separate from config."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import torch

from src.train.manifest import sha256_file


def weight_hash(module):
    """Hash loaded tensor values once at launch, including buffers and dtype."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}|{value.dtype}|{tuple(value.shape)}\n'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class RunRecord:
    def __init__(self, directory, args, model, ema, diffusion, start_step, *, latent_contract):
        self.args = args
        self.root = Path(directory) / 'records'
        self.root.mkdir(exist_ok=True)
        self.id = uuid.uuid4().hex
        self.events = self.root / f'{self.id}.events.jsonl'
        self.start_step = start_step
        config = dict(vars(args))
        files = {}
        # Existing source manifest covers code, VAE, fonts and cache metadata.
        # Explicit additions correct historical omissions without rehashing huge banks.
        paths = {'source_manifest': Path(directory) / 'source_manifest.json'}
        for key in ('data_csv', 'csv', 'eval_csv', 'std_shape_cache', 'resume', 'ckpt'):
            if config.get(key):
                paths[key] = Path(str(config[key]))
        vae = Path(str(config.get('vae_path', '')))
        if config.get('vae_path') and vae.is_dir():
            for path in sorted(vae.rglob('*')):
                if path.suffix in ('.bin', '.safetensors', '.json'):
                    paths[f'vae/{path.relative_to(vae)}'] = path
        for key, path in paths.items():
            files[key] = {'path': str(path), 'sha256': sha256_file(str(path)),
                          'exists': path.is_file()}
        identity = {
            'schema': 1, 'launch_id': self.id,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'start_step': start_step, 'resolved_config': config, 'files': files,
            'loaded_model_sha256': weight_hash(model),
            'loaded_ema_sha256': weight_hash(ema) if ema is not None else None,
            'diffusion_class': f'{type(diffusion).__module__}.{type(diffusion).__name__}',
            'latent_contract': latent_contract,
            'metrics': {str(p): sha256_file(str(p)) for p in
                        sorted(Path('src/eval').rglob('*.py'))},
            'scope': 'launch identity; execution events sample FM/REPA only; '
                     'legacy auxiliary execution is not certified; cache directory '
                     'hashes in source_manifest cover metadata, not all tensor bytes',
        }
        payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2,
                             default=str).encode()
        self.identity_sha256 = hashlib.sha256(payload).hexdigest()
        with (self.root / f'{self.id}.identity.json').open('xb') as f:
            f.write(payload)
        # Existing checkpoints serialize args, linking their existing SHA manifests
        # to this immutable launch. A resume gets a new launch identity.
        args.run_launch_id = self.id
        args.run_identity_sha256 = self.identity_sha256

    def due(self, step):
        return step == self.start_step or (step + 1) % self.args.log_every == 0

    def observe(self, step, batch, t, fm, repa, repa_fn, *, updated, ema_updated,
                repa_called=False):
        """Called only at logging boundaries, after backward; zero is not failure."""
        ids = batch.get('img_id')
        ids = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else ids
        cache = getattr(repa_fn, 'feature_cache', None)
        hit = None
        if cache is not None and ids is not None:
            import numpy as np
            hit = int(np.isin(ids, cache._ids).sum())
        warmup = getattr(repa_fn, 'warmup_steps', 0)
        weight = float(getattr(repa_fn, 'w_repa', 0))
        if warmup > 0:
            weight *= min(1., step / warmup)
        values = torch.stack([fm.detach().float(), repa.detach().float(),
                              t.min().float(), t.max().float()]).cpu().tolist()
        event = {'step': step, 'identity_sha256': self.identity_sha256,
                 'batch_size': len(t), 'img_ids': ids, 'latent_cached': 'latent' in batch,
                 'fm': values[0], 'fm_requires_grad': fm.requires_grad,
                 'repa_weight_effective': weight, 'repa_weighted': values[1],
                 'repa_called': repa_called,
                 'repa_requires_grad': repa.requires_grad, 'cache_hits': hit,
                 't_min': values[2], 't_max': values[3],
                 'optimizer_updated': updated, 'ema_updated': ema_updated}
        with self.events.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=True) + '\n')

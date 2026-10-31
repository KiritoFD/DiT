"""Write the calibration JSON that `JointDecoderObjective` insists on.

`train.py` cross-checks three hashes before it will start a joint arm:
  source_checkpoint_sha256 == sha(spec['source'])
  train_csv_sha256         == sha(cfg.data_csv)
  vae_frozen_sha256        == JointDecoderObjective.frozen_sha

We derive them from the real K checkpoint and the real training CSV so a stale
calibration fails loudly instead of training the wrong decoder.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, '.')
from tools.style_matrix.runtime.common import sha, save   # noqa: E402

K = 'assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt'
CSV = 'assets/p125_20261030/stage_p2.csv'


def frozen_sha(vae_path):
    """Recompute exactly what JointDecoderObjective computes (non-decoder params)."""
    import hashlib
    import torch
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
    digest = hashlib.sha256()
    for name, value in vae.state_dict().items():
        if not name.startswith('decoder.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=K)
    ap.add_argument('--csv', default=CSV)
    ap.add_argument('--vae', default='pretrained_models/sd-vae-ft-ema')
    ap.add_argument('--out-dir', default='assets/d_20261101')
    a = ap.parse_args()

    vsha = frozen_sha(a.vae)
    out = Path(a.out_dir)
    base = dict(source=a.source, source_checkpoint_sha256=sha(a.source),
                train_csv=a.csv, train_csv_sha256=sha(a.csv),
                vae_path=a.vae, vae_frozen_sha256=vsha)

    # D1: decoder-only, image loss is the only signal -> weight 1.0, no GT anchor.
    save(out / 'calibration_D1.json', dict(base, arm='J4', image_weight=1.0,
                                           reconstruction_weight=0.0))
    # D2: FM + weak image loss, keep the GT-reconstruction anchor on the decoder.
    save(out / 'calibration_D2.json', dict(base, arm='J3', image_weight=0.08,
                                           reconstruction_weight=1.0))
    print(json.dumps(dict(vae_frozen_sha256=vsha, source_sha256=base['source_checkpoint_sha256'],
                          csv_sha256=base['train_csv_sha256']), indent=1))
    print('wrote', out / 'calibration_D1.json', 'and', out / 'calibration_D2.json')


if __name__ == '__main__':
    main()

"""Corrected full-batch B/C/D continuation arms; no training side effects."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from src.train.cli import parse_args
from src.train.strong_style import StrongStyle
import logging

for arm in ('B25', 'B50', 'C', 'D'):
    cfg = json.loads((root/f'src/train/configs/v86_strong_{arm}.json').read_text())
    cfg.update(experiment_name=f'v86-full-fixed-10k-{arm}', results_dir=f'assets/results/v86_full_fixed_10k_{arm}',
               max_steps=10000, epochs=10000, ckpt_every=5000, epoch_steps=5000,
               strong_style_full_batch=True, strong_style_samples=0,
               global_batch_size=112 if arm == 'C' else 224)
    path = root/f'src/train/configs/v86_full_fixed_{arm}.json'
    path.write_text(json.dumps(cfg, indent=2)+'\n')
    args = parse_args(['--config', str(path)])
    # Validate critic identity, fixed quantiles and true dataset mappings before GPU use.
    obj = StrongStyle(args, 'cpu', logging.getLogger('preflight'))
    assert len(obj.critic.id2slot) == 23
    print(arm, 'batch=', args.global_batch_size, 'steps=', args.max_steps,
          'writer_ratio=', args.strong_style_ratio, 'slots=', len(obj.critic.id2slot), flush=True)
    del obj

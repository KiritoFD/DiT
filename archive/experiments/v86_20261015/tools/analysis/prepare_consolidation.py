"""Matched 5k extensions from B25@5k EMA, not an optimizer resume."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = 'assets/results/v86_full_fixed_B25/20261015-071058-v86-full-fixed-B25/checkpoints/0005000.pt'
base = json.loads((ROOT/'src/train/configs/v86_full_fixed_B25.json').read_text())
for name, ratio in [('low15', .15), ('matched80', .8)]:
    cfg = dict(base, experiment_name='v86-B25-5k-' + name,
               results_dir='assets/results/v86_consolidate_' + name,
               warm_start_from=source, max_steps=5000, epochs=5000,
               epoch_steps=5000, ckpt_every=5000, lr=1e-5, warmup_steps=0,
               lr_schedule='cosine', min_lr_ratio=1., strong_style_ratio=ratio)
    cfg['_comment'] = ('Parent B25@5k EMA; fresh optimizer on BOTH arms; identical constant LR1e-5; '
                       'local step5000 equals lineage step10000. Parent5k retained separately.')
    (ROOT/f'src/train/configs/v86_consolidate_{name}.json').write_text(json.dumps(cfg, indent=2)+'\n')
    print(name, 'parent=', source, 'additional_steps=5000', 'writer_ratio=', ratio)

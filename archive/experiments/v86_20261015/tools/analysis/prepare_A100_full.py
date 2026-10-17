import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
config = json.loads((root/'src/train/configs/v86_strong_A100.json').read_text())
config.update(experiment_name='v86-A100-full-fixed', results_dir='assets/results/v86_A100_full_fixed',
              max_steps=10000, epochs=10000, ckpt_every=5000, epoch_steps=5000,
              strong_style_full_batch=True, strong_style_samples=0, global_batch_size=224)
(root/'src/train/configs/v86_A100_full_fixed.json').write_text(json.dumps(config, indent=2)+'\n')

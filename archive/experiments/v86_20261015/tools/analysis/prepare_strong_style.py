"""Create independent, matched warm-start arms. Does not launch training."""
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src.train.cli import parse_args
    base = json.loads((root / 'src/train/configs/v86A_writer_frozen_20k.json').read_text())
    base.update(max_steps=10000, epochs=10000, epoch_steps=5000, ckpt_every=5000,
                finetune_callig_table=False, probe_style_grad_steps=0,
                strong_style_samples=0, strong_style_cf_every=4, strong_style_probe_every=10,
                strong_style_max_weight=5., strong_style_cf_ratio=.6, strong_style_brush_ratio=.1)
    arms = [('control', 'off', .0, .25), ('A30', 'A', .3, .25),
            ('A60', 'A', .6, .25), ('A100', 'A', 1., .25),
            ('B25', 'B', .8, .25), ('B50', 'B', .8, .5),
            ('C', 'C', 1.2, .25), ('D', 'D', .8, .25)]
    for name, mode, ratio, quantile in arms:
        steps = 10000
        config = dict(base, experiment_name='v86-strong-' + name,
                      max_steps=steps, epochs=steps,
                      results_dir='assets/results/v86_strong_' + name,
                      strong_style_mode=mode, strong_style_ratio=ratio,
                      strong_style_quantile=quantile,
                      strong_style_calibration='assets/strong_style_real_calibration.json')
        path = root / ('src/train/configs/v86_strong_' + name + '.json')
        path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        parsed = parse_args(['--config', str(path)])
        assert parsed.max_steps == steps and parsed.strong_style_mode == mode
        print(path, 'steps=', parsed.max_steps, 'mode=', parsed.strong_style_mode)


if __name__ == '__main__':
    main()

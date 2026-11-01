"""T0/T1/T2 driver: three 5k continuations of the same P2 EMA.

The arms differ ONLY in the mechanism that touches the Writer condition:

    T0  P2 recipe continued unchanged                               (control)
    T1  identical loss, Writer x Script strata sampled uniformly over the
        strata that actually exist. Real counts run 30 .. 2379 images, so the
        weakest Writers are under-exposed by ~80x, which is the cheapest
        explanation for "Writer style is not learned" and the cheapest to test.
    T2  FM + Writer-contrast ranking at t >= 0.6 (see t2_rank.py): for a real
        sample the true Writer must explain the velocity better than a wrong
        Writer that genuinely has real GT of the same script.

Data, total exposure, optimizer, LR schedule and EMA are identical. Checkpoints
every 2.5k, all kept.

T2 injection is a single explicit subclass of TrainState. The alternative was
patching src/train/trainer.py; this keeps the shared trainer byte-identical so
T0/T1 cannot diverge from it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/root/Workspace/xy/DiT')
sys.path.insert(0, str(ROOT))

P2_START = 'assets/u100_p1p2_20261025/p2_runs/20261025-152857-Unet-flow/checkpoints/0020000.pt'


def make_config(arm: str, steps: int, out_root: Path, base: str,
                batch: int | None = None) -> Path:
    cfg = json.loads((ROOT / base).read_text(encoding='utf-8'))
    out = out_root / arm
    out.mkdir(parents=True, exist_ok=True)
    cfg.update(
        output=str(out),
        max_steps=steps, stop_at_step=steps, lr_decay_steps=steps,
        checkpoint_steps=[], ckpt_every=steps // 2, eval_every=steps // 2,  # trainer requires equal intervals
        log_every=50, eval_before_training=False,
        initialize_from=str(ROOT / P2_START), initialize_state='ema',
    )
    # T2 runs a SECOND forward (wrong-writer branch) on the active sub-batch, so
    # the P2 batch of 512 no longer fits: it OOMed with 76 MiB to spare. Callers
    # lower the batch for T2 only; T0/T1 keep the P2 value so their numbers stay
    # comparable to each other.
    if batch is not None:
        cfg['batch'] = int(batch)
    if arm == 'T1':
        cfg['study_spec'] = str(ROOT / 'assets/style_24h_20261026/T1_study.json')
    path = out / 'config.json'
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def write_t1_spec(csv_path: Path) -> Path:
    """Visible-then-build record: the strata and their real counts, then the spec."""
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))
    counts = Counter((r['calligrapher'], r['script']) for r in rows)
    rec = {
        'sampling': 'writer_script',
        'rationale': (
            'Uniform sampling over Writer x Script strata that ACTUALLY EXIST in '
            'the training CSV. No invented combinations: a stratum with zero real '
            'images is never sampled. pair_fraction stays at the P2 value so the '
            'same-character contrastive share is unchanged; only the marginal '
            'stratum distribution is flattened.'),
        'strata': {f'{w}|{s}': n for (w, s), n in sorted(counts.items())},
        'n_strata': len(counts),
        'count_min': min(counts.values()),
        'count_max': max(counts.values()),
        'exposure_ratio_max_over_min': max(counts.values()) / min(counts.values()),
        'n_rows': len(rows),
    }
    out = ROOT / 'assets/style_24h_20261026/T1_study.json'
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[T1] spec -> {out}  strata={len(counts)} '
          f'exposure spread={rec["exposure_ratio_max_over_min"]:.1f}x', flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['T0', 'T1', 'T2'])
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--base', default='assets/u100_p1p2_20261025/p2.json')
    ap.add_argument('--rank-weight', type=float, default=0.1)
    ap.add_argument('--rank-tau', type=float, default=0.5)
    ap.add_argument('--t-min', type=float, default=0.6)
    ap.add_argument('--out-root', default='assets/style_24h_20261026/ablations')
    ap.add_argument('--batch', type=int, default=None,
                    help='override the training batch (P2 used 512; T2 needs less '
                         'because it runs a second forward for the wrong writer)')
    a = ap.parse_args()

    out_root = ROOT / a.out_root
    if a.arm == 'T1':
        write_t1_spec(ROOT / 'assets/u100_p1p2_20261025/p2_train.csv')
    cfg_path = make_config(a.arm, a.steps, out_root, a.base, batch=a.batch)
    print(f'[{a.arm}] config -> {cfg_path}', flush=True)
    sys.argv = ['train', '--config', str(cfg_path)]

    if a.arm == 'T2':
        from src.train import trainer as trainer_mod
        from src.train.runtime import mainline as mainline_mod

        class WriterRankTrainState(trainer_mod.TrainState):
            """Adds the ranking term; the FM loss itself is untouched."""

            def __init__(self, **kw):
                from src.unet.io import Config
                from tools.style_24h4090.t2_rank import WriterRankAdapter
                cfg = Config(**json.loads(cfg_path.read_text(encoding='utf-8')))
                adapter = WriterRankAdapter(cfg, weight=a.rank_weight, t_min=a.t_min, tau=a.rank_tau,
                                            logger=kw.get('logger'))
                adapter.model = kw['model']
                kw['objective_adapter'] = adapter
                super().__init__(**kw)
                self.objective_adapter = adapter

        trainer_mod.TrainState = WriterRankTrainState

        # The runtime constructs the objective bound to the (compiled) model; give
        # the adapter the same handle so T2 scores the branch the trainer trains.
        Original = mainline_mod.LatentStdObjective

        class BoundRuntime(Original):
            def __init__(self, state):
                super().__init__(state)
                if self.adapter is not None:
                    self.adapter.model = getattr(self, '_model', None) or state.model

            def __call__(self, model, z, t, kwargs, batch, image, step):
                import torch
                raw = getattr(model, '_orig_mod', model)
                all_drop = torch.rand(len(z), device=z.device) < raw.drop_all
                wd = all_drop | (torch.rand(len(z), device=z.device) < raw.drop_writer)
                sd = all_drop | (torch.rand(len(z), device=z.device) < raw.drop_script)
                kwargs['y_callig'] = torch.where(wd, raw.num_writers, kwargs['y_callig'])
                kwargs['y_script'] = torch.where(sd, raw.num_scripts, kwargs['y_script'])
                kwargs['condition_dropped'] = True
                self.adapter.model = model
                return super().__call__(model,z,t,kwargs,batch,image,step)

        mainline_mod.LatentStdObjective = BoundRuntime

    from src.unet.train import main as train_main
    return train_main()


if __name__ == '__main__':
    raise SystemExit(main() or 0)

"""Opt-in same-character sampler and fixed-input conditional velocity ranking."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Sampler

from src.train.gradient_budget import GradientBudget, format_metrics


class PairMixtureSampler(Sampler):
    """First 25% of each batch are adjacent real A/B pairs; remainder ordinary rows.

    Select script with its original augmented-row frequency, then a legal cell
    uniformly WITHIN script. Do not upweight scarce writer pairs to uniform mass.
    Paired endpoints are non-augmented GT; ordinary rows retain augmentation.
    """
    def __init__(self, dataset, batch_size, steps, seed=0, fraction=.25):
        self.rows = dataset.samples
        self.batch_size, self.steps, self.seed, self.epoch = batch_size, steps, seed, 0
        self.n_pair = 2*int(batch_size*fraction/2)
        if self.n_pair < 2 or self.n_pair > batch_size:
            raise ValueError('Invalid pair fraction')
        groups = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(self.rows):
            if not (r.get('aug') or '').strip():
                groups[r['script'], r['character']][r['calligrapher']].append(i)
        self.cells = defaultdict(list)
        for (s, c), writers in sorted(groups.items()):
            keys = sorted(writers)
            for j, a in enumerate(keys):
                for b in keys[j+1:]:
                    self.cells[s].append((writers[a], writers[b]))
        self.scripts = sorted(self.cells)
        if not self.scripts:
            raise ValueError('No legal same-script same-character pairs')
        count = {s: sum(r['script']==s for r in self.rows) for s in self.scripts}
        if sum(count.values()) != len(self.rows):
            raise ValueError('A script has no pair coverage; cannot preserve script mixture')
        self.script_p = np.array([count[s]/len(self.rows) for s in self.scripts])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps*self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch*1000003)
        for _ in range(self.steps):
            ids = []
            for script in rng.choice(self.scripts, self.n_pair//2, p=self.script_p):
                aa, bb = self.cells[script][int(rng.integers(len(self.cells[script])))]
                ids.extend([int(rng.choice(aa)), int(rng.choice(bb))])
            ids.extend(rng.integers(len(self.rows), size=self.batch_size-self.n_pair).tolist())
            yield from ids


def rank_velocity_loss(correct, wrong, target, margin=.01, tau=.1):
    ell_true = (correct.float()-target.float()).flatten(1).square().mean(1)
    ell_wrong = (wrong.float()-target.float()).flatten(1).square().mean(1)
    return F.softplus((margin+ell_true-ell_wrong)/tau).mean(), ell_true, ell_wrong


class ConditionalRank:
    def __init__(self, args, device, logger):
        if args.diffusion_type != 'flow' or args.pair_fraction <= 0:
            raise ValueError('Conditional ranking requires Flow and paired sampler')
        if args.w_critic_writer or args.w_critic_away or args.strong_style_mode != 'off':
            raise ValueError('Rank-FM cannot combine Critic/strong-style')
        self.args, self.logger = args, logger
        self.scale, self.weak = {}, {}
        self.metadata = {}
        import csv
        with open(args.data_csv, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                self.metadata[int(r['img_id'])] = r

    def __call__(self, model, x0, t, kwargs, batch, fm, step, fm_norm=None):
        a = self.args
        n = 2*int(len(x0)*a.pair_fraction/2)
        ids = batch['img_id'][:n].tolist()
        for i in range(0, n, 2):
            u, v = self.metadata[int(ids[i])], self.metadata[int(ids[i+1])]
            if (u['script'], u['character']) != (v['script'], v['character']) or u['calligrapher']==v['calligrapher']:
                raise ValueError('Invalid same-character same-script writer pair')
        device = x0.device
        x0 = x0[:n]
        # Restore every module's mode and RNG: A/B main FM has identical RNG streams.
        modes = [(m, m.training) for m in model.modules()]
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if x0.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(a.global_seed+700000+step)
            tt = a.rank_t_min+(a.rank_t_max-a.rank_t_min)*torch.rand(n, device=device)
            eps = torch.randn_like(x0)
            xt = (1-tt[:, None, None, None])*x0+tt[:, None, None, None]*eps
            target = eps-x0
            flip = torch.arange(n, device=device)^1
            g = batch['std_soft'][:n].to(device).float()
            if g.ndim == 3:
                g = g[:, None]
            # std-soft dataset is uint8, unlike evaluation's floating masks.
            if batch['std_soft'].dtype == torch.uint8:
                g = g/255.
            good = {k: v[:n] for k, v in kwargs.items()
                    if isinstance(v, torch.Tensor) and k not in ('g', 'cond')}
            good.update(y_callig=batch['y_callig_raw'][:n].to(device),
                        y_callig_raw=batch['y_callig_raw'][:n].to(device),
                        y_pair=batch['y_pair'][:n].to(device),
                        y_script=batch['y_script'][:n].to(device), g=g)
            bad = dict(good)
            for key in ('y_callig', 'y_callig_raw', 'y_pair'):
                bad[key] = good[key][flip]
            model.eval()
            try:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=x0.is_cuda):
                    va = model(xt, tt*1000., **good)
                    vb = model(xt, tt*1000., **bad)
                    if isinstance(va, tuple): va = va[0]
                    if isinstance(vb, tuple): vb = vb[0]
                    loss, ell_true, ell_wrong = rank_velocity_loss(va[:, :4], vb[:, :4], target, a.rank_margin, a.rank_tau)
            finally:
                for module, mode in modes:
                    module.training = mode
        value, metrics = GradientBudget(a.strong_style_max_weight, self.scale, self.weak, self.logger).balance(
            {'conditional_rank': (loss, a.rank_gradient_ratio)},
            [p for p in model.parameters() if p.requires_grad], fm_norm, step)
        if fm_norm is not None or step % 50 == 0:
            self.logger.info('[rank-fm] step=%d n=%d true=%.6g wrong=%.6g gap=%.6g %s',
                step, n, float(ell_true.mean()), float(ell_wrong.mean()),
                float((ell_wrong-ell_true).mean()), format_metrics(metrics))
        return value

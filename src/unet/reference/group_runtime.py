"""Group-id style runtime for the writer_slot route.

Differences from the legacy ReferenceTraining: writer labels come from the
pipeline's own calligrapher_id column (Top10 rows 0-9, generic groups 10+),
the bank may merge the pool feature npz with the Top10 feature npz so that
eval/FT targets always find same-group cross-character references, and the
evaluation provider reads the same column instead of style_writer_*.
"""
import json
import logging
from pathlib import Path
import numpy as np
import torch

from .data import ReferenceBank
from .encoder import supcon


class MergedBank:
    def __init__(self, *paths):
        data = [np.load(p, allow_pickle=False) for p in paths]
        ids = np.concatenate([d['ids'] for d in data])
        if len(np.unique(ids)) != len(ids): raise ValueError('merged bank has duplicate ids')
        self.ids = ids
        self.features = np.concatenate([d['features'] for d in data])
        self.writers = np.concatenate([d['writers'] for d in data])
        self.chars = np.concatenate([d['chars'] for d in data])
        self.scripts = np.concatenate([d['scripts'] for d in data])
        self.lookup = {int(v): i for i, v in enumerate(self.ids)}
        pools = {}
        for i, key in enumerate(zip(self.writers, self.scripts)):
            pools.setdefault(key, []).append(i)
        self.pools = {key: np.asarray(v) for key, v in pools.items()}

    positions = ReferenceBank.positions
    sample = ReferenceBank.sample


def read_script_map(tables_dir):
    return {int(k): v for k, v in json.loads(
        (Path(tables_dir) / 'font_remap.json').read_text(encoding='utf-8')).items()}


class StyleTraining:
    """Condition transform + SupCon auxiliary over the merged group bank."""

    def __init__(self, config, base, top10_cache=''):
        caches = [config.reference_cache] + ([top10_cache] if top10_cache else [])
        self.bank = MergedBank(*caches)
        self.top10_cache = str(top10_cache)
        self.base = base; self.config = config
        self.stats = {}
        self.step = 0

    def __call__(self, model, batch, kwargs, x0, step):
        self.step = step
        self.base(model, batch, kwargs, x0, step)
        positions = self.bank.positions(batch['img_id'].cpu().numpy())
        rng = np.random.default_rng(self.config.seed + step)
        refs, mask = self.bank.sample(positions, rng, self.config.reference_max_refs)
        kwargs['reference_features'] = torch.as_tensor(self.bank.features[refs], device=x0.device)
        kwargs['reference_mask'] = torch.as_tensor(mask, device=x0.device)

    def loss(self, model, batch):
        raw = getattr(model, '_orig_mod', model)
        p = self.bank.positions(batch['img_id'].cpu().numpy())
        device = next(raw.parameters()).device
        feature = raw.reference_head(torch.as_tensor(self.bank.features[p], device=device))
        values = [torch.as_tensor(field[p], device=device) for field in
                  (self.bank.writers, self.bank.chars, self.bank.scripts)]
        loss, stats = supcon(feature, *values, temperature=self.config.reference_temperature,
                             hard_weight=self.config.reference_hard_weight)
        self.stats = dict(loss=loss.detach(), **stats)
        if self.step % self.config.log_every == 0:
            logging.getLogger('unet').info('STYLE step=%d supcon=%.6f weight=%g valid=%d hard=%d',
                                           self.step, float(loss.detach()), self.config.reference_con_weight,
                                           int(stats['valid_anchors']), int(stats['hard_negatives']))
        return self.config.reference_con_weight * loss


def bind_group_evaluation(model, config, bank):
    """Fixed training-only references per target from the merged bank."""
    stored = {}
    script_map = read_script_map(config.condition_tables_root)

    def provide(rows):
        key = tuple(int(row['img_id']) for row in rows)
        if key not in stored:
            ids = []; masks = []
            for row in rows:
                group = int(row['calligrapher_id']); script = script_map[int(row['script_id'])]
                pool = bank.pools.get((group, script))
                if pool is None: raise ValueError(f'no reference pool for group {group}')
                pool = pool[bank.chars[pool] != ord(row['character'])]
                if not len(pool): raise ValueError('no cross-character reference for eval target')
                rng = np.random.default_rng(config.seed + int(row['img_id']))
                chosen = rng.permutation(pool)
                _, first = np.unique(bank.chars[chosen], return_index=True)
                chosen = chosen[np.sort(first)][:config.reference_max_refs]
                ids.append(np.pad(chosen, (0, config.reference_max_refs - len(chosen)), mode='edge'))
                masks.append(np.arange(config.reference_max_refs) < len(chosen))
            stored[key] = dict(reference_features=torch.from_numpy(bank.features[np.asarray(ids)]),
                               reference_mask=torch.tensor(np.asarray(masks)))
        return stored[key]
    model.reference_eval_provider = provide

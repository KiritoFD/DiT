"""T2: Writer-contrast FM ranking, implemented as an objective adapter.

No external style encoder is involved -- that is the whole point, since every
external judge we audited (ResNet CE, Pixel D, Latent D) turned out to be
fitted to the source pool.

For a real latent (x, w, c, s) the FM error under the TRUE writer must be
smaller than under a DIFFERENT writer that genuinely has real GT of the same
script (same character when available):

    l_w  = ||v_theta(z_t, g, w ) - v*||^2
    l_w' = ||v_theta(z_t, g, w') - v*||^2
    L_rank = softplus((l_w - l_w') / tau)

Only t >= 0.6 is used: G1 showed the Writer effect is significant at 0.55,
0.75, 0.90 and null at 0.15/0.35, so the ranking signal is asked for exactly
where the Writer condition already matters.

Guard against the obvious failure mode (satisfying the ranking by making the
wrong-writer branch worse instead of the right-writer branch better): the FM
loss on the CORRECT condition is returned as the main term and both branches
are logged every log step, so a margin that grows while fm_wrong rises is
visible rather than hidden.

`needs_prediction=False`: the adapter compares the FM error of the TRUE writer
(already computed by the trainer as `result['loss']`) against the FM error of a
WRONG writer on the same target. That margin is what the ranking needs.

Two earlier detours are recorded so they are not repeated:
  * a cached `self._v_target` that was never populated -> AttributeError at ~step 20;
  * `needs_prediction=True` to recover `pred_xstart` -> OOM at batch 512, because
    that key keeps the whole forward graph alive (flow_matching warns 20G -> 22G).
The fix is one detached `terms["v_target"]` in flow_matching.py: the adapter then
scores the wrong-writer branch against the very same target at no VRAM cost.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/root/Workspace/xy/DiT')


class WriterRankAdapter:
    needs_prediction = False

    def __init__(self, config, *, weight=0.1, t_min=0.6, tau=0.5,
                 char_match=True, logger=None):
        self.config = config
        self.weight = float(weight)
        self.t_min = float(t_min)
        self.tau = float(tau)
        self.char_match = bool(char_match)
        self.logger = logger
        self.null = int(config.num_writers)

        callig = json.loads((ROOT / config.callig_remap_json).read_text())
        font = json.loads((ROOT / config.font_remap_json).read_text())
        self.row_of = {int(k): int(v) for k, v in callig.items()}
        self.scr_of = {int(k): int(v) for k, v in font.items()}

        self.by_char, self.by_script, self.char_of = {}, {}, {}
        for r in csv.DictReader(open(ROOT / config.data_csv, encoding='utf-8-sig')):
            row = self.row_of[int(r['calligrapher_id'])]
            sid = self.scr_of[int(r['script_id'])]
            self.char_of[int(r['img_id'])] = (r['character'], sid)
            self.by_char.setdefault((sid, r['character']), set()).add(row)
            self.by_script.setdefault(sid, set()).add(row)
        self.rng = np.random.default_rng(config.seed)
        # Bound by the driver before the loop: the compiled model and the exact
        # interpolant used for this step (set per call, not stored globally).
        self.model = None

    def _partner(self, writer_row, char, sid):
        pool = (self.by_char.get((sid, char)) if self.char_match else None) \
            or self.by_script.get(sid) or set()
        cands = sorted(p for p in pool if p != writer_row and p != self.null)
        if not cands:
            cands = [p for p in range(self.null) if p != writer_row]
        return cands[int(self.rng.integers(len(cands)))]

    def objective(self, result, z, t, kwargs, batch, step):
        """Replaces result['loss'].mean(); returns a differentiable scalar."""
        per_sample = result['loss']                        # [N], already reduced
        base = per_sample.mean()
        # LatentStdObjective forwards only (result, z, t, kwargs, batch, step);
        # the model reaches us through the runtime around this adapter.
        self.model = getattr(getattr(self, 'runtime', None), 'model', None) or self.model

        active = (t >= self.t_min) & (kwargs['y_callig'] != self.null) & (kwargs['y_script'] != self.config.num_scripts)
        ids = active.nonzero().flatten()
        if len(ids) == 0 or self.model is None:
            return base

        ids_cpu = ids.detach().cpu().tolist()
        ws = kwargs['y_callig'].detach().cpu().tolist()
        img_ids = batch['img_id'].tolist()
        wrong = [self._partner(ws[i], *self.char_of[int(img_ids[i])]) for i in ids_cpu]
        wrong = torch.tensor(wrong, device=z.device, dtype=torch.long)

        sub = {'g': kwargs['g'][ids], 'y_script': kwargs['y_script'][ids],
               'condition_dropped': True}
        if kwargs.get('drop_path_scale') is not None:
            sub['drop_path_scale'] = kwargs['drop_path_scale']
        # The SAME regression target the trainer used, exposed detached so the
        # wrong-writer branch is scored on exactly the same scale.
        target = result.get('v_target')
        if target is None:
            raise RuntimeError('T2 requires the exact detached FM target')
        target = target[ids]
        # v*=eps-z0, hence the EXACT interpolant is z0+t*v*. The shared
        # trainer feeds model time in [0,1000], not [0,1]. Old T2 violated both.
        xt = z[ids] + t[ids, None, None, None] * target
        # Wrong condition is a reference score, never an optimization target:
        # do not satisfy ranking by deliberately degrading that prediction.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = self.model(xt, t[ids] * 1000., wrong, None, **sub)
        wrong_err = ((pred.float() - target.float()) ** 2).mean(dim=(1, 2, 3))

        lw = per_sample[ids].float()
        rank = F.softplus((lw - wrong_err) / self.tau).mean()
        if not torch.isfinite(rank) or not rank.requires_grad:
            raise RuntimeError('T2 ranking must be finite and differentiable')

        if step % 50 == 0 or step < 5:
            if self.logger:
                self.logger.info(
                    'T2_RANK step=%d n_active=%d L_rank=%.6f fm_true=%.6f fm_wrong=%.6f '
                    'margin=%.6f weight=%.4g',
                    step, len(ids_cpu), float(rank.detach()), float(lw.mean()),
                    float(wrong_err.mean()), float((wrong_err - lw).mean()), self.weight)
        return base + rank * self.weight

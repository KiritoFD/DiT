"""Opt-in latent post-training; normal FM is always retained by the caller."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F


from src.loss.writer_objectives import real_mask, writer_margin, calibrated_hinge, cf_loss
from src.train.gradient_budget import GradientBudget, norm


def build_strong_style(args, device, logger):
    """Keep disabled post-training free of critic/table/checkpoint side effects."""
    mode = getattr(args, 'strong_style_mode', 'off')
    if mode not in ('off', 'A', 'B', 'C', 'D'):
        raise ValueError("Invalid strong_style_mode; use 'off' to disable, or A/B/C/D")
    return None if mode == 'off' else StrongStyle(args, device, logger)


class StrongStyle:
    def __init__(self, args, device, logger):
        from src.loss.critic_loss import CriticLoss
        self.args, self.logger = args, logger
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError('parameter-gradient calibration currently requires single-rank training')
        if args.diffusion_type != 'flow' or args.condition_fusion != 'factorized_cat':
            raise ValueError('strong-style currently supports factorized flow only')
        if args.w_critic_writer or args.w_critic_away or args.finetune_callig_table:
            raise ValueError('strong-style must not overlap legacy critic/table interventions')
        if args.strong_style_cf_every < 1 or args.strong_style_samples < 0:
            raise ValueError('invalid style sampling settings')
        if args.strong_style_probe_every < 1 or args.strong_style_max_weight <= 0:
            raise ValueError('invalid gradient calibration settings')
        if any(v < 0 for v in (args.strong_style_ratio, args.strong_style_cf_ratio,
                                args.strong_style_brush_ratio)):
            raise ValueError('gradient ratios must be nonnegative')
        if getattr(args, 'resume_full', ''):
            raise ValueError('strong-style arms currently require fresh optimizer warm-start, not resume_full')
        self.critic = CriticLoss(args.critic_ckpt, device)
        self.critic.build_id2slot(args.data_csv, args.callig_remap_json, args.font_remap_json)
        self.mask = real_mask(self.critic.slots, device)
        self.inverse = {self.critic.slots.index(slot): (w, s)
                        for (w, s), slot in self.critic.id2slot.items()}
        self.scale = {}
        self.targets = None
        # ★ 2026-10-14 [重构 P1-6/PR4] **消除逐样本 CUDA 同步** ✗
        #   旧实现每步做:
        #     for i,(w,s) in enumerate(zip(y_callig_raw.tolist(), y_script.tolist())):
        #         ... args.critic_t_min <= float(t[i]) <= args.critic_t_max   # ✗ 逐元素同步
        #   ⇒ 一个 batch 里 B 次 `.item()` 级同步 ✗（专家 §7.3 P0: "消除 … 逐样本 CUDA 同步" ✓）
        #   新实现: 用**向量化掩码**一次算完 ✓ ⇒ 只有**一次**批量 `.tolist()` ✓
        #   ⚠ 语义必须**逐元素等价** ✓（含顺序 ✓ 边界 ✓ 无 RNG ✓）—— 见
        #     tests/test_refactor_pr4.py 的等价性测试 ✓
        self._slot_lut = self._build_slot_lut()
        self._mask_row_sum = self.mask.reshape(self.mask.shape[0], -1).sum(dim=1)
        # g 的量纲判定（0-1 还是 0-255）是**数据属性** ✓ 只需判一次 ✓（旧实现每步 `g.max()` ✗）
        self._g_scale = None
        if args.strong_style_mode in ('B', 'C', 'D'):
            artifact = json.loads(Path(args.strong_style_calibration).read_text(encoding='utf-8'))
            digest = hashlib.sha256(Path(args.critic_ckpt).read_bytes()).hexdigest()
            if artifact['critic_sha256'] != digest or artifact['slots'] != self.critic.slots:
                raise ValueError('calibration critic identity mismatch')
            key = str(args.strong_style_quantile)
            self.targets = torch.tensor([artifact['writer_targets'][key][slot]
                if not slot.startswith('std·') else 0 for slot in self.critic.slots], device=device)
            self.brush_targets = torch.tensor([artifact['brush_targets'][key].get(slot, 0)
                                               for slot in self.critic.slots], device=device)
            if not torch.isfinite(self.targets).all() or not torch.isfinite(self.brush_targets).all():
                raise ValueError('nonfinite calibration targets')
        logger.info('[strong-style] mode=%s writer_ratio=%.3f cf_ratio=%.3f brush_ratio=%.3f samples=%d',
                    args.strong_style_mode, args.strong_style_ratio, args.strong_style_cf_ratio,
                    args.strong_style_brush_ratio, args.strong_style_samples)

    # ─────────────── P1-6/PR4: 向量化辅助（无逐样本同步 ✗）───────────────
    def _build_slot_lut(self):
        """(raw_callig, script) -> slot 下标 的**稠密查表** ✓（-1 = 无映射 ✓）。

        由 `critic.id2slot` 一次性构建 ✓（稀疏 dict ✗ → 稠密张量 ✓ 便于向量化 ✓）。
        """
        pairs = list(self.critic.id2slot.keys())
        if not pairs:
            return torch.zeros((1, 1), dtype=torch.long)
        max_w = max(int(w) for w, _ in pairs) + 1
        max_s = max(int(s) for _, s in pairs) + 1
        lut = torch.full((max_w, max_s), -1, dtype=torch.long)
        for (w, s), slot in self.critic.id2slot.items():
            lut[int(w), int(s)] = self.critic.slots.index(slot)
        return lut

    def _ensure_luts(self):
        """惰性建查表 ✓ —— 兼容 `__new__` 构造的实例（如测试夹具 ✓ 它们绕过 `__init__` ✗）。"""
        if getattr(self, "_slot_lut", None) is None:
            self._slot_lut = self._build_slot_lut()
            self._mask_row_sum = self.mask.reshape(self.mask.shape[0], -1).sum(dim=1)
        return self._slot_lut

    def _eligible_rows(self, batch, t):
        """**向量化**的资格筛选 ✓ —— 与旧的 Python 循环**逐元素等价** ✓（含顺序 ✓ 无 RNG ✓）。

        旧语义（逐字保留 ✓）:
            for i,(w,s) in enumerate(zip(y_callig_raw.tolist(), y_script.tolist())):
                slot = id2slot.get((w,s))
                if slot is not None and (full_batch or t_min <= float(t[i]) <= t_max):
                    yi = slots.index(slot)
                    if int(mask[yi].sum()) >= 2: 收 (i,w,s,yi)
        """
        args = self.args
        w_all = batch['y_callig_raw'].reshape(-1).long()
        s_all = batch['y_script'].reshape(-1).long()
        dev = w_all.device
        lut = self._ensure_luts().to(dev)
        in_range = (w_all >= 0) & (w_all < lut.shape[0]) & (s_all >= 0) & (s_all < lut.shape[1])
        slots = torch.full_like(w_all, -1)
        if bool(in_range.any()):
            slots[in_range] = lut[w_all[in_range], s_all[in_range]]
        ok = slots >= 0
        if not getattr(args, 'strong_style_full_batch', False):
            # ⚠ 必须转 **float64** 再比 ✗ —— 旧代码 `float(t[i]) <= args.critic_t_max`
            #   是 **float64** 比较 ✓；若直接用 float32 张量比标量, torch 会把标量
            #   提升为 float32 ✗ ⇒ 边界值上**语义不同** ✗✗
            #   实测: t=0.8 存成 float32 = 0.800000011920929 ✗
            #     float64 比较: 0.8000000119 <= 0.8  ⇒ False（**排除** ✓）
            #     float32 比较: 0.8000000119 <= 0.8000000119 ⇒ True（**包含** ✗ 不一致 ✗）
            #   （此差异由 tests/test_refactor_pr4.py 的边界测试抓到 ✓）
            t_flat = t.reshape(-1).to(dev).double()
            ok = ok & (t_flat >= float(args.critic_t_min)) & (t_flat <= float(args.critic_t_max))
        row_sum = self._mask_row_sum.to(dev)
        ok = ok & (row_sum[slots.clamp(min=0)] >= 2)
        # ★ 唯一一次批量同步 ✓（旧实现是 B 次逐元素 ✗）
        keep = ok.nonzero(as_tuple=False).reshape(-1).tolist()
        wl = w_all.tolist()
        sl = s_all.tolist()
        stl = slots.tolist()
        return [(i, int(wl[i]), int(sl[i]), int(stl[i])) for i in keep]

    def __call__(self, model, x0, t, kwargs, batch, fm, step, fm_norm=None):
        args = self.args
        if args.strong_style_mode == 'C' and step % args.strong_style_cf_every:
            return fm.new_zeros(())
        # Raw labels, not independently dropped labels from the normal FM forward.
        eligible = self._eligible_rows(batch, t)
        if not eligible:
            if step % 50 == 0:
                self.logger.info('[strong-style-step] step=%d active=0 (no eligible t/script)', step)
            return fm.new_zeros(())
        order = (list(range(len(eligible))) if args.strong_style_samples == 0 else
                 torch.randperm(len(eligible)).tolist()[:args.strong_style_samples])
        rows = [eligible[i] for i in order]
        idx = torch.tensor([r[0] for r in rows], device=x0.device)
        ya = torch.tensor([r[3] for r in rows], device=x0.device)
        ka = {k: v[idx] for k, v in kwargs.items() if isinstance(v, torch.Tensor)
              and k not in ('y_pair', 'return_intermediate_layers', 'return_intermediate_layer')}
        ka['y_callig'] = torch.tensor([r[1] for r in rows], device=x0.device)
        ka['y_script'] = torch.tensor([r[2] for r in rows], device=x0.device)
        ka.pop('y_callig_raw', None)
        # Use canonical, undropped STD for both generator and critic.
        g = batch['std_soft'][idx.cpu()].float().to(x0.device)
        if g.ndim == 3:
            g = g[:, None]
        # ★ 2026-10-14 [重构 P1-6] 量纲判定**只做一次** ✓（旧实现每步 `g.max() > 1.5` ✗
        #   那是每步一次 CUDA 同步 ✗）。量纲是**数据属性** ✓ 不会在 run 中改变 ✓
        #   ⇒ 缓存后语义完全一致 ✓ 且省掉每步同步 ✓
        _gs = getattr(self, '_g_scale', None)
        if _gs is None:
            _gs = 1.0 if float(g.detach().amax()) <= 1.5 else (1.0 / 255.0)
            self._g_scale = _gs
        if _gs != 1.0:
            g = g * _gs
        ka['g'] = g
        tt = t[idx].float()
        if getattr(args, 'strong_style_full_batch', False):
            tt = torch.rand_like(tt) * (args.critic_t_max - args.critic_t_min) + args.critic_t_min
        noise = torch.randn_like(x0[idx])
        xt = (1 - tt[:, None, None, None]) * x0[idx] + tt[:, None, None, None] * noise
        # eval disables independent condition dropout, NOT autograd.
        modes = [(m, m.training) for m in model.modules()]
        model.eval()
        try:
            def predict(kw):
                from src.loss.flow_matching import TIME_SCALE
                with torch.autocast(x0.device.type, dtype=torch.bfloat16, enabled=x0.is_cuda):
                    velocity = model(xt, tt * TIME_SCALE, **kw)
                if isinstance(velocity, tuple):
                    velocity = velocity[0]
                return xt - tt[:, None, None, None] * velocity[:, :x0.shape[1]].float()
            za = predict(ka)
            si = torch.tensor([self.critic.s2i[self.critic.slots[r[3]].split('·')[1]]
                               for r in rows], device=x0.device)
            oa = self.critic.model(za.float(), g, si)
            logits = oa['slot_logits']
            margin = writer_margin(logits, ya, self.mask)
            if args.strong_style_mode == 'A':
                lw = F.cross_entropy(logits.masked_fill(~self.mask[ya], -torch.inf), ya)
            else:
                lw = calibrated_hinge(margin, self.targets[ya])
            parts = {'writer': (lw, args.strong_style_ratio)}
            if args.strong_style_mode == 'C':
                rivals = []
                for target in ya.tolist():
                    valid = [j for j in self.inverse if j != target and self.mask[target, j]]
                    if not valid:
                        raise ValueError('no mapped counterfactual writer')
                    rivals.append(valid[torch.randint(len(valid), ()).item()])
                yb = torch.tensor(rivals, device=x0.device)
                kb = dict(ka)
                kb['y_callig'] = torch.tensor([self.inverse[j][0] for j in rivals], device=x0.device)
                zb = predict(kb)  # exactly the same xt, t, g and script; only writer differs
                ob = self.critic.model(zb.float(), g, si)
                parts['writer'] = ((lw + calibrated_hinge(writer_margin(ob['slot_logits'], yb, self.mask),
                                                         self.targets[yb])) / 2, args.strong_style_ratio)
                parts['cf'] = (cf_loss(logits, ob['slot_logits'], ya, yb, self.mask), args.strong_style_cf_ratio)
            if args.strong_style_mode == 'D':
                parts['brush'] = (calibrated_hinge(oa['brush_logit'], self.brush_targets[ya]),
                                  args.strong_style_brush_ratio)
        finally:
            for module, training in modes:
                module.training = training
        parameters = [p for p in model.parameters() if p.requires_grad]
        probe = step % args.strong_style_probe_every == 0 or any(k not in self.scale for k in parts)
        nf = (fm_norm if fm_norm is not None else norm(fm, parameters)) if probe else None
        budget = GradientBudget(args.strong_style_max_weight, self.scale,
                                self.__dict__.setdefault('weak_probes', {}), self.logger)
        total, messages = budget.balance(parts, parameters, nf, step)
        if probe or step % 50 == 0:
            self.logger.info('[strong-style-step] step=%d active=%d margin=%.4f %s',
                             step, len(rows), margin.detach().mean(), ' | '.join(messages))
        return total

# -*- coding: utf-8 -*-
"""注入层的**层级敏感度探针**：让模型自己招供"它想在哪些层看骨架"。

只跑几百~几千步（几分钟），从两个互补角度量化每一层。

A. **梯度饥渴度**（Gradient Hunger）
   ``‖∂L/∂W_out‖`` / ``‖∂L/∂(block 输出)‖``  —— 分子是零初始化注入层的权重梯度，
   分母是该 block 输出梯度。
   ⚠ **必须做这个归一化**：实测（12 层全挂, 800 步）原始 ``‖∂L/∂W_out‖`` 随深度
   **单调递增**（0.25@L0 → 0.40@L10），那是"反向路径越短梯度越大"的深度偏置，
   测的是深度不是需求。除以同层 block 输出梯度后两者共享该偏置，比值剩下的才是
   "这一层对骨架注入的净需求"。同时记录 ``‖W_out‖``（注入模块"苏醒"到什么程度）。

B. **空间寻址度**（Spatial Routing）
   * ``diag``    ：Q 第 k 个 token 投向 K 中**同一网格位置**骨架 token 的权重。
                   随机/未学 ≈ 1/N（N=256 → 0.00391）。**这是"是否在做空间寻址"的直接证据**。
   * ``entropy`` ：注意力熵（nats）；均匀分布 = ln N ≈ 5.545。
   * ``maxw``    ：每 query 最大注意力权重均值（"锐度"）。

   ⚠ **读数前先看 diag 是否离开 1/N**：out_proj 零初始化 → 反向传播时
   ``∂L/∂(q,k,v) ∝ W_out = 0``，所以前若干步**注意力完全学不动**（实测 800 步时
   12 层的 diag 全在 0.96~1.05× uniform，熵 ≈ 5.3）。此时 diag 排名是噪声，
   必须跑够步数（或先让 W_out 长起来）才有意义。

判据：``ratio``（净需求）与 ``diag``（真在寻址）**都高的层**才是该注入的层；
只看其中一个都会被上面两个陷阱骗到。

用法（train.py 内）::

    probe = InjectLayerProbe(args, model, logger)   # --probe-inject-every > 0 才生效
    probe.maybe_log(train_steps)                    # 放在 train_steps += 1 之后
"""
from __future__ import annotations

import os


class InjectLayerProbe(object):
    def __init__(self, args, model, logger=None):
        self.every = int(getattr(args, "probe_inject_every", 0) or 0)
        self.csv_path = str(getattr(args, "probe_inject_csv", "") or "")
        self.logger = logger
        inner = getattr(model, "module", model)          # 兼容 DDP 包装
        self.model = inner
        self.grad = {}          # 注入层 out_proj 权重梯度范数
        self.blk_grad = {}      # 同层 block 输出梯度范数 (归一化分母)
        self.written = False

        inj = getattr(inner, "glyph_injections", None)
        if self.every <= 0:
            return
        if inj is None:
            raise RuntimeError(
                "[inject-probe] 模型没有 glyph_injections —— 探针需要逐层注入模块 "
                "(glyph_inject_layers>0, glyph_inject_mode 为 adaln/xattn)。")

        blocks = getattr(inner, "blocks", None)
        for i, m in enumerate(inj):
            if not hasattr(m, "out_proj"):
                raise RuntimeError(f"[inject-probe] 第 {i} 个注入模块没有 out_proj，无法探测")
            m.probe = True              # 让模块在 forward 里记录 attention 统计
            m.probe_stats = None
            w = m.out_proj.weight
            if not hasattr(w, "register_post_accumulate_grad_hook"):
                raise RuntimeError(
                    "[inject-probe] torch <2.1 缺少 register_post_accumulate_grad_hook；"
                    "否则梯度会在 opt.step/zero_grad 后拿不到。")
            w.register_post_accumulate_grad_hook(self._mk_w_hook(i))
            if blocks is not None and i < len(blocks):
                blocks[i].register_full_backward_hook(self._mk_blk_hook(i))

        if self.csv_path:
            d = os.path.dirname(self.csv_path)
            if d:
                os.makedirs(d, exist_ok=True)
        if self.logger is not None:
            self.logger.info(
                "[inject-probe] 已挂 %d 层, 并挂 %d 个 block 反向 hook（用于深度归一化）；"
                "每 %d 步记录；csv=%s",
                len(inj), len(self.blk_grad) if self.blk_grad else len(inj),
                self.every, self.csv_path or "(不落盘)")

    # ------------------------------------------------------------------ hooks
    def _mk_w_hook(self, layer_idx):
        def _hook(grad):
            self.grad[layer_idx] = float(grad.detach().norm())
        return _hook

    def _mk_blk_hook(self, layer_idx):
        def _hook(_mod, _grad_in, grad_out):
            if grad_out and grad_out[0] is not None:
                self.blk_grad[layer_idx] = float(grad_out[0].detach().norm())
        return _hook

    # ------------------------------------------------------------------- dump
    def maybe_log(self, step):
        if self.every <= 0 or int(step) % self.every != 0:
            return
        inj = getattr(self.model, "glyph_injections", None) or []
        rows = []
        for i, m in enumerate(inj):
            st = getattr(m, "probe_stats", None) or {}
            g = self.grad.get(i, float("nan"))
            bg = self.blk_grad.get(i, float("nan"))
            ratio = (g / bg) if (bg == bg and bg > 0) else float("nan")
            rows.append({
                "step": int(step), "layer": i,
                "grad_norm": g, "blk_grad": bg,
                "ratio": ratio,
                "w_norm": float(m.out_proj.weight.detach().norm()),
                "diag": st.get("diag", float("nan")),
                "entropy": st.get("entropy", float("nan")),
                "maxw": st.get("maxw", float("nan")),
                "n_key": st.get("n_key", 0),
            })

        _lines = ["", "=== Layer Sensitivity Probe @ step %d ===" % step,
                  " layer |  grad‖Wout‖ |  blk_grad  |  ratio  |   ‖Wout‖  |   diag   | diag/uni | entropy"]
        _nk = float(rows[0]["n_key"] or 256) or 256.0
        for r in rows:
            _lines.append("  %4d | %12.6f | %10.5f | %7.4f | %9.6f | %8.5f | %7.2fx | %7.4f"
                          % (r["layer"], r["grad_norm"], r["blk_grad"], r["ratio"],
                             r["w_norm"], r["diag"], r["diag"] / (1.0 / _nk),
                             r["entropy"]))

        def _top(key, k=4):
            v = [r for r in rows if r[key] == r[key]]
            v.sort(key=lambda r: -r[key])
            return [r["layer"] for r in v[:k]]

        _lines.append(" ratio(净需求) top4: %s" % _top("ratio"))
        _lines.append(" w_norm(苏醒)   top4: %s" % _top("w_norm"))
        _lines.append(" diag(寻址)     top4: %s   [uniform=%.5f]"
                      % (_top("diag"), 1.0 / _nk))
        _txt = "\n".join(_lines)
        if self.logger is not None:
            self.logger.info("%s", _txt)
        else:
            print(_txt)

        if self.csv_path:
            _new = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a") as fh:
                if _new:
                    fh.write("step,layer,grad_norm,blk_grad,ratio,w_norm,"
                             "diag,entropy,maxw,n_key\n")
                for r in rows:
                    fh.write("%d,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d\n"
                             % (r["step"], r["layer"], r["grad_norm"], r["blk_grad"],
                                r["ratio"], r["w_norm"], r["diag"], r["entropy"],
                                r["maxw"], r["n_key"]))
            self.written = True

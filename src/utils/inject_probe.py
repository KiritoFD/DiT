# -*- coding: utf-8 -*-
"""注入层的**层级敏感度探针**：让模型自己招供"它想在哪些层看到条件"。

只跑几百~几千步（几分钟），覆盖**两条注入通路**：

【通路 A】cross-attn 适配器（glyph_injections，骨架 std skel 走这条）
  A1 梯度饥渴度 ``ratio = ‖∂L/∂W_out‖ / ‖∂L/∂(block 输出)‖``
     ⚠ 必须归一化：实测原始 ``‖∂L/∂W_out‖`` 随深度**单调递增**（0.25@L0 → 0.40@L10），
     那是"反向路径越短梯度越大"的深度偏置，测的是深度不是需求。
     同时记录 ``‖W_out‖``（注入模块苏醒到什么程度）。
  A2 空间寻址度 ``diag``：Q 第 k 个 token 投向 K 中**同一网格位置**骨架 token 的权重。
     随机/未学 ≈ 1/N（N=256 → 0.00391）。熵的均匀参考 = ln N ≈ 5.545。
     ⚠ out_proj 零初始化 → ``∂L/∂(q,k,v) ∝ W_out = 0``，前几百步注意力**完全学不动**
     （实测 800 步时 12 层 diag 全在 0.96~1.05× uniform）→ 必须跑够步数才能读。

【通路 B】条件分层路由（cond_router.scales，表条件走这条）
  B1 每个注入层的可学习强度 scale 及其梯度 ``|∂L/∂scale_k|``。
     scale 是个**标量**乘子（``c_k = t + y_emb * scale_k``），所以它的梯度就是
     "这一层多想要表条件"的**直接、无参数量混淆**的度量；
     ``ratio = |∂L/∂scale_k| / ‖∂L/∂(block 输出)‖`` 同样消掉深度偏置。
  B2 scale 的训练轨迹本身就是结果（→0 = 这层不想要；→ 大 = 这层贪求）。

**协议**：把注入层临时开到**全层**（`cond_inject_at`=0..11 或 `glyph_inject_at`=0..11），
跑几百~几千步 —— 这是 RelaCtrl 式"全层扰动 → 看哪层响应最大"的离线廉价版本。

输出：控制台表 + csv（``kind`` 列区分 adapter / router）。

用法（train.py 内）::

    probe = InjectLayerProbe(args, model, logger)   # --probe-inject-every > 0 才生效
    probe.maybe_log(train_steps)                    # 放在 train_steps += 1 之后
"""
from __future__ import annotations

import os

import torch


class InjectLayerProbe(object):
    def __init__(self, args, model, logger=None):
        self.every = int(getattr(args, "probe_inject_every", 0) or 0)
        self.csv_path = str(getattr(args, "probe_inject_csv", "") or "")
        self.logger = logger
        inner = getattr(model, "module", model)          # 兼容 DDP 包装
        self.model = inner
        self.w_grad = {}        # 适配器 out_proj 权重梯度范数
        self.blk_grad = {}      # block 输出梯度范数（归一化分母，两条通路共用）
        self.router_grad = None  # 路由 scale 的逐层 |grad|
        self.written = False

        if self.every <= 0:
            return

        blocks = getattr(inner, "blocks", None)
        inj = getattr(inner, "glyph_injections", None)
        router = getattr(inner, "cond_router", None)
        has_router_scale = hasattr(getattr(router, "scales", None),
                                   "register_post_accumulate_grad_hook")
        if inj is None and not has_router_scale:
            raise RuntimeError(
                "[inject-probe] 模型里既没有 glyph_injections 也没有可训练的 "
                "cond_router.scales —— 没有可探测的注入层。")

        _need = set()
        if inj is not None:
            for i, m in enumerate(inj):
                if not hasattr(m, "out_proj"):
                    raise RuntimeError(f"[inject-probe] 第 {i} 个注入模块没有 out_proj")
                m.probe = True            # 让模块 forward 里记录 attention 统计
                m.probe_stats = None
                w = m.out_proj.weight
                if not hasattr(w, "register_post_accumulate_grad_hook"):
                    raise RuntimeError("[inject-probe] torch <2.1 缺少 "
                                       "register_post_accumulate_grad_hook")
                w.register_post_accumulate_grad_hook(self._mk_w_hook(i))
                _need.add(i)
        if has_router_scale:
            _need |= set(int(b) for b in getattr(router, "layers", []) or [])

            def _rh(grad):
                self.router_grad = [float(x) for x in grad.detach().abs().flatten()]
            router.scales.register_post_accumulate_grad_hook(_rh)

        if blocks is not None:
            for i in sorted(_need):
                if 0 <= i < len(blocks):
                    blocks[i].register_full_backward_hook(self._mk_blk_hook(i))

        if self.csv_path:
            d = os.path.dirname(self.csv_path)
            if d:
                os.makedirs(d, exist_ok=True)
        if self.logger is not None:
            self.logger.info(
                "[inject-probe] 已挂: adapter=%s 层, router_scale=%s, block 反向 hook=%d 个; "
                "每 %d 步记录; csv=%s",
                "-" if inj is None else len(inj),
                "-" if not has_router_scale else len(getattr(router, "layers", [])),
                len(self.blk_grad) if self.blk_grad else len(_need),
                self.every, self.csv_path or "(不落盘)")

    # ------------------------------------------------------------------ hooks
    def _mk_w_hook(self, layer_idx):
        def _hook(grad):
            self.w_grad[layer_idx] = float(grad.detach().norm())
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
        inner = self.model
        inj = getattr(inner, "glyph_injections", None)
        router = getattr(inner, "cond_router", None)
        rows = []
        blk = lambda i: self.blk_grad.get(i, float("nan"))

        if inj is not None:
            for i, m in enumerate(inj):
                st = getattr(m, "probe_stats", None) or {}
                g = self.w_grad.get(i, float("nan"))
                bg = blk(i)
                rows.append(dict(
                    step=int(step), kind="adapter", layer=i,
                    grad_norm=g, blk_grad=bg,
                    ratio=(g / bg) if (bg == bg and bg > 0) else float("nan"),
                    w_norm=float(m.out_proj.weight.detach().norm()),
                    diag=st.get("diag", float("nan")),
                    entropy=st.get("entropy", float("nan")),
                    maxw=st.get("maxw", float("nan")),
                    n_key=st.get("n_key", 0)))

        if router is not None and hasattr(router, "scales"):
            lay = list(getattr(router, "layers", []) or [])
            sv = [float(x) for x in router.scales.detach().flatten()]
            gv = self.router_grad or [float("nan")] * len(lay)
            for k, b in enumerate(lay):
                g = gv[k] if k < len(gv) else float("nan")
                bg = blk(b)
                rows.append(dict(
                    step=int(step), kind="router", layer=b,
                    grad_norm=g, blk_grad=bg,
                    ratio=(g / bg) if (bg == bg and bg > 0) else float("nan"),
                    w_norm=sv[k] if k < len(sv) else float("nan"),
                    diag=float("nan"), entropy=float("nan"),
                    maxw=float("nan"), n_key=0))

        lines = ["", "=== Layer Sensitivity Probe @ step %d ===" % step]
        for kind in ("adapter", "router"):
            sub = [r for r in rows if r["kind"] == kind]
            if not sub:
                continue
            if kind == "adapter":
                nk = float(sub[0]["n_key"] or 256) or 256.0
                lines.append("[adapter]  layer |  grad‖Wout‖ |  blk_grad |  ratio |   ‖Wout‖  "
                             "|   diag   | diag/uni | entropy")
                for r in sub:
                    lines.append("          %4d | %11.6f | %9.5f | %6.4f | %9.6f "
                                 "| %8.5f | %7.2fx | %7.4f"
                                 % (r["layer"], r["grad_norm"], r["blk_grad"], r["ratio"],
                                    r["w_norm"], r["diag"], r["diag"] / (1.0 / nk),
                                    r["entropy"]))
            else:
                lines.append("[router ]  layer | |∂L/∂scale| |  blk_grad |  ratio |  scale")
                for r in sub:
                    lines.append("          %4d | %11.6f | %9.5f | %6.4f | %7.4f"
                                 % (r["layer"], r["grad_norm"], r["blk_grad"],
                                    r["ratio"], r["w_norm"]))
            v = [r for r in sub if r["ratio"] == r["ratio"]]
            v.sort(key=lambda r: -r["ratio"])
            lines.append("  %-7s ratio top4: %s" % (kind, [r["layer"] for r in v[:4]]))
            if kind == "router":
                lines.append("  router scale 现值: %s"
                             % {r["layer"]: round(r["w_norm"], 4) for r in sub})
        txt = "\n".join(lines)
        if self.logger is not None:
            self.logger.info("%s", txt)
        else:
            print(txt)

        if self.csv_path:
            _new = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a") as fh:
                if _new:
                    fh.write("step,kind,layer,grad_norm,blk_grad,ratio,w_norm,"
                             "diag,entropy,maxw,n_key\n")
                for r in rows:
                    fh.write("%d,%s,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d\n"
                             % (r["step"], r["kind"], r["layer"], r["grad_norm"],
                                r["blk_grad"], r["ratio"], r["w_norm"], r["diag"],
                                r["entropy"], r["maxw"], r["n_key"]))
            self.written = True

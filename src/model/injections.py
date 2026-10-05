# -*- coding: utf-8 -*-
"""注入层（2026-10-03 从 ``src/model/legacy/controlnet.py`` 上移）。

## 为什么要搬
``src/model/dit.py`` 的 **adaLN 注入分支是活代码**，却一直
``from .legacy.controlnet import ZeroAdaLNInjection`` —— 而 ControlNet（两阶段）
线早已废弃归档。**活代码 import 遗物**是脆的：只要归档 ``src/model/legacy/``，
训练立刻崩。这里把真正在跑的注入层上移成独立模块，解掉这条依赖。

## ⚠ 实现逐字未改（只换了家）
``state_dict`` 键名仍是 ``proj.weight`` / ``proj.bias``（键名不含类所在路径），
所以所有历史 adaLN ckpt 照常加载。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def zero_init_linear(in_f, out_f):
    lin = nn.Linear(in_f, out_f)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class ZeroAdaLNInjection(nn.Module):
    """adaLN 式零初始化注入：``out = x * (1 + s) + t``。

    ``s``/``t`` 由同一个 zero-init Linear 产出，因此 init 时 s=t=0，
    注入严格为恒等 —— 与 ControlNet 的 zero-conv warm-start 语义一致。

    梯度种子（为什么 step 0 就能学到东西）：
        d(out)/d(W) = x   （ctrl block 输出，非零 → W 立刻有梯度）
        d(out)/d(b) = 1   （bias 立刻有梯度）
        d(out)/d(x) = W = 0 → **ctrl blocks 在 W 变非零前收不到梯度**
    这是 ControlNet 的正确行为（先学注入权重，再学控制特征）。
    """

    def __init__(self, hidden_size, mode="modulate"):
        super().__init__()
        if mode not in ("modulate", "add"):
            raise ValueError(f"Unknown injection mode={mode!r}")
        self.mode = mode
        self.proj = zero_init_linear(hidden_size, hidden_size * (2 if mode == "modulate" else 1))

    def forward(self, x, feat):
        if self.mode == "modulate":
            s, t = self.proj(feat).chunk(2, dim=-1)
            return x * (1.0 + s) + t
        return x + self.proj(feat)


class TableCondRouter(nn.Module):
    """三表条件的**分层路由**：只在指定 block 允许条件进入，其余层**直接切断**。

    语义（对照 v54 的做法）：
        v54      ``c = t_emb + y_emb`` 喂给**全部** block —— 全局 adaLN，无处不注入；
        本模块   仅 ``layers`` 里的 block 拿到 ``y_emb``；其余 block 只拿 ``t_emb``，
                 条件在那几层**完全不参与任何计算**（不是"少注入"，是"不注入"）。
                 被切断的层：高频纹理/质感仍由主干自己的时间步调制渲染。

    依据（UNIC-Adapter 层级法则，1-indexed；我们的 block 是 0-indexed）：
        浅层 1-3（0,1,2）   对显式结构条件不敏感 → 注入等于噪声，不注入；
        中前 4-8（3..7）    物体轮廓 / 间架 / 低频拓扑的主战场 → **只在这里注入**；
        深层 9-12（8..11）  负责高频纹理（飞白/渗化/纸感）→ 注入会字迹发干、塑料描边感。

    每层一个可学习强度 ``scales``（init 1.0 = 恒等起步），于是：
        * step0 行为 = "v54 的条件，但只喂给这几层"，干净可比；
        * 训完 ``scales`` 就是"该层多需要这个条件"的曲线 —— 层级法则可以被**测出来**，
          而不只是被引用。
    """

    def __init__(self, layers, depth, learnable_scale=True):
        super().__init__()
        depth = int(depth)
        layers = sorted(set(int(b) for b in layers))
        bad = [b for b in layers if b < 0 or b >= depth]
        if bad:
            raise ValueError(f"[cond-route] 注入层越界: {bad} (depth={depth})")
        if not layers:
            raise ValueError("[cond-route] 空层列表 = 条件全切断, 请显式确认后再来")
        self.layers = layers
        self.depth = depth
        self._map = {b: k for k, b in enumerate(layers)}
        if learnable_scale:
            self.scales = nn.Parameter(torch.ones(len(layers)))
        else:
            self.register_buffer("scales", torch.ones(len(layers)), persistent=False)

    def __repr__(self):
        return (f"TableCondRouter(layers={self.layers}, depth={self.depth}, "
                f"learnable_scale={isinstance(self.scales, nn.Parameter)})")

    def allows(self, block_idx):
        """该 block 是否允许表条件进入。"""
        return block_idx in self._map

    def cond(self, block_idx, t_emb, y_emb):
        """该 block 的 adaLN 条件向量。

        白名单层 -> ``t_emb + y_emb * scale_k``；其余层 -> ``t_emb``（条件不参与计算）。
        """
        k = self._map.get(block_idx)
        if k is None:
            return t_emb
        return t_emb + y_emb * self.scales[k]

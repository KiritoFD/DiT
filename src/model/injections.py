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

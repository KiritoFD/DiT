"""utils/fasterkan.py —— FasterKAN 的占位实现。

⚠ moyun_2.py:30 `from utils.fasterkan import FasterKAN`，但注册表里
   **所有变体都是 use_kan=False**（见 docs/system/61_ref_moyun_code_audit.md §4），
   这个模块在 ref 里也缺失 —— 属于作者废弃的研究分支。

   这里给一个能 import 的占位：真要用 FasterKAN 请自行实现/装包。
"""
import torch
import torch.nn as nn


class FasterKAN(nn.Module):
    """占位：直接做线性变换（不是真的 KAN）。

    ⚠ 仅为让 `use_kan=False` 的路径能 import 成功。
       若要开 use_kan，必须换成真正的 KAN 实现。
    """

    def __init__(self, layers_hidden=None, grid_min=-2.0, grid_max=2.0,
                 num_grids=8, **kwargs):
        super().__init__()
        layers_hidden = layers_hidden or [64, 64]
        self.layers = nn.ModuleList([
            nn.Linear(layers_hidden[i], layers_hidden[i + 1])
            for i in range(len(layers_hidden) - 1)
        ])
        self.act = nn.SiLU()

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x

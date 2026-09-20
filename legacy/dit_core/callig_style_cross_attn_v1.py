# -*- coding: utf-8 -*-
"""CalligStyleCrossAttn v1（原版存档，2026-09-19 被 v15 多模态风格版替代）。

## 存档原因

v15（docs/919 + 多模态风格 K=4 设计）把书家条件从「1 个 128 维向量」升级为
「K=4 个 token (B, K, D)」的 MultiStyleEmbedder 查表。原版 CalligStyleCrossAttn
内部用 `style_proj: Linear(callig_dim -> n_style * hidden)` 把**单个向量**展开成
n_style 个 style token —— 与 (B, K, D) 输入不兼容。新版直接接收查表 token 并给
Q 加 2D sincos 位置编码（原版 Q 无位置，见 dit.py ZeroCrossAttention 的 q_pos 教训）。

按「新代码零死代码」的原则，v1 从 `src/model/dit.py` 原样移出到本文件；
`src/train/configs/` 里引用 `callig_style_attn` 的 5 个 c41x_scratch 历史 config
同步移入 `legacy/configs/`。

## 使用史

**没有任何已训练 run 启用过本模块**（doc 60：从未公平测过）——它只存在于
c41x_scratch 系列 re-eval config 里。因此删除不破坏任何 ckpt 的复评。

## 原版接口（如需复评旧实验，从本文件恢复）

    CalligStyleCrossAttn(callig_dim, hidden_size, num_heads, n_style=8)
    forward(g_tok, e_callig)   # e_callig: (N, callig_dim) 单个书家向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalligStyleCrossAttn(nn.Module):
    """callig 空间化的正确形态: 书家风格以 cross-attention 内容寻址方式调制骨架.

    与外挂(callig_spatial)的本质区别 —— 外挂把书家 128 维向量解算成"16x16 固定
    空间模板"加到骨架上, N 书家 N 个常数模板, 与写哪个字无关 -> 无"书家x字"交互,
    纯死重(实测 strict ±0.002)。

    cross-attn: 书家向量 -> N_style 个 style token(风格维度分解),
    query = 骨架 token g_tok(256, 随字变化的二维结构), K/V = style token。
    attention 权重是数据相关的: 同一书家写不同字, 骨架 token 内容不同 -> 权重不同
    -> 各 token 依自身结构动态吸收书家风格 -> 产生"书家x字x位置"三方交互,
    骨架按"这个字 x 这个书家"组合变形(结体差异), 而非固定常数模板。
    out_proj zero-init -> resume 恒等。
    """

    def __init__(self, callig_dim, hidden_size, num_heads, n_style=8):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.n_style = n_style
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # 书家向量 -> N_style 个 style token (风格维度分解)
        self.style_proj = nn.Sequential(
            nn.LayerNorm(callig_dim),
            nn.Linear(callig_dim, n_style * hidden_size),
        )
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, g_tok, e_callig):
        """g_tok: (N, 256, D) 骨架 token; e_callig: (N, C) 书家向量 -> 返回书家化骨架."""
        B, Nq, D = g_tok.shape
        style = self.style_proj(e_callig).view(B, self.n_style, D)   # (N, n_style, D)
        q = self.q_proj(self.norm_q(g_tok)).view(
            B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_kv(style)).view(
            B, self.n_style, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.norm_kv(style)).view(
            B, self.n_style, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return g_tok + self.out_proj(out)

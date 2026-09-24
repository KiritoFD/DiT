# -*- coding: utf-8 -*-
"""骨架查询层。

画布当 Query，带位置的骨架 token 当 Key 和 Value。风格改变的是 Query 和 Key
的方向，不改变它们的长度。长度由 QK-RMSNorm 吃掉，所以风格投影可以变大，
logit 仍然在 sqrt(head_dim) 附近。

不用 clamp。截断处的梯度是 0，上一轮就是在 softmax 之前这样炸的。
出口用一个可学习的 LayerScale，初值 0.1。它只有一个数，需要时可以涨到 1。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlyphQuery(nn.Module):
    def __init__(self, d_model, cond_dim, num_heads=4, rank=64, window=0,
                 grid_size=16):
        super().__init__()
        d_model = int(d_model)
        if d_model % int(num_heads) != 0:
            raise ValueError("d_model 必须被 num_heads 整除")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // self.num_heads
        self.d_model = d_model
        self.grid_size = int(round(grid_size))
        self.window = int(window)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 风格只提供一个加到 Q/K 上的方向。加在归一化之前，长度被 RMSNorm 吃掉。
        self.style_to_q = nn.Linear(int(cond_dim), d_model)
        self.style_to_k = nn.Linear(int(cond_dim), d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # LayerScale。exp(0.1) 不是初值，初值直接存 log(0.1)。
        self.out_log_scale = nn.Parameter(torch.tensor(-2.302585))
        self.reset()

        if self.window > 0:
            n = self.grid_size * self.grid_size
            idx = torch.arange(n)
            row, col = idx // self.grid_size, idx % self.grid_size
            half = self.window // 2
            keep = ((row.unsqueeze(1) - row.unsqueeze(0)).abs() <= half) & \
                   ((col.unsqueeze(1) - col.unsqueeze(0)).abs() <= half)
            keep = keep | torch.eye(n, dtype=torch.bool)
            self.register_buffer("win_mask", keep.unsqueeze(0).unsqueeze(0),
                                 persistent=False)
        else:
            self.win_mask = None

    def reset(self):
        """主干 ``_basic_init`` 会把 Linear 重新 xavier 一遍，所以它之后还要再调。"""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.style_to_q.weight)
        nn.init.zeros_(self.style_to_q.bias)
        nn.init.zeros_(self.style_to_k.weight)
        nn.init.zeros_(self.style_to_k.bias)
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(proj.weight, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.out_log_scale.data.fill_(-2.302585)

    def forward(self, q_src, g_tok, e_cond, pos, keep=None):
        b, n, _ = q_src.shape
        nc = g_tok.shape[1]
        if len({b, int(g_tok.shape[0]), int(e_cond.shape[0])}) != 1:
            raise RuntimeError(
                f"batch 不一致: q={b} g={int(g_tok.shape[0])} "
                f"style={int(e_cond.shape[0])}")
        h, hd = self.num_heads, self.head_dim
        pos = pos.to(q_src.dtype)
        e = e_cond.to(q_src.dtype)
        q = self.q_proj(self.norm_q(q_src + pos))
        q = q + self.style_to_q(e).unsqueeze(1)
        k = self.k_proj(self.norm_kv(g_tok + pos))
        k = k + self.style_to_k(e).unsqueeze(1)
        v = self.v_proj(self.norm_kv(g_tok + pos))

        q = q.view(b, n, h, hd).transpose(1, 2)
        k = k.view(b, nc, h, hd).transpose(1, 2)
        v = v.view(b, nc, h, hd).transpose(1, 2)
        q = F.rms_norm(q, (hd,))
        k = F.rms_norm(k, (hd,))

        scores = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
        if self.win_mask is not None and nc == self.win_mask.shape[-1]:
            scores = scores.masked_fill(~self.win_mask.to(torch.bool),
                                        torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, self.d_model)
        out = self.out_proj(out)
        if keep is not None:
            out = out * keep.to(out.dtype).view(-1, 1, 1)
        return q_src + self.out_log_scale.exp().to(out.dtype) * out

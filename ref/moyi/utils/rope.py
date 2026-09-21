"""utils/rope.py —— VisionRotaryEmbeddingFast 的最小实现。

⚠ 背景：moyun 的 `train_moyun2_RF.sh` 用 `--if-rope 0`，`Moyun.forward` 里
   `if not self.if_rope: x = x + self.pos_embed`，**rope 分支根本不会被调用**。
   但 `moyun_2.py:28` 是顶层 import，模块必须存在才能 import 成功。

   所以这里给一个**能跑通 import 的最小实现**，不做完整 RoPE。
   （原实现是把 rope 加在 token 序列上、每个 block 之前 —— 不是标准 q/k RoPE，
     见 docs/system/61_ref_moyun_code_audit.md §3.4，作者自己关掉了。）
"""
import torch
import torch.nn as nn


class VisionRotaryEmbeddingFast(nn.Module):
    """最小占位实现：按 (dim//4, dim//4) 的 2D 网格生成 cos/sin。

    ⚠ 未做完整数值验证 —— 仅保证 `if_rope=False` 时能 import 成功。
       若将来要开 if_rope，请先按标准 q/k RoPE 重写。
    """

    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, num_heads=16):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len or pt_seq_len
        self.num_heads = num_heads
        half = dim // 4
        freqs = 1.0 / (10000 ** (torch.arange(half).float() / half))
        t = torch.arange(self.ft_seq_len).float()
        freqs = torch.outer(t, freqs)          # (seq, half)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(self, x):
        """x: (B, T, C)。按 T 的位置乘上 cos/sin（简化版，仅占位）。"""
        T = x.shape[1]
        cos = self.freqs_cos[:T]
        sin = self.freqs_sin[:T]
        half = cos.shape[-1]
        x1, x2 = x[..., :half], x[..., half:2 * half]
        rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        rest = x[..., 2 * half:]
        return torch.cat([rot, rest], dim=-1) if rest.numel() else rot

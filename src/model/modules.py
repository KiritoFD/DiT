# -*- coding: utf-8 -*-
"""src.model.modules — 现代化 transformer 组件（RMSNorm / SwiGLU / 2D-RoPE / QK-Norm）。

与 DiT 原版（2022）组件的对应关系与动机
----------------------------------------
| 组件      | 原版                      | 本文件                      | 动机                          |
|-----------|---------------------------|-----------------------------|-------------------------------|
| 归一化    | LayerNorm(无 affine)      | RMSNorm(无 affine)          | 更快、更省显存、训练更稳      |
| FFN       | GELU(tanh) MLP            | SwiGLU                      | 等参数下更强                  |
| 位置编码  | 固定 2D sin-cos（加到 x） | 2D axial RoPE（作用 q/k）   | 相对位置、无外推问题          |
| Attention | timm.Attention            | QK-Norm + SDPA              | 防 attention logits 爆炸、更快|

关键设计约束
------------
1. **与 adaLN-Zero 完全兼容**：归一化层默认 ``elementwise_affine=False``
   （与 DiT 原版一致），尺度/平移全部由 adaLN 的 ``scale/shift`` 提供。
   因此 adaLN 零初始化 → 每个 block 初始仍为恒等映射。
2. **SwiGLU 参数量与原 GELU MLP 严格相等**：
   原版 ``2 * D * 4D = 8D²``；SwiGLU 取 ``h = 2/3 * 4D = 8D/3``，
   ``3 * D * h = 8D²``。这样对比实验是同参数量的公平对比。
3. **RoPE 缓存注册为 ``persistent=False`` buffer**：不进 ``state_dict``，
   因此不会破坏任何已有 checkpoint 的 key 集合。
4. **Attention forward 的 ``rope`` 是可选参数**，默认 ``None`` —— 老代码
   （如 ControlNet 的 ``DiTBlockSimple``）继续按 ``block(x, c)`` 调用不会炸。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_SDPA = getattr(F, "scaled_dot_product_attention", None)   # torch >= 2.0

# xformers memory-efficient attention：torch<2.0 且无 flash-attn 时的替代
# （注意 4090 服务器 torch=1.13.1 装的是 xformers 0.0.16，这里显式探测）
_XOPS = None
try:
    import xformers.ops as _xoops  # noqa: E402
    _XOPS = _xoops
except Exception:  # noqa: BLE001
    _XOPS = None

_SDPA_WARNED = False


def resolve_attn_impl(attn_impl):
    """把 attn_impl 解析成实际可用的实现。

    优先级：``sdpa``(torch>=2.0, F.scaled_dot_product_attention) >
    ``xformers``(memory_efficient_attention, torch<2.0 时的替代) >
    ``eager``(手写 softmax, 物化完整 attention 矩阵, 最慢最占显存)。
    早期版本在 torch<2.0 时把 ``attn_impl="sdpa"`` 悄悄降级成 eager，
    attention 矩阵被完整物化 (B·H·N² 个 fp32 元素) 导致显存翻倍/OOM。
    这里宁可显式报警，也不要静默降级。
    """
    global _SDPA_WARNED
    impl = (attn_impl or "auto").lower()
    if impl == "auto":
        if _SDPA is not None:
            impl = "sdpa"
        elif _XOPS is not None:
            impl = "xformers"
        else:
            impl = "eager"
    if impl == "sdpa" and _SDPA is None:
        if not _SDPA_WARNED:
            import logging
            if _XOPS is not None:
                logging.getLogger(__name__).warning(
                    "[modules] torch=%s 无 F.scaled_dot_product_attention，"
                    "已改用 xformers memory_efficient_attention；若出现数值/速度问题，"
                    "可显式设置 attn_impl='eager'。", torch.__version__)
            else:
                logging.getLogger(__name__).warning(
                    "[modules] attn_impl='sdpa' 不可用（torch=%s 无 "
                    "F.scaled_dot_product_attention，且未安装 xformers/flash-attn），"
                    "已回退到 'eager'。eager 会物化整个 attention 矩阵，"
                    "显存占用显著更高 —— 请相应调小 batch 或启用 use_checkpoint。",
                    torch.__version__)
            _SDPA_WARNED = True
        impl = "xformers" if _XOPS is not None else "eager"
    if impl not in ("sdpa", "xformers", "eager"):
        raise ValueError(f"Unknown attn_impl={attn_impl!r} "
                         f"(expected 'auto'/'sdpa'/'xformers'/'eager')")
    return impl


# --------------------------------------------------------------------------- #
# 归一化
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization。

    与 DiT 用法一致默认不带可学习的 affine（尺度由 adaLN 提供）。
    计算内部强制 fp32，向外 cast 回输入 dtype —— bf16 autocast 下更稳。
    """

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__()
        self.eps = float(eps)
        self.dim = int(dim)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        out = xf * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out.to(dtype)

    def extra_repr(self):
        return f"dim={self.dim}, eps={self.eps}, affine={self.weight is not None}"


def build_norm(norm_type, dim, eps=1e-6):
    """norm_type: "rms" | "layer"。均返回无 affine 的归一化（尺度交给 adaLN）。"""
    if norm_type == "rms":
        return RMSNorm(dim, eps=eps, elementwise_affine=False)
    if norm_type == "layer":
        return nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
    raise ValueError(f"Unknown norm_type={norm_type!r} (expected 'rms' or 'layer')")


# --------------------------------------------------------------------------- #
# FFN
# --------------------------------------------------------------------------- #
class Mlp(nn.Module):
    """原版 GELU(tanh) MLP，保留用于对照实验。参数名 fc1/fc2 与原实现一致。"""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN（LLaMA / Lumina 式），参数量对齐 GELU MLP。

    ``h = multiple_of * ceil((2/3) * mlp_ratio * D / multiple_of)``

    参数命名：``fc1``(up) / ``fc_gate``(gate) / ``fc2``(down)。
    这样 LoRA 注入目标 ``mlp.fc1`` / ``mlp.fc2`` 与旧代码保持一致，
    只需在新目标列表里补一个 ``mlp.fc_gate``。
    """

    def __init__(self, dim, hidden_dim=None, mlp_ratio=4.0, multiple_of=64, drop=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * mlp_ratio)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(F.silu(self.fc1(x)) * self.fc_gate(x)))


def build_mlp(mlp_type, dim, mlp_ratio=4.0, **kwargs):
    """mlp_type: "swiglu" | "gelu"。"""
    if mlp_type == "swiglu":
        return SwiGLUFeedForward(dim, mlp_ratio=mlp_ratio, **kwargs)
    if mlp_type == "gelu":
        hidden = int(dim * mlp_ratio)
        return Mlp(dim, hidden, **kwargs)
    raise ValueError(f"Unknown mlp_type={mlp_type!r} (expected 'swiglu' or 'gelu')")


# --------------------------------------------------------------------------- #
# 2D RoPE
# --------------------------------------------------------------------------- #
def precompute_rope_2d(grid_size, head_dim, theta=100.0, device=None, dtype=torch.float32):
    """2D axial RoPE 的 cos/sin 表。

    把 head_dim 的频率维分成两半：前半给 y，后半给 x；每半再按 rotate_half
    约定复制一次，最终 cos/sin 形状均为 ``(grid_size**2, head_dim)``。

    head_dim 必须能被 4 整除（两个轴各占 head_dim/4 个频率）。
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"2D RoPE requires head_dim % 4 == 0, got head_dim={head_dim} "
            f"(hidden_size / num_heads). Adjust hidden_size or num_heads.")
    d = head_dim // 4
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, device=device, dtype=torch.float32) / d))
    pos = torch.arange(grid_size, device=device, dtype=torch.float32)
    fy = torch.outer(pos, inv_freq)                      # (g, d)
    fx = torch.outer(pos, inv_freq)                      # (g, d)
    freqs = torch.cat([
        fy[:, None, :].expand(grid_size, grid_size, d),  # (g, g, d)
        fx[None, :, :].expand(grid_size, grid_size, d),  # (g, g, d)
    ], dim=-1).reshape(grid_size * grid_size, 2 * d)     # (N, head_dim/2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (N, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (N, head_dim)
    return cos.to(dtype), sin.to(dtype)


def apply_rope(x, cos, sin):
    """对 (B, H, N, Dh) 的 q/k 施加 RoPE。cos/sin: (N, Dh)。

    广播: (B,H,N,Dh) * (1,1,N,Dh) —— cos/sin 的 N 维必须对齐 token 维。
    """
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    # 让 cos/sin 跟随 x 的 dtype，避免 bf16 的 x 被 fp32 的 cos/sin 提升为 fp32，
    # 从而使 q/k 与 v 的 dtype 不一致（xformers memory_efficient_attention 严格要求统一 dtype）。
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos[None, None, :, :] + rot * sin[None, None, :, :]


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    """Multi-head self-attention，可选 QK-Norm / RoPE / SDPA。

    qk_norm 用 RMSNorm(head_dim)（带 affine，这是 QK-Norm 的标准做法：
    抑制 attention logits 的幅度爆炸，让高学习率 / bf16 训练更稳）。
    """

    def __init__(self, hidden_size, num_heads, qkv_bias=True, qk_norm=True,
                 attn_drop=0.0, proj_drop=0.0, attn_impl="sdpa"):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) must be divisible by num_heads({self.num_heads})")
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_impl = resolve_attn_impl(attn_impl)

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x, rope=None):
        """x: (B, N, D)；rope: 可选 (cos, sin)，各为 (N, head_dim)。"""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                       # (B, H, N, Dh)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.attn_impl == "sdpa":
            out = _SDPA(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        elif self.attn_impl == "xformers":
            # xformers 期望 (B, N, H, Dh)，这里 q/k/v 是 (B, H, N, Dh)
            out = _XOPS.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_bias=None, p=self.attn_drop if self.training else 0.0)
            out = out.transpose(1, 2)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if self.attn_drop and self.training:
                attn = F.dropout(attn, p=self.attn_drop)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


# --------------------------------------------------------------------------- #
# PatchEmbed（去掉 timm 依赖，接口与 timm 版保持一致）
# --------------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """(N, C, H, W) -> (N, T, D)，patch conv。

    属性与 timm 版对齐：``num_patches`` / ``patch_size``(tuple) / ``proj``。
    """

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=384, bias=True):
        super().__init__()
        self.input_size = (int(input_size), int(input_size))
        self.patch_size = (int(patch_size), int(patch_size))
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.grid_size = self.input_size[0] // self.patch_size[0]
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size,
                              kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# --------------------------------------------------------------------------- #
# DiT Block（现代化）
# --------------------------------------------------------------------------- #
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """adaLN-Zero DiT block，归一化/FFN/注意力均可切换。

    forward 的 ``rope`` 是**可选**参数，保持与原调用 ``block(x, c)`` 兼容。
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 norm_type="rms", mlp_type="swiglu", qk_norm=True,
                 attn_impl="sdpa", **block_kwargs):
        super().__init__()
        self.norm1 = build_norm(norm_type, hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                              qk_norm=qk_norm, attn_impl=attn_impl, **block_kwargs)
        self.norm2 = build_norm(norm_type, hidden_size)
        self.mlp = build_mlp(mlp_type, hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),
                                                  rope=rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# --------------------------------------------------------------------------- #
# FinalLayer
# --------------------------------------------------------------------------- #
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, norm_type="rms"):
        super().__init__()
        self.norm_final = build_norm(norm_type, hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


__all__ = [
    "RMSNorm", "build_norm", "Mlp", "SwiGLUFeedForward", "build_mlp",
    "precompute_rope_2d", "apply_rope", "Attention", "PatchEmbed",
    "DiTBlock", "FinalLayer", "modulate",
]

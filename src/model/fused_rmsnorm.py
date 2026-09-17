# -*- coding: utf-8 -*-
"""fused_rmsnorm.py — 单遍融合 RMSNorm (Triton, fwd+bwd)。

动机（2026-09-16 infra 审计）
----------------------------
torch.compile 会把 `torch.nn.functional.rms_norm`（或手写 RMSNorm）**分解**成两个内核：
  1) `triton_red_fused_..._mean_pow_rsqrt`：读一遍 x 求 mean(x^2)
  2) `triton_poi_fused_..._rsqrt_mul`  ：**再读一遍 x**、乘 rsqrt、写出 y
→ 每次 norm 移动 141 MB（读47+读47+写47），而理论上单遍只需 94 MB（读47+写47）。

本模块用 Triton 写一个**行内两遍但数据驻留寄存器**的 kernel：整行 N(≤512) 一次性 load 进
寄存器 → 归约 → 归一化 → 写出。因为一行只有 N 个元素，寄存器装得下，所以是"单次读"。

- 只处理 **无 affine 权重** 的 RMSNorm（本模型的 norm1/norm2）。带权重的 qk-norm 走原路径。
- 无梯度参数，因此不改变任何 state_dict key，可安全接续已有 ckpt。
- 非 CUDA / 无 triton / N 过大时自动回退。
"""

import torch
import torch.nn as nn

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

MAX_N = 4096          # 单行超过这个宽度就回退（寄存器放不下风险）


if _HAS_TRITON:

    @triton.jit
    def _rms_fwd_kernel(X, Y, N, eps,
                        stride_xm, stride_ym,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        cmask = cols < N
        offs = rows[:, None] * stride_xm + cols[None, :]
        mask = cmask[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=1)                       # (BLOCK_M,)
        rrms = tl.rsqrt(ss / N + eps)                    # (BLOCK_M,)
        y = x * rrms[:, None]
        tl.store(Y + rows[:, None] * stride_ym + cols[None, :],
                 y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _rms_bwd_kernel(DY, X, DX, N, eps,
                        stride_m,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        cmask = cols < N
        offs = rows[:, None] * stride_m + cols[None, :]
        mask = cmask[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=1)
        rrms = tl.rsqrt(ss / N + eps)
        xn = x * rrms[:, None]
        c = tl.sum(dy * xn, axis=1) / N                  # (BLOCK_M,)
        dx = (dy - xn * c[:, None]) * rrms[:, None]
        tl.store(DX + offs, dx.to(DX.dtype.element_ty), mask=mask)


class _FusedRMSNormFn(torch.autograd.Function):
    """无权重、按最后一维归一化的 RMSNorm。输入任意形状 (..., N) -> 展平成 (M, N)。"""

    @staticmethod
    def forward(ctx, x, eps):
        N = x.shape[-1]
        xf = x.reshape(-1, N)
        M = xf.shape[0]
        y = torch.empty_like(xf)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_M = max(1, min(16, 4096 // BLOCK_N))
        grid = (triton.cdiv(M, BLOCK_M),)
        _rms_fwd_kernel[grid](xf, y, N, eps, xf.stride(0), y.stride(0),
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)
        ctx.save_for_backward(xf)
        ctx.eps = eps
        ctx.M, ctx.N = M, N
        return y.reshape(x.shape)

    @staticmethod
    def backward(ctx, dy):
        (xf,) = ctx.saved_tensors
        N = ctx.N
        dyf = dy.reshape(-1, N).contiguous()
        dx = torch.empty_like(dyf)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_M = max(1, min(16, 4096 // BLOCK_N))
        grid = (triton.cdiv(dyf.shape[0], BLOCK_M),)
        _rms_bwd_kernel[grid](dyf, xf, dx, N, ctx.eps, dyf.stride(0),
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)
        return dx.reshape(dy.shape), None


def fused_rmsnorm(x, eps=1e-6):
    return _FusedRMSNormFn.apply(x, float(eps))


def available(x, n):
    return bool(_HAS_TRITON and x.is_cuda
                and x.dtype in (torch.bfloat16, torch.float16)
                and n <= MAX_N and n % 8 == 0 and x.is_contiguous())

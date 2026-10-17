# -*- coding: utf-8 -*-
"""Fused Triton operators for DiT (adaLN-RMSNorm, SwiGLU, Residual-Gate).

Provides drop-in replacements for elementwise bottlenecks in DiT blocks:
1. fused_adaln_rmsnorm: Fuses RMSNorm and adaLN modulation into a single SRAM kernel.
   Saves 50% memory bandwidth by avoiding intermediate unmodulated normalized tensors.
2. fused_swiglu: Fuses SiLU(x1) * x2 into a single kernel, avoiding materializing
   the intermediate activation in HBM.
3. fused_residual_gate: Fuses gate-scaling and residual accumulation.
"""
from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# =========================================================================== #
# 1. Fused adaLN-RMSNorm
# =========================================================================== #
# Math:
#   rms = rsqrt(mean(x^2, dim=-1) + eps)
#   x_hat = x * rms
#   y = x_hat * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
# =========================================================================== #

if HAS_TRITON:
    @triton.jit
    def _adaln_rmsnorm_fwd_kernel(
        X_ptr,          # [B, T, D]
        SCALE_ptr,      # [B, D]
        SHIFT_ptr,      # [B, D]
        Y_ptr,          # [B, T, D]
        RMS_ptr,        # [B, T]
        stride_xb, stride_xt, stride_xd,
        stride_sb, stride_sd,
        stride_hb, stride_hd,
        stride_yb, stride_yt, stride_yd,
        stride_rb, stride_rt,
        B, T, D,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        pid_bt = tl.program_id(0)
        b = pid_bt // T
        t = pid_bt % T

        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        # Pointers
        x_ptrs = X_ptr + b * stride_xb + t * stride_xt + cols * stride_xd
        scale_ptrs = SCALE_ptr + b * stride_sb + cols * stride_sd
        shift_ptrs = SHIFT_ptr + b * stride_hb + cols * stride_hd
        y_ptrs = Y_ptr + b * stride_yb + t * stride_yt + cols * stride_yd
        rms_ptr = RMS_ptr + b * stride_rb + t * stride_rt

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Compute RMS: sqrt(mean(x^2) + eps)
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / D
        rms = tl.rsqrt(mean_sq + eps)

        # Store rms for backward pass
        tl.store(rms_ptr, rms)

        # Modulation: (x * rms) * (1 + scale) + shift
        x_norm = x * rms
        y = x_norm * (1.0 + scale) + shift

        tl.store(y_ptrs, y, mask=mask)


    @triton.jit
    def _adaln_rmsnorm_bwd_dx_kernel(
        DY_ptr,         # [B, T, D]
        X_ptr,          # [B, T, D]
        SCALE_ptr,      # [B, D]
        RMS_ptr,        # [B, T]
        DX_ptr,         # [B, T, D]
        DSCALE_OUT_ptr, # [B, T, D] intermediate before token sum
        stride_dyb, stride_dyt, stride_dyd,
        stride_xb, stride_xt, stride_xd,
        stride_sb, stride_sd,
        stride_rb, stride_rt,
        stride_dxb, stride_dxt, stride_dxd,
        stride_sob, stride_sot, stride_sod,
        B, T, D,
        BLOCK_D: tl.constexpr,
    ):
        pid_bt = tl.program_id(0)
        b = pid_bt // T
        t = pid_bt % T

        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        dy_ptrs = DY_ptr + b * stride_dyb + t * stride_dyt + cols * stride_dyd
        x_ptrs = X_ptr + b * stride_xb + t * stride_xt + cols * stride_xd
        scale_ptrs = SCALE_ptr + b * stride_sb + cols * stride_sd
        rms_ptr = RMS_ptr + b * stride_rb + t * stride_rt
        dx_ptrs = DX_ptr + b * stride_dxb + t * stride_dxt + cols * stride_dxd
        dscale_ptrs = DSCALE_OUT_ptr + b * stride_sob + t * stride_sot + cols * stride_sod

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        rms = tl.load(rms_ptr).to(tl.float32)

        x_hat = x * rms
        gamma = 1.0 + scale
        dy_hat = dy * gamma

        # Gradient w.r.t scale contribution: dy * x_hat
        tl.store(dscale_ptrs, dy * x_hat, mask=mask)

        # Gradient w.r.t x:
        # dx = rms * [dy_hat - (1/D) * x_hat * sum(dy_hat * x_hat)]
        dot = tl.sum(dy_hat * x_hat, axis=0)
        dx = rms * (dy_hat - (x_hat * dot) / D)

        tl.store(dx_ptrs, dx, mask=mask)


class FusedAdaLNRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6):
        # x: [B, T, D], scale: [B, D], shift: [B, D]
        orig_shape = x.shape
        orig_dtype = x.dtype
        B, T, D = orig_shape
        assert scale.shape == (B, D) and shift.shape == (B, D)

        if not x.is_cuda or not HAS_TRITON:
            # Fallback for CPU / no-triton
            xf = x.float()
            rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            x_norm = xf * rms
            y = x_norm * (1.0 + scale.unsqueeze(1).float()) + shift.unsqueeze(1).float()
            ctx.save_for_backward(x, scale, rms)
            ctx.eps = eps
            return y.to(orig_dtype)

        # Determine BLOCK_D as next power of 2
        BLOCK_D = triton.next_power_of_2(D)

        y = torch.empty_like(x)
        rms = torch.empty((B, T), device=x.device, dtype=torch.float32)

        grid = (B * T,)
        _adaln_rmsnorm_fwd_kernel[grid](
            x, scale, shift, y, rms,
            x.stride(0), x.stride(1), x.stride(2),
            scale.stride(0), scale.stride(1),
            shift.stride(0), shift.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
            rms.stride(0), rms.stride(1),
            B, T, D, eps,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x, scale, rms)
        ctx.eps = eps
        ctx.BLOCK_D = BLOCK_D
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, scale, rms = ctx.saved_tensors
        B, T, D = x.shape
        orig_dtype = dy.dtype

        # dshift is straightforward sum over tokens T
        # dy: [B, T, D] -> dshift: [B, D]
        dshift = dy.sum(dim=1)

        if not x.is_cuda or not HAS_TRITON:
            gamma = (1.0 + scale.unsqueeze(1).float())
            dy_hat = dy.float() * gamma
            x_hat = x.float() * rms
            dscale = (dy.float() * x_hat).sum(dim=1)
            dot = (dy_hat * x_hat).sum(dim=-1, keepdim=True)
            dx = rms * (dy_hat - (x_hat * dot) / D)
            return dx.to(orig_dtype), dscale.to(scale.dtype), dshift.to(scale.dtype), None

        BLOCK_D = ctx.BLOCK_D
        dx = torch.empty_like(x)
        dscale_tmp = torch.empty_like(x)

        grid = (B * T,)
        _adaln_rmsnorm_bwd_dx_kernel[grid](
            dy, x, scale, rms, dx, dscale_tmp,
            dy.stride(0), dy.stride(1), dy.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            scale.stride(0), scale.stride(1),
            rms.stride(0), rms.stride(1),
            dx.stride(0), dx.stride(1), dx.stride(2),
            dscale_tmp.stride(0), dscale_tmp.stride(1), dscale_tmp.stride(2),
            B, T, D,
            BLOCK_D=BLOCK_D,
        )

        dscale = dscale_tmp.sum(dim=1)
        return dx, dscale.to(scale.dtype), dshift.to(scale.dtype), None


def fused_adaln_rmsnorm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm + adaLN modulation: (x / RMS(x)) * (1 + scale) + shift."""
    return FusedAdaLNRMSNormFunction.apply(x, scale, shift, eps)


# =========================================================================== #
# 2. Fused SwiGLU: SiLU(x1) * x2
# =========================================================================== #
if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        X1_ptr, X2_ptr, OUT_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements

        x1 = tl.load(X1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # silu(x1) = x1 * sigmoid(x1)
        sig = tl.sigmoid(x1)
        silu = x1 * sig
        out = silu * x2

        tl.store(OUT_ptr + offsets, out, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        DOUT_ptr, X1_ptr, X2_ptr, DX1_ptr, DX2_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements

        dout = tl.load(DOUT_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(x1)
        silu = x1 * sig
        # d(silu)/dx1 = sig * (1 + x1 * (1 - sig))
        dsilu = sig * (1.0 + x1 * (1.0 - sig))

        dx1 = dout * x2 * dsilu
        dx2 = dout * silu

        tl.store(DX1_ptr + offsets, dx1, mask=mask)
        tl.store(DX2_ptr + offsets, dx2, mask=mask)


class FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape
        x1_c = x1.contiguous()
        x2_c = x2.contiguous()
        if not x1.is_cuda or not HAS_TRITON:
            out = F.silu(x1_c) * x2_c
            ctx.save_for_backward(x1_c, x2_c)
            return out

        out = torch.empty_like(x1_c)
        n_elements = x1_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _swiglu_fwd_kernel[grid](
            x1_c, x2_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        ctx.save_for_backward(x1_c, x2_c)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        x1_c, x2_c = ctx.saved_tensors
        dout_c = dout.contiguous()
        if not x1_c.is_cuda or not HAS_TRITON:
            sig = torch.sigmoid(x1_c)
            silu = x1_c * sig
            dsilu = sig * (1.0 + x1_c * (1.0 - sig))
            dx1 = dout_c * x2_c * dsilu
            dx2 = dout_c * silu
            return dx1, dx2

        dx1 = torch.empty_like(x1_c)
        dx2 = torch.empty_like(x2_c)
        n_elements = x1_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _swiglu_bwd_kernel[grid](
            dout_c, x1_c, x2_c, dx1, dx2, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return dx1, dx2


def fused_swiglu(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation: SiLU(x1) * x2."""
    return FusedSwiGLUFunction.apply(x1, x2)


# =========================================================================== #
# 3. Fused Gate + Residual Add: y = x + gate * res
# =========================================================================== #
if HAS_TRITON:
    @triton.jit
    def _gate_residual_fwd_kernel(
        X_ptr, RES_ptr, GATE_ptr, Y_ptr,
        stride_xb, stride_xt, stride_xd,
        stride_rb, stride_rt, stride_rd,
        stride_gb, stride_gd,
        stride_yb, stride_yt, stride_yd,
        B, T, D,
        BLOCK_D: tl.constexpr,
    ):
        pid_bt = tl.program_id(0).to(tl.int64)
        b = pid_bt // T
        t = pid_bt % T

        cols = tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D

        x_ptrs = X_ptr + b * stride_xb + t * stride_xt + cols * stride_xd
        res_ptrs = RES_ptr + b * stride_rb + t * stride_rt + cols * stride_rd
        gate_ptrs = GATE_ptr + b * stride_gb + cols * stride_gd
        y_ptrs = Y_ptr + b * stride_yb + t * stride_yt + cols * stride_yd

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)

        y = x + gate * res
        tl.store(y_ptrs, y, mask=mask)

    @triton.jit
    def _gate_residual_bwd_dres_kernel(
        DY_ptr, GATE_ptr, DRES_ptr, DGATE_OUT_ptr,
        stride_dyb, stride_dyt, stride_dyd,
        stride_gb, stride_gd,
        stride_drb, stride_drt, stride_drd,
        stride_dob, stride_dot, stride_dod,
        RES_ptr, stride_rb, stride_rt, stride_rd,
        B, T, D,
        BLOCK_D: tl.constexpr,
    ):
        pid_bt = tl.program_id(0).to(tl.int64)
        b = pid_bt // T
        t = pid_bt % T

        cols = tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D

        dy_ptrs = DY_ptr + b * stride_dyb + t * stride_dyt + cols * stride_dyd
        gate_ptrs = GATE_ptr + b * stride_gb + cols * stride_gd
        dres_ptrs = DRES_ptr + b * stride_drb + t * stride_drt + cols * stride_drd
        dgate_out_ptrs = DGATE_OUT_ptr + b * stride_dob + t * stride_dot + cols * stride_dod
        res_ptrs = RES_ptr + b * stride_rb + t * stride_rt + cols * stride_rd

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)

        dres = dy * gate
        dgate_elem = dy * res

        tl.store(dres_ptrs, dres, mask=mask)
        tl.store(dgate_out_ptrs, dgate_elem, mask=mask)


class FusedGateResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, res: torch.Tensor, gate: torch.Tensor):
        # x: [B, T, D], res: [B, T, D], gate: [B, D]
        assert x.shape == res.shape
        B, T, D = x.shape
        assert gate.shape == (B, D)

        if not x.is_cuda or not HAS_TRITON:
            y = x + gate.unsqueeze(1) * res
            ctx.save_for_backward(res, gate)
            return y

        BLOCK_D = triton.next_power_of_2(D)
        y = torch.empty_like(x)
        grid = (B * T,)

        _gate_residual_fwd_kernel[grid](
            x, res, gate, y,
            x.stride(0), x.stride(1), x.stride(2),
            res.stride(0), res.stride(1), res.stride(2),
            gate.stride(0), gate.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
            B, T, D,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(res, gate)
        ctx.BLOCK_D = BLOCK_D
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        res, gate = ctx.saved_tensors
        B, T, D = res.shape

        # dx is mathematically dy! Zero extra computation needed
        dx = dy

        if not res.is_cuda or not HAS_TRITON:
            dres = dy * gate.unsqueeze(1)
            dgate = (dy * res).sum(dim=1)
            return dx, dres, dgate

        BLOCK_D = ctx.BLOCK_D
        dres = torch.empty_like(res)
        dgate_tmp = torch.empty_like(res)

        grid = (B * T,)
        _gate_residual_bwd_dres_kernel[grid](
            dy, gate, dres, dgate_tmp,
            dy.stride(0), dy.stride(1), dy.stride(2),
            gate.stride(0), gate.stride(1),
            dres.stride(0), dres.stride(1), dres.stride(2),
            dgate_tmp.stride(0), dgate_tmp.stride(1), dgate_tmp.stride(2),
            res, res.stride(0), res.stride(1), res.stride(2),
            B, T, D,
            BLOCK_D=BLOCK_D,
        )
        dgate = dgate_tmp.sum(dim=1)
        return dx, dres, dgate


def fused_gate_residual(x: torch.Tensor, res: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fused gate scaling and residual accumulation: x + gate.unsqueeze(1) * res."""
    return FusedGateResidualFunction.apply(x, res, gate)


# =========================================================================== #
# 4. Fused SwiGLU with Packed Input [B, T, 2 * D_mlp]
# =========================================================================== #
if HAS_TRITON:
    @triton.jit
    def _swiglu_packed_fwd_kernel(
        H_ptr, OUT_ptr,
        stride_hb, stride_ht, stride_hd,
        stride_ob, stride_ot, stride_od,
        B, T, D_mlp,
        BLOCK_D: tl.constexpr,
    ):
        pid_bt = tl.program_id(0).to(tl.int64)
        b = pid_bt // T
        t = pid_bt % T

        cols = tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D_mlp

        # x1 at h[b, t, 0..D_mlp-1], x2 at h[b, t, D_mlp..2*D_mlp-1]
        x1_ptrs = H_ptr + b * stride_hb + t * stride_ht + cols * stride_hd
        x2_ptrs = H_ptr + b * stride_hb + t * stride_ht + (cols + D_mlp) * stride_hd
        out_ptrs = OUT_ptr + b * stride_ob + t * stride_ot + cols * stride_od

        x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(x1)
        silu = x1 * sig
        out = silu * x2
        tl.store(out_ptrs, out, mask=mask)

    @triton.jit
    def _swiglu_packed_bwd_kernel(
        DOUT_ptr, H_ptr, DH_ptr,
        stride_dob, stride_dot, stride_dod,
        stride_hb, stride_ht, stride_hd,
        stride_dhb, stride_dht, stride_dhd,
        B, T, D_mlp,
        BLOCK_D: tl.constexpr,
    ):
        pid_bt = tl.program_id(0).to(tl.int64)
        b = pid_bt // T
        t = pid_bt % T

        cols = tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D_mlp

        dout_ptrs = DOUT_ptr + b * stride_dob + t * stride_dot + cols * stride_dod
        x1_ptrs = H_ptr + b * stride_hb + t * stride_ht + cols * stride_hd
        x2_ptrs = H_ptr + b * stride_hb + t * stride_ht + (cols + D_mlp) * stride_hd

        dx1_ptrs = DH_ptr + b * stride_dhb + t * stride_dht + cols * stride_dhd
        dx2_ptrs = DH_ptr + b * stride_dhb + t * stride_dht + (cols + D_mlp) * stride_dhd

        dout = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(x1)
        silu = x1 * sig
        dsilu = sig * (1.0 + x1 * (1.0 - sig))

        dx1 = dout * x2 * dsilu
        dx2 = dout * silu

        tl.store(dx1_ptrs, dx1, mask=mask)
        tl.store(dx2_ptrs, dx2, mask=mask)


class FusedSwiGLUPackedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: torch.Tensor):
        # h: [B, T, 2 * D_mlp]
        B, T, total_D = h.shape
        assert total_D % 2 == 0
        D_mlp = total_D // 2

        if not h.is_cuda or not HAS_TRITON:
            x1 = h[..., :D_mlp]
            x2 = h[..., D_mlp:]
            out = F.silu(x1) * x2
            ctx.save_for_backward(h)
            ctx.D_mlp = D_mlp
            return out

        h_c = h.contiguous()
        BLOCK_D = triton.next_power_of_2(D_mlp)
        out = torch.empty((B, T, D_mlp), device=h.device, dtype=h.dtype)
        grid = (B * T,)

        _swiglu_packed_fwd_kernel[grid](
            h_c, out,
            h_c.stride(0), h_c.stride(1), h_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            B, T, D_mlp,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(h_c)
        ctx.D_mlp = D_mlp
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        h_c, = ctx.saved_tensors
        D_mlp = ctx.D_mlp
        B, T, _ = dout.shape

        if not h_c.is_cuda or not HAS_TRITON:
            x1 = h_c[..., :D_mlp]
            x2 = h_c[..., D_mlp:]
            sig = torch.sigmoid(x1)
            silu = x1 * sig
            dsilu = sig * (1.0 + x1 * (1.0 - sig))
            dx1 = dout * x2 * dsilu
            dx2 = dout * silu
            return torch.cat([dx1, dx2], dim=-1)

        BLOCK_D = ctx.BLOCK_D
        dh = torch.empty_like(h_c)
        dout_c = dout.contiguous()
        grid = (B * T,)

        _swiglu_packed_bwd_kernel[grid](
            dout_c, h_c, dh,
            dout_c.stride(0), dout_c.stride(1), dout_c.stride(2),
            h_c.stride(0), h_c.stride(1), h_c.stride(2),
            dh.stride(0), dh.stride(1), dh.stride(2),
            B, T, D_mlp,
            BLOCK_D=BLOCK_D,
        )
        return dh


def fused_swiglu_packed(h: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU on packed tensor [B, T, 2 * D_mlp] -> [B, T, D_mlp]."""
    return FusedSwiGLUPackedFunction.apply(h)


# =========================================================================== #
# 5. Fused 2D RoPE: (B, H, N, Dh) with cos/sin (N, Dh)
# =========================================================================== #
if HAS_TRITON:
    @triton.jit
    def _rope_fwd_kernel(
        X_ptr, COS_ptr, SIN_ptr, OUT_ptr,
        stride_xb, stride_xh, stride_xn, stride_xd,
        stride_cn, stride_cd,
        stride_sn, stride_sd,
        stride_ob, stride_oh, stride_on, stride_od,
        B, H, N, Dh, half_d,
        BLOCK_DH: tl.constexpr,
    ):
        pid_bhn = tl.program_id(0).to(tl.int64)
        b = pid_bhn // (H * N)
        rem = pid_bhn % (H * N)
        h = rem // N
        n = rem % N

        cols = tl.arange(0, BLOCK_DH).to(tl.int64)
        mask = cols < half_d

        # x1 at cols, x2 at (cols + half_d)
        x1_ptrs = X_ptr + b * stride_xb + h * stride_xh + n * stride_xn + cols * stride_xd
        x2_ptrs = X_ptr + b * stride_xb + h * stride_xh + n * stride_xn + (cols + half_d) * stride_xd

        c1_ptrs = COS_ptr + n * stride_cn + cols * stride_cd
        c2_ptrs = COS_ptr + n * stride_cn + (cols + half_d) * stride_cd

        s1_ptrs = SIN_ptr + n * stride_sn + cols * stride_sd
        s2_ptrs = SIN_ptr + n * stride_sn + (cols + half_d) * stride_sd

        out1_ptrs = OUT_ptr + b * stride_ob + h * stride_oh + n * stride_on + cols * stride_od
        out2_ptrs = OUT_ptr + b * stride_ob + h * stride_oh + n * stride_on + (cols + half_d) * stride_od

        x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

        c1 = tl.load(c1_ptrs, mask=mask, other=0.0).to(tl.float32)
        c2 = tl.load(c2_ptrs, mask=mask, other=0.0).to(tl.float32)
        s1 = tl.load(s1_ptrs, mask=mask, other=0.0).to(tl.float32)
        s2 = tl.load(s2_ptrs, mask=mask, other=0.0).to(tl.float32)

        # rot = [-x2, x1] -> rot1 = -x2, rot2 = x1
        # out1 = x1 * c1 + rot1 * s1 = x1 * c1 - x2 * s1
        # out2 = x2 * c2 + rot2 * s2 = x2 * c2 + x1 * s2
        out1 = x1 * c1 - x2 * s1
        out2 = x2 * c2 + x1 * s2

        tl.store(out1_ptrs, out1, mask=mask)
        tl.store(out2_ptrs, out2, mask=mask)

    @triton.jit
    def _rope_bwd_kernel(
        DOUT_ptr, COS_ptr, SIN_ptr, DX_ptr,
        stride_dob, stride_doh, stride_don, stride_dod,
        stride_cn, stride_cd,
        stride_sn, stride_sd,
        stride_dxb, stride_dxh, stride_dxn, stride_dxd,
        B, H, N, Dh, half_d,
        BLOCK_DH: tl.constexpr,
    ):
        pid_bhn = tl.program_id(0).to(tl.int64)
        b = pid_bhn // (H * N)
        rem = pid_bhn % (H * N)
        h = rem // N
        n = rem % N

        cols = tl.arange(0, BLOCK_DH).to(tl.int64)
        mask = cols < half_d

        dout1_ptrs = DOUT_ptr + b * stride_dob + h * stride_doh + n * stride_don + cols * stride_dod
        dout2_ptrs = DOUT_ptr + b * stride_dob + h * stride_doh + n * stride_don + (cols + half_d) * stride_dod

        c1_ptrs = COS_ptr + n * stride_cn + cols * stride_cd
        c2_ptrs = COS_ptr + n * stride_cn + (cols + half_d) * stride_cd

        s1_ptrs = SIN_ptr + n * stride_sn + cols * stride_sd
        s2_ptrs = SIN_ptr + n * stride_sn + (cols + half_d) * stride_sd

        dx1_ptrs = DX_ptr + b * stride_dxb + h * stride_dxh + n * stride_dxn + cols * stride_dxd
        dx2_ptrs = DX_ptr + b * stride_dxb + h * stride_dxh + n * stride_dxn + (cols + half_d) * stride_dxd

        dout1 = tl.load(dout1_ptrs, mask=mask, other=0.0).to(tl.float32)
        dout2 = tl.load(dout2_ptrs, mask=mask, other=0.0).to(tl.float32)

        c1 = tl.load(c1_ptrs, mask=mask, other=0.0).to(tl.float32)
        c2 = tl.load(c2_ptrs, mask=mask, other=0.0).to(tl.float32)
        s1 = tl.load(s1_ptrs, mask=mask, other=0.0).to(tl.float32)
        s2 = tl.load(s2_ptrs, mask=mask, other=0.0).to(tl.float32)

        # d(loss)/dx1 = dout1 * c1 + dout2 * s2
        # d(loss)/dx2 = dout2 * c2 - dout1 * s1
        dx1 = dout1 * c1 + dout2 * s2
        dx2 = dout2 * c2 - dout1 * s1

        tl.store(dx1_ptrs, dx1, mask=mask)
        tl.store(dx2_ptrs, dx2, mask=mask)


class FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: [B, H, N, Dh], cos/sin: [N, Dh]
        B, H, N, Dh = x.shape
        assert cos.shape == (N, Dh) and sin.shape == (N, Dh)
        half_d = Dh // 2

        if not x.is_cuda or not HAS_TRITON:
            x1, x2 = x.chunk(2, dim=-1)
            rot = torch.cat([-x2, x1], dim=-1)
            c = cos.to(x.dtype)[None, None, :, :]
            s = sin.to(x.dtype)[None, None, :, :]
            out = x * c + rot * s
            ctx.save_for_backward(cos, sin)
            return out

        x_c = x.contiguous()
        cos_c = cos.contiguous()
        sin_c = sin.contiguous()

        BLOCK_DH = triton.next_power_of_2(half_d)
        out = torch.empty_like(x_c)
        grid = (B * H * N,)

        _rope_fwd_kernel[grid](
            x_c, cos_c, sin_c, out,
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            cos_c.stride(0), cos_c.stride(1),
            sin_c.stride(0), sin_c.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, N, Dh, half_d,
            BLOCK_DH=BLOCK_DH,
        )
        ctx.save_for_backward(cos_c, sin_c)
        ctx.BLOCK_DH = BLOCK_DH
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        cos_c, sin_c = ctx.saved_tensors
        B, H, N, Dh = dout.shape
        half_d = Dh // 2

        if not dout.is_cuda or not HAS_TRITON:
            dout1, dout2 = dout.chunk(2, dim=-1)
            c1, c2 = cos_c.to(dout.dtype).chunk(2, dim=-1)
            s1, s2 = sin_c.to(dout.dtype).chunk(2, dim=-1)
            c1 = c1[None, None, :, :]
            c2 = c2[None, None, :, :]
            s1 = s1[None, None, :, :]
            s2 = s2[None, None, :, :]
            dx1 = dout1 * c1 + dout2 * s2
            dx2 = dout2 * c2 - dout1 * s1
            return torch.cat([dx1, dx2], dim=-1), None, None

        BLOCK_DH = ctx.BLOCK_DH
        dx = torch.empty_like(dout)
        dout_c = dout.contiguous()
        grid = (B * H * N,)

        _rope_bwd_kernel[grid](
            dout_c, cos_c, sin_c, dx,
            dout_c.stride(0), dout_c.stride(1), dout_c.stride(2), dout_c.stride(3),
            cos_c.stride(0), cos_c.stride(1),
            sin_c.stride(0), sin_c.stride(1),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            B, H, N, Dh, half_d,
            BLOCK_DH=BLOCK_DH,
        )
        return dx, None, None


def fused_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused 2D RoPE rotation for (B, H, N, Dh) with zero chunk/cat allocations."""
    return FusedRoPEFunction.apply(x, cos, sin)


# =========================================================================== #
# 6. Fused QK-Norm + 2D RoPE: RMSNorm(x, weight) -> RoPE(., cos, sin)
# =========================================================================== #
if HAS_TRITON:
    @triton.jit
    def _qknorm_rope_fwd_kernel(
        X_ptr, WEIGHT_ptr, COS_ptr, SIN_ptr, OUT_ptr, RSTD_ptr,
        stride_xb, stride_xh, stride_xn, stride_xd,
        stride_cn, stride_cd,
        stride_sn, stride_sd,
        stride_ob, stride_oh, stride_on, stride_od,
        B, H, N, Dh, half_d,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_DH: tl.constexpr,
        eps: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        n_idx = pid % N
        h_idx = (pid // N) % H
        b_idx = pid // (N * H)

        offsets_1 = tl.arange(0, BLOCK_DH).to(tl.int64)
        offsets_2 = offsets_1 + half_d

        mask_1 = offsets_1 < half_d
        mask_2 = offsets_2 < Dh

        x_base = X_ptr + b_idx * stride_xb + h_idx * stride_xh + n_idx * stride_xn
        x1 = tl.load(x_base + offsets_1 * stride_xd, mask=mask_1, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + offsets_2 * stride_xd, mask=mask_2, other=0.0).to(tl.float32)

        sum_sq = tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)
        var = sum_sq / Dh
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(RSTD_ptr + pid, rstd)

        u1 = x1 * rstd
        u2 = x2 * rstd

        if HAS_WEIGHT:
            w1 = tl.load(WEIGHT_ptr + offsets_1, mask=mask_1, other=1.0).to(tl.float32)
            w2 = tl.load(WEIGHT_ptr + offsets_2, mask=mask_2, other=1.0).to(tl.float32)
            u1 = u1 * w1
            u2 = u2 * w2

        c_base = COS_ptr + n_idx * stride_cn
        s_base = SIN_ptr + n_idx * stride_sn

        cos1 = tl.load(c_base + offsets_1 * stride_cd, mask=mask_1, other=1.0).to(tl.float32)
        cos2 = tl.load(c_base + offsets_2 * stride_cd, mask=mask_2, other=1.0).to(tl.float32)
        sin1 = tl.load(s_base + offsets_1 * stride_sd, mask=mask_1, other=0.0).to(tl.float32)
        sin2 = tl.load(s_base + offsets_2 * stride_sd, mask=mask_2, other=0.0).to(tl.float32)

        y1 = u1 * cos1 - u2 * sin1
        y2 = u2 * cos2 + u1 * sin2

        out_base = OUT_ptr + b_idx * stride_ob + h_idx * stride_oh + n_idx * stride_on
        tl.store(out_base + offsets_1 * stride_od, y1, mask=mask_1)
        tl.store(out_base + offsets_2 * stride_od, y2, mask=mask_2)

    @triton.jit
    def _qknorm_rope_bwd_kernel(
        DOUT_ptr, X_ptr, RSTD_ptr, WEIGHT_ptr, COS_ptr, SIN_ptr, DX_ptr, DWEIGHT_ptr,
        stride_db, stride_dh, stride_dn, stride_dd,
        stride_xb, stride_xh, stride_xn, stride_xd,
        stride_cn, stride_cd,
        stride_sn, stride_sd,
        stride_oxb, stride_oxh, stride_oxn, stride_oxd,
        B, H, N, Dh, half_d,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_DH: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        n_idx = pid % N
        h_idx = (pid // N) % H
        b_idx = pid // (N * H)

        offsets_1 = tl.arange(0, BLOCK_DH).to(tl.int64)
        offsets_2 = offsets_1 + half_d

        mask_1 = offsets_1 < half_d
        mask_2 = offsets_2 < Dh

        dout_base = DOUT_ptr + b_idx * stride_db + h_idx * stride_dh + n_idx * stride_dn
        dy1 = tl.load(dout_base + offsets_1 * stride_dd, mask=mask_1, other=0.0).to(tl.float32)
        dy2 = tl.load(dout_base + offsets_2 * stride_dd, mask=mask_2, other=0.0).to(tl.float32)

        c_base = COS_ptr + n_idx * stride_cn
        s_base = SIN_ptr + n_idx * stride_sn

        cos1 = tl.load(c_base + offsets_1 * stride_cd, mask=mask_1, other=1.0).to(tl.float32)
        cos2 = tl.load(c_base + offsets_2 * stride_cd, mask=mask_2, other=1.0).to(tl.float32)
        sin1 = tl.load(s_base + offsets_1 * stride_sd, mask=mask_1, other=0.0).to(tl.float32)
        sin2 = tl.load(s_base + offsets_2 * stride_sd, mask=mask_2, other=0.0).to(tl.float32)

        du1 = dy1 * cos1 + dy2 * sin2
        du2 = dy2 * cos2 - dy1 * sin1

        rstd = tl.load(RSTD_ptr + pid).to(tl.float32)

        x_base = X_ptr + b_idx * stride_xb + h_idx * stride_xh + n_idx * stride_xn
        x1 = tl.load(x_base + offsets_1 * stride_xd, mask=mask_1, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + offsets_2 * stride_xd, mask=mask_2, other=0.0).to(tl.float32)

        w1 = x1 * rstd
        w2 = x2 * rstd

        if HAS_WEIGHT:
            gamma1 = tl.load(WEIGHT_ptr + offsets_1, mask=mask_1, other=1.0).to(tl.float32)
            gamma2 = tl.load(WEIGHT_ptr + offsets_2, mask=mask_2, other=1.0).to(tl.float32)

            dw1 = du1 * w1
            dw2 = du2 * w2
            tl.atomic_add(DWEIGHT_ptr + offsets_1, dw1, mask=mask_1)
            tl.atomic_add(DWEIGHT_ptr + offsets_2, dw2, mask=mask_2)

            v1 = du1 * gamma1
            v2 = du2 * gamma2
        else:
            v1 = du1
            v2 = du2

        dot_vw = (tl.sum(v1 * w1, axis=0) + tl.sum(v2 * w2, axis=0)) / Dh
        dx1 = rstd * (v1 - w1 * dot_vw)
        dx2 = rstd * (v2 - w2 * dot_vw)

        dx_base = DX_ptr + b_idx * stride_oxb + h_idx * stride_oxh + n_idx * stride_oxn
        tl.store(dx_base + offsets_1 * stride_oxd, dx1, mask=mask_1)
        tl.store(dx_base + offsets_2 * stride_oxd, dx2, mask=mask_2)


class FusedQKNormRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: Optional[torch.Tensor],
                cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6):
        B, H, N, Dh = x.shape
        half_d = Dh // 2
        has_weight = weight is not None

        if not x.is_cuda or not HAS_TRITON:
            # Fallback
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(var + eps)
            normed = (x.float() * rstd).to(x.dtype)
            if has_weight:
                normed = normed * weight
            x1, x2 = normed.chunk(2, dim=-1)
            rot = torch.cat([-x2, x1], dim=-1)
            c = cos.to(x.dtype)[None, None, :, :]
            s = sin.to(x.dtype)[None, None, :, :]
            out = normed * c + rot * s
            ctx.save_for_backward(x, weight, cos, sin, rstd)
            ctx.eps = eps
            ctx.has_weight = has_weight
            return out

        x_c = x.contiguous()
        cos_c = cos.contiguous()
        sin_c = sin.contiguous()
        weight_c = weight.contiguous() if has_weight else None

        BLOCK_DH = triton.next_power_of_2(half_d)
        out = torch.empty_like(x_c)
        rstd = torch.empty((B * H * N,), device=x.device, dtype=torch.float32)
        grid = (B * H * N,)

        _qknorm_rope_fwd_kernel[grid](
            x_c, weight_c if has_weight else x_c, cos_c, sin_c, out, rstd,
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            cos_c.stride(0), cos_c.stride(1),
            sin_c.stride(0), sin_c.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, N, Dh, half_d,
            HAS_WEIGHT=has_weight,
            BLOCK_DH=BLOCK_DH,
            eps=eps,
        )

        ctx.save_for_backward(x_c, weight_c, cos_c, sin_c, rstd)
        ctx.has_weight = has_weight
        ctx.BLOCK_DH = BLOCK_DH
        ctx.half_d = half_d
        ctx.Dh = Dh
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        x_c, weight_c, cos_c, sin_c, rstd = ctx.saved_tensors
        B, H, N, Dh = dout.shape
        has_weight = ctx.has_weight
        half_d = ctx.half_d
        BLOCK_DH = ctx.BLOCK_DH

        if not dout.is_cuda or not HAS_TRITON:
            dout_f = dout.float()
            c1, c2 = cos_c.float().chunk(2, dim=-1)
            s1, s2 = sin_c.float().chunk(2, dim=-1)
            c1, c2 = c1[None, None, :, :], c2[None, None, :, :]
            s1, s2 = s1[None, None, :, :], s2[None, None, :, :]
            dy1, dy2 = dout_f.chunk(2, dim=-1)
            du1 = dy1 * c1 + dy2 * s2
            du2 = dy2 * c2 - dy1 * s1
            du = torch.cat([du1, du2], dim=-1)

            x_f = x_c.float()
            w = x_f * rstd.view(B, H, N, 1)
            dweight = (du * w).sum(dim=(0, 1, 2)).to(weight_c.dtype) if has_weight else None
            v = du * weight_c.float() if has_weight else du
            dot_vw = (v * w).mean(dim=-1, keepdim=True)
            dx = (rstd.view(B, H, N, 1) * (v - w * dot_vw)).to(dout.dtype)
            return dx, dweight, None, None, None

        dout_c = dout.contiguous()
        dx = torch.empty_like(dout_c)
        dweight = torch.zeros((Dh,), device=dout.device, dtype=torch.float32) if has_weight else None
        grid = (B * H * N,)

        _qknorm_rope_bwd_kernel[grid](
            dout_c, x_c, rstd, weight_c if has_weight else x_c,
            cos_c, sin_c, dx, dweight if has_weight else dx,
            dout_c.stride(0), dout_c.stride(1), dout_c.stride(2), dout_c.stride(3),
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            cos_c.stride(0), cos_c.stride(1),
            sin_c.stride(0), sin_c.stride(1),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            B, H, N, Dh, half_d,
            HAS_WEIGHT=has_weight,
            BLOCK_DH=BLOCK_DH,
        )

        dweight_out = dweight.to(weight_c.dtype) if has_weight else None
        return dx, dweight_out, None, None, None


def fused_qknorm_rope(x: torch.Tensor, weight: Optional[torch.Tensor],
                      cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm + 2D RoPE rotation for Attention Q/K vectors with zero intermediate DRAM write."""
    return FusedQKNormRoPEFunction.apply(x, weight, cos, sin, eps)


# =========================================================================== #
# 7. Fused Gate + Residual Add + adaLN-RMSNorm:
#    x_res = x + gate * res
#    x_norm = RMSNorm(x_res) * (1 + scale) + shift
# =========================================================================== #
if HAS_TRITON:
    @triton.jit
    def _gate_res_adaln_fwd_kernel(
        X_ptr, RES_ptr, GATE_ptr, SCALE_ptr, SHIFT_ptr,
        OUT_RES_ptr, OUT_NORM_ptr, RSTD_ptr,
        stride_xb, stride_xt, stride_xd,
        stride_rb, stride_rt, stride_rd,
        stride_gb, stride_gd,
        stride_sb, stride_sd,
        stride_shb, stride_shd,
        stride_orb, stride_ort, stride_ord,
        stride_onb, stride_ont, stride_ond,
        B, T, D,
        BLOCK_D: tl.constexpr,
        eps: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        t_idx = pid % T
        b_idx = pid // T

        offsets = tl.arange(0, BLOCK_D).to(tl.int64)
        mask = offsets < D

        x_ptrs = X_ptr + b_idx * stride_xb + t_idx * stride_xt + offsets * stride_xd
        res_ptrs = RES_ptr + b_idx * stride_rb + t_idx * stride_rt + offsets * stride_rd
        gate_ptrs = GATE_ptr + b_idx * stride_gb + offsets * stride_gd

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)

        x_res = x + gate * res

        out_res_ptrs = OUT_RES_ptr + b_idx * stride_orb + t_idx * stride_ort + offsets * stride_ord
        tl.store(out_res_ptrs, x_res, mask=mask)

        sum_sq = tl.sum(x_res * x_res, axis=0)
        var = sum_sq / D
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(RSTD_ptr + pid, rstd)

        scale_ptrs = SCALE_ptr + b_idx * stride_sb + offsets * stride_sd
        shift_ptrs = SHIFT_ptr + b_idx * stride_shb + offsets * stride_shd

        scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptrs, mask=mask, other=0.0).to(tl.float32)

        normed = x_res * rstd
        y_norm = normed * (1.0 + scale) + shift

        out_norm_ptrs = OUT_NORM_ptr + b_idx * stride_onb + t_idx * stride_ont + offsets * stride_ond
        tl.store(out_norm_ptrs, y_norm, mask=mask)

    @triton.jit
    def _gate_res_adaln_bwd_kernel(
        DOUT_RES_ptr, DOUT_NORM_ptr, X_RES_ptr, RSTD_ptr,
        GATE_ptr, SCALE_ptr, RES_ptr,
        DX_ptr, DRES_ptr, DGATE_ptr, DSCALE_ptr, DSHIFT_ptr,
        stride_drb, stride_drt, stride_drd,
        stride_dnb, stride_dnt, stride_dnd,
        stride_xrb, stride_xrt, stride_xrd,
        stride_gb, stride_gd,
        stride_sb, stride_sd,
        stride_rb, stride_rt, stride_rd,
        stride_oxb, stride_oxt, stride_oxd,
        stride_orb, stride_ort, stride_ord,
        B, T, D,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        t_idx = pid % T
        b_idx = pid // T

        offsets = tl.arange(0, BLOCK_D).to(tl.int64)
        mask = offsets < D

        dn_ptrs = DOUT_NORM_ptr + b_idx * stride_dnb + t_idx * stride_dnt + offsets * stride_dnd
        dy_norm = tl.load(dn_ptrs, mask=mask, other=0.0).to(tl.float32)

        xr_ptrs = X_RES_ptr + b_idx * stride_xrb + t_idx * stride_xrt + offsets * stride_xrd
        x_res = tl.load(xr_ptrs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + pid).to(tl.float32)

        normed = x_res * rstd

        tl.atomic_add(DSCALE_ptr + b_idx * stride_sb + offsets * stride_sd, dy_norm * normed, mask=mask)
        tl.atomic_add(DSHIFT_ptr + b_idx * stride_sb + offsets * stride_sd, dy_norm, mask=mask)

        scale_ptrs = SCALE_ptr + b_idx * stride_sb + offsets * stride_sd
        scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        du = dy_norm * (1.0 + scale)

        dot_du_normed = tl.sum(du * normed, axis=0) / D
        dx_norm_in = rstd * (du - normed * dot_du_normed)

        dr_ptrs = DOUT_RES_ptr + b_idx * stride_drb + t_idx * stride_drt + offsets * stride_drd
        dy_res = tl.load(dr_ptrs, mask=mask, other=0.0).to(tl.float32)

        dx_total = dy_res + dx_norm_in

        dx_ptrs = DX_ptr + b_idx * stride_oxb + t_idx * stride_oxt + offsets * stride_oxd
        tl.store(dx_ptrs, dx_total, mask=mask)

        gate_ptrs = GATE_ptr + b_idx * stride_gb + offsets * stride_gd
        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        dres = dx_total * gate
        dres_ptrs = DRES_ptr + b_idx * stride_orb + t_idx * stride_ort + offsets * stride_ord
        tl.store(dres_ptrs, dres, mask=mask)

        res_ptrs = RES_ptr + b_idx * stride_rb + t_idx * stride_rt + offsets * stride_rd
        res = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(DGATE_ptr + b_idx * stride_gb + offsets * stride_gd, dx_total * res, mask=mask)


class FusedGateResidualAdaLNRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, res: torch.Tensor, gate: torch.Tensor,
                scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6):
        B, T, D = x.shape

        if not x.is_cuda or not HAS_TRITON:
            x_res = x + gate.unsqueeze(1) * res
            var = x_res.float().pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(var + eps)
            normed = (x_res.float() * rstd).to(x.dtype)
            x_norm = normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            ctx.save_for_backward(x_res, rstd, gate, scale, shift, res)
            ctx.eps = eps
            return x_res, x_norm

        x_c = x.contiguous()
        res_c = res.contiguous()
        gate_c = gate.contiguous()
        scale_c = scale.contiguous()
        shift_c = shift.contiguous()

        BLOCK_D = triton.next_power_of_2(D)
        out_res = torch.empty_like(x_c)
        out_norm = torch.empty_like(x_c)
        rstd = torch.empty((B * T,), device=x.device, dtype=torch.float32)
        grid = (B * T,)

        _gate_res_adaln_fwd_kernel[grid](
            x_c, res_c, gate_c, scale_c, shift_c,
            out_res, out_norm, rstd,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            res_c.stride(0), res_c.stride(1), res_c.stride(2),
            gate_c.stride(0), gate_c.stride(1),
            scale_c.stride(0), scale_c.stride(1),
            shift_c.stride(0), shift_c.stride(1),
            out_res.stride(0), out_res.stride(1), out_res.stride(2),
            out_norm.stride(0), out_norm.stride(1), out_norm.stride(2),
            B, T, D,
            BLOCK_D=BLOCK_D,
            eps=eps,
        )

        ctx.save_for_backward(out_res, rstd, gate_c, scale_c, shift_c, res_c)
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        return out_res, out_norm

    @staticmethod
    def backward(ctx, dout_res: torch.Tensor, dout_norm: torch.Tensor):
        x_res, rstd, gate_c, scale_c, shift_c, res_c = ctx.saved_tensors
        B, T, D = dout_res.shape

        if not dout_res.is_cuda or not HAS_TRITON:
            dn_f = dout_norm.float()
            normed = x_res.float() * rstd.view(B, T, 1)
            dscale = (dn_f * normed).sum(dim=1).to(scale_c.dtype)
            dshift = dn_f.sum(dim=1).to(shift_c.dtype)
            du = dn_f * (1.0 + scale_c.float().unsqueeze(1))
            dot_du_normed = (du * normed).mean(dim=-1, keepdim=True)
            dx_norm_in = rstd.view(B, T, 1) * (du - normed * dot_du_normed)

            dx_total = dout_res.float() + dx_norm_in
            dx = dx_total.to(dout_res.dtype)
            dres = (dx_total * gate_c.float().unsqueeze(1)).to(res_c.dtype)
            dgate = (dx_total * res_c.float()).sum(dim=1).to(gate_c.dtype)
            return dx, dres, dgate, dscale, dshift, None

        dout_res_c = dout_res.contiguous()
        dout_norm_c = dout_norm.contiguous()

        dx = torch.empty_like(dout_res_c)
        dres = torch.empty_like(dout_res_c)
        dgate = torch.zeros_like(gate_c, dtype=torch.float32)
        dscale = torch.zeros_like(scale_c, dtype=torch.float32)
        dshift = torch.zeros_like(shift_c, dtype=torch.float32)

        BLOCK_D = ctx.BLOCK_D
        grid = (B * T,)

        _gate_res_adaln_bwd_kernel[grid](
            dout_res_c, dout_norm_c, x_res, rstd,
            gate_c, scale_c, res_c,
            dx, dres, dgate, dscale, dshift,
            dout_res_c.stride(0), dout_res_c.stride(1), dout_res_c.stride(2),
            dout_norm_c.stride(0), dout_norm_c.stride(1), dout_norm_c.stride(2),
            x_res.stride(0), x_res.stride(1), x_res.stride(2),
            gate_c.stride(0), gate_c.stride(1),
            scale_c.stride(0), scale_c.stride(1),
            res_c.stride(0), res_c.stride(1), res_c.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            dres.stride(0), dres.stride(1), dres.stride(2),
            B, T, D,
            BLOCK_D=BLOCK_D,
        )

        return dx, dres, dgate.to(gate_c.dtype), dscale.to(scale_c.dtype), dshift.to(shift_c.dtype), None


def fused_gate_residual_adaln_rmsnorm(
    x: torch.Tensor, res: torch.Tensor, gate: torch.Tensor,
    scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused x_res = x + gate * res and x_norm = RMSNorm(x_res) * (1 + scale) + shift in single pass."""
    return FusedGateResidualAdaLNRMSNormFunction.apply(x, res, gate, scale, shift, eps)


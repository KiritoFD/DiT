# -*- coding: utf-8 -*-
"""Unit tests for Triton fused operators vs PyTorch reference implementations."""
import pytest
import torch
import torch.nn.functional as F

from src.model.fused_ops import (
    fused_adaln_rmsnorm, fused_swiglu, fused_gate_residual,
    fused_swiglu_packed, fused_rope, fused_qknorm_rope,
    fused_gate_residual_adaln_rmsnorm, HAS_TRITON
)
from src.model.modules import RMSNorm, modulate, apply_rope


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("B,T,D", [
    (2, 16, 384),
    (4, 256, 384),
    (8, 64, 512),
])
def test_fused_adaln_rmsnorm_forward(dtype, B, T, D):
    torch.manual_seed(42)
    device = "cuda"

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    scale = torch.randn(B, D, device=device, dtype=dtype)
    shift = torch.randn(B, D, device=device, dtype=dtype)

    # Reference implementation
    norm = RMSNorm(D, eps=1e-6, elementwise_affine=False)
    ref = modulate(norm(x), shift, scale)

    # Fused implementation
    fused = fused_adaln_rmsnorm(x, scale, shift, eps=1e-6)

    # Numerical verification
    tol = 1e-4 if dtype == torch.float32 else (2e-2 if dtype == torch.bfloat16 else 5e-3)
    assert torch.allclose(ref, fused, atol=tol, rtol=tol), f"Max diff: {(ref - fused).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_adaln_rmsnorm_backward(dtype):
    torch.manual_seed(42)
    device = "cuda"
    B, T, D = 4, 32, 384

    # 1. Autograd reference
    x_ref = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    scale_ref = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    shift_ref = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)

    norm = RMSNorm(D, eps=1e-6, elementwise_affine=False)
    y_ref = modulate(norm(x_ref), shift_ref, scale_ref)
    loss_ref = (y_ref ** 2).sum()
    loss_ref.backward()

    # 2. Fused operator
    x_fused = x_ref.detach().clone().requires_grad_(True)
    scale_fused = scale_ref.detach().clone().requires_grad_(True)
    shift_fused = shift_ref.detach().clone().requires_grad_(True)

    y_fused = fused_adaln_rmsnorm(x_fused, scale_fused, shift_fused, eps=1e-6)
    loss_fused = (y_fused ** 2).sum()
    loss_fused.backward()

    tol = 1e-3 if dtype == torch.float32 else 5e-2
    assert torch.allclose(x_ref.grad, x_fused.grad, atol=tol, rtol=tol), f"dx diff: {(x_ref.grad - x_fused.grad).abs().max()}"
    assert torch.allclose(scale_ref.grad, scale_fused.grad, atol=tol, rtol=tol), f"dscale diff: {(scale_ref.grad - scale_fused.grad).abs().max()}"
    assert torch.allclose(shift_ref.grad, shift_fused.grad, atol=tol, rtol=tol), f"dshift diff: {(shift_ref.grad - shift_fused.grad).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_swiglu(dtype):
    torch.manual_seed(42)
    device = "cuda"
    shape = (4, 256, 1024)

    x1_ref = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    x2_ref = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    y_ref = F.silu(x1_ref) * x2_ref
    loss_ref = (y_ref ** 2).sum()
    loss_ref.backward()

    x1_fused = x1_ref.detach().clone().requires_grad_(True)
    x2_fused = x2_ref.detach().clone().requires_grad_(True)

    y_fused = fused_swiglu(x1_fused, x2_fused)
    loss_fused = (y_fused ** 2).sum()
    loss_fused.backward()

    tol = 1e-4 if dtype == torch.float32 else (2e-2 if dtype == torch.bfloat16 else 5e-3)
    assert torch.allclose(y_ref, y_fused, atol=tol, rtol=tol)
    assert torch.allclose(x1_ref.grad, x1_fused.grad, atol=tol, rtol=tol)
    assert torch.allclose(x2_ref.grad, x2_fused.grad, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_gate_residual(dtype):
    torch.manual_seed(42)
    device = "cuda"
    B, T, D = 4, 32, 384

    x_ref = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    res_ref = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    gate_ref = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)

    y_ref = x_ref + gate_ref.unsqueeze(1) * res_ref
    loss_ref = (y_ref ** 2).mean()
    loss_ref.backward()

    x_fused = x_ref.detach().clone().requires_grad_(True)
    res_fused = res_ref.detach().clone().requires_grad_(True)
    gate_fused = gate_ref.detach().clone().requires_grad_(True)

    y_fused = fused_gate_residual(x_fused, res_fused, gate_fused)
    loss_fused = (y_fused ** 2).mean()
    loss_fused.backward()

    tol = 1e-4 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    assert torch.allclose(y_ref, y_fused, atol=tol, rtol=tol)
    assert torch.allclose(x_ref.grad, x_fused.grad, atol=tol, rtol=tol)
    assert torch.allclose(res_ref.grad, res_fused.grad, atol=tol, rtol=tol)
    assert torch.allclose(gate_ref.grad, gate_fused.grad, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_swiglu_packed(dtype):
    torch.manual_seed(42)
    device = "cuda"
    B, T, D_mlp = 4, 32, 1024

    x1 = torch.randn(B, T, D_mlp, device=device, dtype=dtype)
    x2 = torch.randn(B, T, D_mlp, device=device, dtype=dtype)
    h_ref = torch.cat([x1, x2], dim=-1).requires_grad_(True)

    y_ref = F.silu(h_ref[..., :D_mlp]) * h_ref[..., D_mlp:]
    loss_ref = (y_ref ** 2).sum()
    loss_ref.backward()

    h_fused = h_ref.detach().clone().requires_grad_(True)
    y_fused = fused_swiglu_packed(h_fused)
    loss_fused = (y_fused ** 2).sum()
    loss_fused.backward()

    tol = 1e-4 if dtype == torch.float32 else (2e-2 if dtype == torch.bfloat16 else 5e-3)
    assert torch.allclose(y_ref, y_fused, atol=tol, rtol=tol)
    assert torch.allclose(h_ref.grad, h_fused.grad, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_rope(dtype):
    torch.manual_seed(42)
    device = "cuda"
    B, H, N, Dh = 4, 8, 64, 48

    x_ref = torch.randn(B, H, N, Dh, device=device, dtype=dtype, requires_grad=True)
    cos = torch.randn(N, Dh, device=device, dtype=dtype)
    sin = torch.randn(N, Dh, device=device, dtype=dtype)

    y_ref = apply_rope(x_ref, cos, sin)
    loss_ref = (y_ref ** 2).mean()
    loss_ref.backward()

    x_fused = x_ref.detach().clone().requires_grad_(True)
    y_fused = fused_rope(x_fused, cos, sin)
    loss_fused = (y_fused ** 2).mean()
    loss_fused.backward()

    tol = 1e-4 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    assert torch.allclose(y_ref, y_fused, atol=tol, rtol=tol)
    assert torch.allclose(x_ref.grad, x_fused.grad, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_weight", [True, False])
def test_fused_qknorm_rope(dtype, has_weight):
    torch.manual_seed(42)
    device = "cuda"
    B, H, N, Dh = 2, 8, 32, 48

    x_ref = torch.randn(B, H, N, Dh, device=device, dtype=dtype, requires_grad=True)
    cos = torch.randn(N, Dh, device=device, dtype=dtype)
    sin = torch.randn(N, Dh, device=device, dtype=dtype)
    weight_ref = torch.randn(Dh, device=device, dtype=dtype, requires_grad=True) if has_weight else None

    # Reference computation
    var = x_ref.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + 1e-6)
    normed = (x_ref.float() * rstd).to(dtype)
    if has_weight:
        normed = normed * weight_ref
    y_ref = apply_rope(normed, cos, sin)
    loss_ref = (y_ref ** 2).mean()
    loss_ref.backward()

    # Fused computation
    x_fused = x_ref.detach().clone().requires_grad_(True)
    weight_fused = weight_ref.detach().clone().requires_grad_(True) if has_weight else None
    y_fused = fused_qknorm_rope(x_fused, weight_fused, cos, sin)
    loss_fused = (y_fused ** 2).mean()
    loss_fused.backward()

    tol = 1e-4 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    assert torch.allclose(y_ref, y_fused, atol=tol, rtol=tol)
    assert torch.allclose(x_ref.grad, x_fused.grad, atol=tol, rtol=tol)
    if has_weight:
        assert torch.allclose(weight_ref.grad, weight_fused.grad, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_gate_residual_adaln_rmsnorm(dtype):
    torch.manual_seed(42)
    device = "cuda"
    B, T, D = 4, 32, 384

    x_ref = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    res_ref = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    gate_ref = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    scale_ref = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    shift_ref = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)

    # Reference
    x_res_ref = x_ref + gate_ref.unsqueeze(1) * res_ref
    var = x_res_ref.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + 1e-6)
    normed = (x_res_ref.float() * rstd).to(dtype)
    x_norm_ref = normed * (1.0 + scale_ref.unsqueeze(1)) + shift_ref.unsqueeze(1)

    loss_ref = (x_res_ref ** 2).mean() + (x_norm_ref ** 2).mean()
    loss_ref.backward()

    # Fused
    x_fused = x_ref.detach().clone().requires_grad_(True)
    res_fused = res_ref.detach().clone().requires_grad_(True)
    gate_fused = gate_ref.detach().clone().requires_grad_(True)
    scale_fused = scale_ref.detach().clone().requires_grad_(True)
    shift_fused = shift_ref.detach().clone().requires_grad_(True)

    x_res_fused, x_norm_fused = fused_gate_residual_adaln_rmsnorm(
        x_fused, res_fused, gate_fused, scale_fused, shift_fused
    )
    loss_fused = (x_res_fused ** 2).mean() + (x_norm_fused ** 2).mean()
    loss_fused.backward()

    tol = 1e-4 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    assert torch.allclose(x_res_ref, x_res_fused, atol=tol, rtol=tol)
    assert torch.allclose(x_norm_ref, x_norm_fused, atol=tol, rtol=tol)
    assert torch.allclose(x_ref.grad, x_fused.grad, atol=tol, rtol=tol)
    assert torch.allclose(res_ref.grad, res_fused.grad, atol=tol, rtol=tol)
    assert torch.allclose(gate_ref.grad, gate_fused.grad, atol=tol, rtol=tol)
    assert torch.allclose(scale_ref.grad, scale_fused.grad, atol=tol, rtol=tol)
    assert torch.allclose(shift_ref.grad, shift_fused.grad, atol=tol, rtol=tol)

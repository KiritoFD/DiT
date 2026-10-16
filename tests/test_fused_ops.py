# -*- coding: utf-8 -*-
"""Unit tests for Triton fused operators vs PyTorch reference implementations."""
import pytest
import torch
import torch.nn.functional as F

from src.model.fused_ops import fused_adaln_rmsnorm, fused_swiglu, HAS_TRITON
from src.model.modules import RMSNorm, modulate


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

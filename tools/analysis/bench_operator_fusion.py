# -*- coding: utf-8 -*-
"""Comprehensive benchmark for baseline vs Triton-fused operators in DiT on RTX 4090.

Measures:
1. Micro-benchmarks: Fused adaLN-RMSNorm vs baseline, Fused SwiGLU vs baseline.
2. Macro-benchmarks: Single DiTBlock forward/backward latency & peak memory.
3. End-to-end model benchmark: Full 20-layer DiT-2Cond forward+backward throughput.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.modules import DiTBlock, RMSNorm, modulate, apply_rope
from src.model.fused_ops import (
    fused_adaln_rmsnorm, fused_swiglu, fused_gate_residual,
    fused_swiglu_packed, fused_rope, fused_qknorm_rope,
    fused_gate_residual_adaln_rmsnorm, HAS_TRITON
)
from src.model.dit import DiT_2Cond


def bench_fn(fn, warmups=20, iters=100):
    """Accurately measure CUDA execution time using CUDA Events."""
    torch.cuda.synchronize()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    return elapsed_ms


def run_micro_benchmarks(B=192, T=256, D=384, D_mlp=1024, dtype=torch.bfloat16):
    print("=" * 80)
    print(f"1. MICRO-BENCHMARKS: [B={B}, T={T}, D={D}, dtype={dtype}] on {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    device = "cuda"

    # --- 1.1 Fused adaLN-RMSNorm vs Baseline ---
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    scale = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    shift = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    norm = RMSNorm(D, eps=1e-6, elementwise_affine=False)

    def baseline_adaln_fwd():
        return modulate(norm(x), shift, scale)

    def fused_adaln_fwd():
        return fused_adaln_rmsnorm(x, scale, shift, eps=1e-6)

    # Memory traffic estimate:
    # Baseline: x (read) + x_norm (write) + x_norm (read) + scale/shift (read) + y (write)
    # Fused: x (read) + scale/shift (read) + y (write)
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    tensor_bytes = B * T * D * bytes_per_elem

    # Warmup & verify correctness
    out_ref = baseline_adaln_fwd()
    out_fused = fused_adaln_fwd()
    max_diff = (out_ref - out_fused).abs().max().item()
    print(f"-> adaLN-RMSNorm Forward Max Diff: {max_diff:.6e} (Aligned)")

    t_fwd_base = bench_fn(baseline_adaln_fwd)
    t_fwd_fused = bench_fn(fused_adaln_fwd)

    # Backward benchmarks
    def baseline_adaln_step():
        x.grad = scale.grad = shift.grad = None
        y = baseline_adaln_fwd()
        loss = y.sum()
        loss.backward()

    def fused_adaln_step():
        x.grad = scale.grad = shift.grad = None
        y = fused_adaln_fwd()
        loss = y.sum()
        loss.backward()

    t_step_base = bench_fn(baseline_adaln_step)
    t_step_fused = bench_fn(fused_adaln_step)

    print(f"adaLN-RMSNorm Forward : Base={t_fwd_base:6.3f} ms | Fused={t_fwd_fused:6.3f} ms | Speedup: {t_fwd_base/t_fwd_fused:.2f}x")
    print(f"adaLN-RMSNorm Fwd+Bwd : Base={t_step_base:6.3f} ms | Fused={t_step_fused:6.3f} ms | Speedup: {t_step_base/t_step_fused:.2f}x")

    # --- 1.2 Fused SwiGLU vs Baseline ---
    h1 = torch.randn(B, T, D_mlp, device=device, dtype=dtype, requires_grad=True)
    h2 = torch.randn(B, T, D_mlp, device=device, dtype=dtype, requires_grad=True)

    def baseline_swiglu_fwd():
        return F.silu(h1) * h2

    def fused_swiglu_fwd():
        return fused_swiglu(h1, h2)

    diff_swiglu = (baseline_swiglu_fwd() - fused_swiglu_fwd()).abs().max().item()
    print(f"-> SwiGLU Forward Max Diff: {diff_swiglu:.6e} (Aligned)")

    t_swiglu_base = bench_fn(baseline_swiglu_fwd)
    t_swiglu_fused = bench_fn(fused_swiglu_fwd)

    def baseline_swiglu_step():
        h1.grad = h2.grad = None
        y = baseline_swiglu_fwd()
        y.sum().backward()

    def fused_swiglu_step():
        h1.grad = h2.grad = None
        y = fused_swiglu_fwd()
        y.sum().backward()

    t_swiglu_step_base = bench_fn(baseline_swiglu_step)
    t_swiglu_step_fused = bench_fn(fused_swiglu_step)

    print(f"SwiGLU Forward        : Base={t_swiglu_base:6.3f} ms | Fused={t_swiglu_fused:6.3f} ms | Speedup: {t_swiglu_base/t_swiglu_fused:.2f}x")
    print(f"SwiGLU Fwd+Bwd        : Base={t_swiglu_step_base:6.3f} ms | Fused={t_swiglu_step_fused:6.3f} ms | Speedup: {t_swiglu_step_base/t_swiglu_step_fused:.2f}x")

    # --- 1.3 Fused Gate + Residual Add ---
    res = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    gate = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)

    def baseline_gate_fwd():
        return x + gate.unsqueeze(1) * res

    def fused_gate_fwd():
        return fused_gate_residual(x, res, gate)

    diff_gate = (baseline_gate_fwd() - fused_gate_fwd()).abs().max().item()
    print(f"-> Gate+Residual Forward Max Diff: {diff_gate:.6e} (Aligned)")

    t_gate_base = bench_fn(baseline_gate_fwd)
    t_gate_fused = bench_fn(fused_gate_fwd)

    def baseline_gate_step():
        x.grad = res.grad = gate.grad = None
        y = baseline_gate_fwd()
        y.sum().backward()

    def fused_gate_step():
        x.grad = res.grad = gate.grad = None
        y = fused_gate_fwd()
        y.sum().backward()

    t_gate_step_base = bench_fn(baseline_gate_step)
    t_gate_step_fused = bench_fn(fused_gate_step)

    print(f"Gate+Res Forward      : Base={t_gate_base:6.3f} ms | Fused={t_gate_fused:6.3f} ms | Speedup: {t_gate_base/t_gate_fused:.2f}x")
    print(f"Gate+Res Fwd+Bwd      : Base={t_gate_step_base:6.3f} ms | Fused={t_gate_step_fused:6.3f} ms | Speedup: {t_gate_step_base/t_gate_step_fused:.2f}x")

    # --- 1.4 Fused 2D RoPE ---
    H = 8
    Dh = D // H
    q = torch.randn(B, H, T, Dh, device=device, dtype=dtype, requires_grad=True)
    cos = torch.randn(T, Dh, device=device, dtype=dtype)
    sin = torch.randn(T, Dh, device=device, dtype=dtype)

    def baseline_rope_fwd():
        return apply_rope(q, cos, sin)

    def fused_rope_fwd():
        return fused_rope(q, cos, sin)

    diff_rope = (baseline_rope_fwd() - fused_rope_fwd()).abs().max().item()
    print(f"-> RoPE Forward Max Diff: {diff_rope:.6e} (Aligned)")

    t_rope_base = bench_fn(baseline_rope_fwd)
    t_rope_fused = bench_fn(fused_rope_fwd)

    def baseline_rope_step():
        q.grad = None
        y = baseline_rope_fwd()
        y.sum().backward()

    def fused_rope_step():
        q.grad = None
        y = fused_rope_fwd()
        y.sum().backward()

    t_rope_step_base = bench_fn(baseline_rope_step)
    t_rope_step_fused = bench_fn(fused_rope_step)

    print(f"RoPE Forward          : Base={t_rope_base:6.3f} ms | Fused={t_rope_fused:6.3f} ms | Speedup: {t_rope_base/t_rope_fused:.2f}x")
    print(f"RoPE Fwd+Bwd          : Base={t_rope_step_base:6.3f} ms | Fused={t_rope_step_fused:6.3f} ms | Speedup: {t_rope_step_base/t_rope_step_fused:.2f}x")

    # --- 1.5 FFN GEMM + SwiGLU (2 GEMMs vs 1 Packed GEMM + Fused SwiGLU) ---
    x_ffn = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    w1 = torch.randn(D_mlp, D, device=device, dtype=dtype, requires_grad=True)
    w_gate = torch.randn(D_mlp, D, device=device, dtype=dtype, requires_grad=True)

    def baseline_ffn_fwd():
        h1 = F.linear(x_ffn, w1)
        h2 = F.linear(x_ffn, w_gate)
        return F.silu(h1) * h2

    def fused_ffn_fwd():
        w_packed = torch.cat([w1, w_gate], dim=0)
        h = F.linear(x_ffn, w_packed)
        return fused_swiglu_packed(h)

    diff_ffn = (baseline_ffn_fwd() - fused_ffn_fwd()).abs().max().item()
    print(f"-> FFN GEMM+SwiGLU Forward Max Diff: {diff_ffn:.6e} (Aligned)")

    t_ffn_base = bench_fn(baseline_ffn_fwd)
    t_ffn_fused = bench_fn(fused_ffn_fwd)

    def baseline_ffn_step():
        x_ffn.grad = w1.grad = w_gate.grad = None
        y = baseline_ffn_fwd()
        y.sum().backward()

    def fused_ffn_step():
        x_ffn.grad = w1.grad = w_gate.grad = None
        y = fused_ffn_fwd()
        y.sum().backward()

    t_ffn_step_base = bench_fn(baseline_ffn_step)
    t_ffn_step_fused = bench_fn(fused_ffn_step)

    print(f"FFN 2-GEMM+Act Fwd    : Base={t_ffn_base:6.3f} ms | Fused={t_ffn_fused:6.3f} ms | Speedup: {t_ffn_base/t_ffn_fused:.2f}x")
    print(f"FFN 2-GEMM+Act Fwd+Bwd: Base={t_ffn_step_base:6.3f} ms | Fused={t_ffn_step_fused:6.3f} ms | Speedup: {t_ffn_step_base/t_ffn_step_fused:.2f}x")

    # --- 1.6 Fused QK-Norm + 2D RoPE ---
    gamma = torch.randn(Dh, device=device, dtype=dtype, requires_grad=True)

    def baseline_qknorm_rope_fwd():
        var = q.float().pow(2).mean(dim=-1, keepdim=True)
        normed = (q.float() * torch.rsqrt(var + 1e-6)).to(dtype) * gamma
        return apply_rope(normed, cos, sin)

    def fused_qknorm_rope_fwd():
        return fused_qknorm_rope(q, gamma, cos, sin)

    diff_qk = (baseline_qknorm_rope_fwd() - fused_qknorm_rope_fwd()).abs().max().item()
    print(f"-> QKNorm+RoPE Forward Max Diff: {diff_qk:.6e} (Aligned)")

    t_qk_base = bench_fn(baseline_qknorm_rope_fwd)
    t_qk_fused = bench_fn(fused_qknorm_rope_fwd)

    def baseline_qk_step():
        q.grad = gamma.grad = None
        y = baseline_qknorm_rope_fwd()
        y.sum().backward()

    def fused_qk_step():
        q.grad = gamma.grad = None
        y = fused_qknorm_rope_fwd()
        y.sum().backward()

    t_qk_step_base = bench_fn(baseline_qk_step)
    t_qk_step_fused = bench_fn(fused_qk_step)

    print(f"QKNorm+RoPE Forward   : Base={t_qk_base:6.3f} ms | Fused={t_qk_fused:6.3f} ms | Speedup: {t_qk_base/t_qk_fused:.2f}x")
    print(f"QKNorm+RoPE Fwd+Bwd   : Base={t_qk_step_base:6.3f} ms | Fused={t_qk_step_fused:6.3f} ms | Speedup: {t_qk_step_base/t_qk_step_fused:.2f}x")

    # --- 1.7 Fused Gate+Residual + adaLN-RMSNorm ---
    scale = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    shift = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)

    def baseline_gate_res_adaln_fwd():
        x_res = x + gate.unsqueeze(1) * res
        var = x_res.float().pow(2).mean(dim=-1, keepdim=True)
        normed = (x_res.float() * torch.rsqrt(var + 1e-6)).to(dtype)
        x_norm = normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x_res, x_norm

    def fused_gate_res_adaln_fwd():
        return fused_gate_residual_adaln_rmsnorm(x, res, gate, scale, shift)

    diff_gra1 = (baseline_gate_res_adaln_fwd()[0] - fused_gate_res_adaln_fwd()[0]).abs().max().item()
    diff_gra2 = (baseline_gate_res_adaln_fwd()[1] - fused_gate_res_adaln_fwd()[1]).abs().max().item()
    print(f"-> GateRes+adaLN Forward Max Diff: {max(diff_gra1, diff_gra2):.6e} (Aligned)")

    t_gra_base = bench_fn(baseline_gate_res_adaln_fwd)
    t_gra_fused = bench_fn(fused_gate_res_adaln_fwd)

    def baseline_gra_step():
        x.grad = res.grad = gate.grad = scale.grad = shift.grad = None
        y1, y2 = baseline_gate_res_adaln_fwd()
        (y1.sum() + y2.sum()).backward()

    def fused_gra_step():
        x.grad = res.grad = gate.grad = scale.grad = shift.grad = None
        y1, y2 = fused_gate_res_adaln_fwd()
        (y1.sum() + y2.sum()).backward()

    t_gra_step_base = bench_fn(baseline_gra_step)
    t_gra_step_fused = bench_fn(fused_gra_step)

    print(f"GateRes+adaLN Forward : Base={t_gra_base:6.3f} ms | Fused={t_gra_fused:6.3f} ms | Speedup: {t_gra_base/t_gra_fused:.2f}x")
    print(f"GateRes+adaLN Fwd+Bwd : Base={t_gra_step_base:6.3f} ms | Fused={t_gra_step_fused:6.3f} ms | Speedup: {t_gra_step_base/t_gra_step_fused:.2f}x")


def run_block_benchmarks(B=192, T=256, D=384, num_heads=8, dtype=torch.bfloat16):
    print("\n" + "=" * 80)
    print(f"2. BLOCK-LEVEL BENCHMARK: DiTBlock [B={B}, T={T}, D={D}, heads={num_heads}]")
    print("=" * 80)

    device = "cuda"
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    c = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)

    # 1. Baseline Block
    block_base = DiTBlock(
        hidden_size=D, num_heads=num_heads, mlp_ratio=4.0,
        norm_type="rms", mlp_type="swiglu", qk_norm=True,
        attn_impl="sdpa", use_fused_ops=False
    ).to(device=device, dtype=dtype)

    # 2. Fused Block
    block_fused = DiTBlock(
        hidden_size=D, num_heads=num_heads, mlp_ratio=4.0,
        norm_type="rms", mlp_type="swiglu", qk_norm=True,
        attn_impl="sdpa", use_fused_ops=True
    ).to(device=device, dtype=dtype)

    # Copy weights to ensure identical math
    block_fused.load_state_dict(block_base.state_dict())

    # Correctness check
    out_b = block_base(x, c)
    out_f = block_fused(x, c)
    max_d = (out_b - out_f).abs().max().item()
    print(f"-> Full DiTBlock Forward Max Diff: {max_d:.6e}")

    # Forward latency
    def step_fwd_base():
        return block_base(x, c)

    def step_fwd_fused():
        return block_fused(x, c)

    t_fwd_b = bench_fn(step_fwd_base)
    t_fwd_f = bench_fn(step_fwd_fused)

    # Full Step (Fwd + Bwd)
    def full_step_base():
        block_base.zero_grad(set_to_none=True)
        out = block_base(x, c)
        out.sum().backward()

    def full_step_fused():
        block_fused.zero_grad(set_to_none=True)
        out = block_fused(x, c)
        out.sum().backward()

    t_step_b = bench_fn(full_step_base)
    t_step_f = bench_fn(full_step_fused)

    print(f"DiTBlock Forward Latency  : Base={t_fwd_b:6.3f} ms | Fused={t_fwd_f:6.3f} ms | Speedup: {t_fwd_b/t_fwd_f:.2f}x")
    print(f"DiTBlock Fwd+Bwd Latency  : Base={t_step_b:6.3f} ms | Fused={t_step_f:6.3f} ms | Speedup: {t_step_b/t_step_f:.2f}x")
    print(f"-> 20-Layer Projected Saving: {(t_step_b - t_step_f) * 20:6.2f} ms per training step!")


def run_full_model_benchmarks(B=192, depth=20, D=384, num_heads=8, dtype=torch.bfloat16):
    print("\n" + "=" * 80)
    print(f"3. FULL MODEL BENCHMARK: DiT-2Cond 20L [B={B}, depth={depth}, D={D}, heads={num_heads}]")
    print("=" * 80)

    device = "cuda"
    x = torch.randn(B, 4, 32, 32, device=device)
    t = torch.rand(B, device=device)
    y_callig = torch.randint(0, 10, (B,), device=device)
    y_char = torch.randint(0, 8451, (B,), device=device)
    g = torch.randn(B, 1, 64, 64, device=device)

    print("Constructing Baseline 20L Model (use_fused_ops=False)...")
    model_base = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        hidden_size=D, depth=depth, num_heads=num_heads,
        mlp_ratio=4.0, num_calligraphers=10, num_characters=8451,
        learn_sigma=False, norm_type="rms", mlp_type="swiglu",
        qk_norm=True, rope=True, use_fused_ops=False,
        cond_inject_at="3,6,9,12,15"
    ).to(device=device)

    print("Constructing Fused 20L Model (use_fused_ops=True)...")
    model_fused = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        hidden_size=D, depth=depth, num_heads=num_heads,
        mlp_ratio=4.0, num_calligraphers=10, num_characters=8451,
        learn_sigma=False, norm_type="rms", mlp_type="swiglu",
        qk_norm=True, rope=True, use_fused_ops=True,
        cond_inject_at="3,6,9,12,15"
    ).to(device=device)

    model_fused.load_state_dict(model_base.state_dict())

    # Measure memory & latency
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def train_step_base():
        model_base.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = model_base(x, t, y_callig=y_callig, y_char=y_char, g=g)
            loss = out.sum()
        loss.backward()

    t_model_base = bench_fn(train_step_base, warmups=5, iters=20)
    mem_base = torch.cuda.max_memory_allocated() / (1024 ** 3)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def train_step_fused():
        model_fused.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = model_fused(x, t, y_callig=y_callig, y_char=y_char, g=g)
            loss = out.sum()
        loss.backward()

    t_model_fused = bench_fn(train_step_fused, warmups=5, iters=20)
    mem_fused = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"\nBaseline {depth}L (Unfused) : {t_model_base:6.2f} ms/step | {1000.0/t_model_base:5.2f} steps/s | Peak VRAM: {mem_base:.2f} GiB")
    print(f"Fused {depth}L (Triton)     : {t_model_fused:6.2f} ms/step | {1000.0/t_model_fused:5.2f} steps/s | Peak VRAM: {mem_fused:.2f} GiB")
    print(f"==> Step Latency Reduced by {(t_model_base - t_model_fused):.2f} ms ({((t_model_base - t_model_fused)/t_model_base)*100:.1f}%)")
    print(f"==> Training Throughput Boost: {((1000.0/t_model_fused) / (1000.0/t_model_base) - 1.0)*100:.1f}%!")
    print(f"==> Peak VRAM Saved: {(mem_base - mem_fused)*1024:.1f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=384)
    args = parser.parse_args()

    run_micro_benchmarks(B=args.batch, T=256, D=args.hidden)
    run_block_benchmarks(B=args.batch, T=256, D=args.hidden)
    run_full_model_benchmarks(B=args.batch, depth=args.depth, D=args.hidden)

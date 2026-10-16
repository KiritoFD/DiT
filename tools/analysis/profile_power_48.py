# -*- coding: utf-8 -*-
"""Profile sustained power, GPU utilization, and memory bandwidth on RTX 4090 (48G)."""
from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.dit import DiT_2Cond


class NvmlProfiler:
    def __init__(self):
        self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.nvml.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        assert self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) == 0

    def get_power(self) -> float:
        power = ctypes.c_uint()
        self.nvml.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power))
        return power.value / 1000.0  # W

    def shutdown(self):
        self.nvml.nvmlShutdown()


def run_power_bench(use_fused: bool, steps: int = 50, B: int = 192, depth: int = 20, D: int = 384):
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(B, 4, 32, 32, device=device)
    t = torch.rand(B, device=device)
    y_callig = torch.randint(0, 10, (B,), device=device)
    y_char = torch.randint(0, 8451, (B,), device=device)
    g = torch.randn(B, 1, 64, 64, device=device)

    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        hidden_size=D, depth=depth, num_heads=8,
        mlp_ratio=4.0, num_calligraphers=10, num_characters=8451,
        learn_sigma=False, norm_type="rms", mlp_type="swiglu",
        qk_norm=True, rope=True, use_fused_ops=use_fused,
        cond_inject_at="3,6,9,12,15"
    ).to(device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = model(x, t, y_callig=y_callig, y_char=y_char, g=g)
            loss = out.sum()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    nvml = NvmlProfiler()
    powers = []
    times = []

    t_start = time.perf_counter()
    for s in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = model(x, t, y_callig=y_callig, y_char=y_char, g=g)
            loss = out.sum()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        powers.append(nvml.get_power())

    total_time = time.perf_counter() - t_start
    nvml.shutdown()

    avg_p = sum(powers) / len(powers)
    min_p = min(powers)
    max_p = max(powers)
    avg_t = sum(times) / len(times)
    steps_s = steps / total_time

    tag = "FUSED (Triton)" if use_fused else "BASELINE (Unfused)"
    print(f"[{tag}] {steps} steps completed in {total_time:.2f}s:")
    print(f"  Step Latency : {avg_t:.2f} ms/step ({steps_s:.2f} steps/s)")
    print(f"  GPU Power    : avg={avg_p:.1f}W | min={min_p:.1f}W | max={max_p:.1f}W | range={max_p - min_p:.1f}W")
    return {"avg_t": avg_t, "avg_p": avg_p, "min_p": min_p, "max_p": max_p, "range_p": max_p - min_p}


if __name__ == "__main__":
    print("=" * 80)
    print("SUSTAINED TRAINING LOOP GPU POWER & STABILITY BENCHMARK (RTX 4090 48G)")
    print("=" * 80)
    res_base = run_power_bench(use_fused=False, steps=40)
    torch.cuda.empty_cache()
    res_fused = run_power_bench(use_fused=True, steps=40)

    print("\n" + "=" * 80)
    print("POWER & THROUGHPUT SUMMARY:")
    print("=" * 80)
    print(f"Average Power : Base={res_base['avg_p']:.1f}W -> Fused={res_fused['avg_p']:.1f}W (+{res_fused['avg_p'] - res_base['avg_p']:.1f}W)")
    print(f"Power Fluctuation Range: Base={res_base['range_p']:.1f}W -> Fused={res_fused['range_p']:.1f}W ({res_fused['range_p'] - res_base['range_p']:+.1f}W)")

import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from src.unet.io import Config, build, diffusion

def get_model_flops(config, b=2):
    model = build(config, initialize_tables=True).cuda()
    layout = torch.channels_last if config.memory_format == "channels_last" else torch.contiguous_format
    model = model.to(memory_format=layout)
    x = torch.randn(b, 4, 32, 32, device="cuda").to(memory_format=layout)
    t = torch.tensor([500.0] * b, device="cuda")
    w = torch.zeros(b, dtype=torch.long, device="cuda")
    s = torch.zeros(b, dtype=torch.long, device="cuda")
    g = torch.rand(b, 1, 64, 64, device="cuda")
    
    with FlopCounterMode(display=False) as flop_mode:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, t, w, g=g, y_script=s)
        fwd_flops = flop_mode.get_total_flops()
    
    total_step_flops_per_sample = (fwd_flops * 3.0) / b
    return fwd_flops / b, total_step_flops_per_sample

def check_graph_breaks(config):
    model = build(config, initialize_tables=True).cuda()
    b = 2
    x = torch.randn(b, 4, 32, 32, device="cuda")
    t = torch.tensor([500.0] * b, device="cuda")
    w = torch.zeros(b, dtype=torch.long, device="cuda")
    s = torch.zeros(b, dtype=torch.long, device="cuda")
    g = torch.rand(b, 1, 64, 64, device="cuda")
    
    explanation = torch._dynamo.explain(model)(x, t, w, g=g, y_script=s)
    return explanation.graph_break_count, explanation.break_reasons

def profile_run(config, batch=288, use_channels_last=True, compile_mode=None, steps=20):
    layout = torch.channels_last if use_channels_last else torch.contiguous_format
    model = build(config, initialize_tables=True).cuda().to(memory_format=layout)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, fused=True)
    process = diffusion(config)
    
    forward_fn = model
    if compile_mode:
        forward_fn = torch.compile(model, mode=compile_mode)
    
    z = torch.randn(batch, 4, 32, 32, device="cuda").to(memory_format=layout)
    g = torch.rand(batch, 1, 64, 64, device="cuda")
    w = torch.randint(0, config.num_writers, (batch,), device="cuda")
    s = torch.randint(0, config.num_scripts, (batch,), device="cuda")
    
    # Warmup
    warmup_steps = 15 if compile_mode else 3
    for i in range(warmup_steps):
        opt.zero_grad(set_to_none=True)
        t = process.sample_t(batch, z.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            terms = process.training_losses(forward_fn, z, t, model_kwargs={"g": g, "y_callig": w, "y_script": s})
            loss = terms["loss"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        t = process.sample_t(batch, z.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            terms = process.training_losses(forward_fn, z, t, model_kwargs={"g": g, "y_callig": w, "y_script": s})
            loss = terms["loss"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    step_s = steps / elapsed
    img_s = step_s * batch
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_res = torch.cuda.max_memory_reserved() / (1024**3)
    
    return {
        "batch": batch,
        "layout": "channels_last" if use_channels_last else "contiguous",
        "compile": compile_mode or "eager",
        "step_s": step_s,
        "img_s": img_s,
        "peak_alloc_GiB": peak_alloc,
        "peak_res_GiB": peak_res,
    }

def main():
    cfg = Config()
    print("=== 1. Computing Model FLOPs ===")
    fwd_flops, total_flops = get_model_flops(cfg, b=2)
    print(f"Forward FLOPs per sample: {fwd_flops / 1e9:.3f} GFLOPs")
    print(f"Total (Fwd+Bwd) FLOPs per sample: {total_flops / 1e9:.3f} GFLOPs")
    
    print("\n=== 2. Checking Inductor Graph Breaks ===")
    breaks, reasons = check_graph_breaks(cfg)
    print(f"Graph break count: {breaks}")
    if breaks > 0:
        print(f"Break reasons: {reasons}")
    else:
        print("PERFECT FULL-GRAPH CAPTURE (0 graph breaks)!")
        
    PEAK_HARDWARE_TFLOPS = 165.15 # RTX 4090 dense BF16 tensor core peak
    
    print(f"\n=== 3. Profiling Layouts, Compile & MFU (Peak = {PEAK_HARDWARE_TFLOPS:.1f} TFLOPS) ===")
    runs = [
        # (batch, channels_last, compile_mode)
        (128, False, None),
        (128, True, None),
        (288, False, None),
        (288, True, None),
        (288, True, "default"),
        (384, True, "default"),
        (512, True, "default"),
    ]
    
    for b, cl, cmp_mode in runs:
        layout_name = "CL" if cl else "NCHW"
        cmp_name = cmp_mode or "eager"
        print(f"Testing batch={b}, layout={layout_name}, compile={cmp_name} ...", end=" ", flush=True)
        res = profile_run(cfg, batch=b, use_channels_last=cl, compile_mode=cmp_mode, steps=20)
        achieved_tflops = (res["img_s"] * total_flops) / 1e12
        mfu = (achieved_tflops / PEAK_HARDWARE_TFLOPS) * 100.0
        res["achieved_TFLOPS"] = achieved_tflops
        res["MFU_percent"] = mfu
        print(f"Done! {res['step_s']:.2f} step/s | {res['img_s']:.1f} img/s | {achieved_tflops:.2f} TFLOPS | MFU: {mfu:.2f}% | VRAM: {res['peak_alloc_GiB']:.2f}G/{res['peak_res_GiB']:.2f}G")

if __name__ == "__main__":
    main()

import sys, os, time
base = "/home/ds/Workspace/moyi/ref/moyi"
sys.path.insert(0, base)
sys.path.insert(0, os.path.join(base, "moyun"))

import torch
import torch.nn as nn
from moyun_2 import DiT_models
from utils.Sampler.RF import RF

print("=" * 60)
print("【Benchmarking Moyun-12channel-B on RTX 4090】")
print("=" * 60)

device = "cuda"
model = DiT_models["moyun-12channel-B"](
    input_size=32,
    num_classes=6000,
    learn_sigma=False,
    if_rope=False
).to(device)

rf = RF(without_t=False)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

for bs in [32, 48, 64, 80, 96, 112, 128]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(bs, 12, 32, 32, device=device)
        y = torch.randint(0, 5000, (bs, 3), device=device)
        stroke = torch.zeros(bs, dtype=torch.long, device=device)
        fdict = {"calligrapher_x": 0.16, "font_x": 0.08, "charactor_x": 0.08, "_num_classes": 6000}
        
        # Warmup
        for _ in range(3):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = rf.forward(model, x, feature_dict=fdict, y=y, stroke=stroke)[0]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
        torch.cuda.synchronize()
        t0 = time.time()
        n_iters = 10
        for _ in range(n_iters):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = rf.forward(model, x, feature_dict=fdict, y=y, stroke=stroke)[0]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        
        step_s = n_iters / dt
        smp_s = step_s * bs
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Batch {bs:3d}: Mem={mem:5.2f} GB | Speed={step_s:5.2f} step/s ({smp_s:6.1f} smp/s) | Loss={loss.item():.4f}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Batch {bs:3d}: OOM")
            break
        else:
            raise e

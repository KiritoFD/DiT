import os, sys, glob, csv, time
import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

dev = th.device("cuda")

print("=================================================================")
print("【SkelNet-Sp 显存扩展基准测试 (拉满 20G VRAM)】")
print("=================================================================")

from src.model.dit import DiT_2Cond_Sp_2
model = DiT_2Cond_Sp_2(
    callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
    condition_fusion="factorized_cat", cond_fusion_norm="split",
    glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
    use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
    glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, rope_theta=100.0
).to(dev).train()

opt = th.optim.AdamW(model.parameters(), lr=3e-4)

# Test physical batch sizes on GPU
for b in [128, 256, 384, 512, 640, 768]:
    th.cuda.empty_cache()
    th.cuda.reset_peak_memory_stats()
    try:
        z = th.randn(b, 4, 32, 32, device=dev)
        t = th.full((b,), 500.0, device=dev)
        y = th.zeros(b, dtype=th.long, device=dev)
        g = th.randn(b, 4, 32, 32, device=dev)
        
        t0 = time.time()
        with th.autocast("cuda", dtype=th.bfloat16):
            v1 = model(z, t, y_callig=y, y_char=None, g=g)
            loss = v1.sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        th.cuda.synchronize()
        dt = time.time() - t0
        
        vram = th.cuda.max_memory_allocated() / (1024**3)
        sps = 1.0 / dt
        smp = b * sps
        print(f"  Physical Batch={b:3d} | Peak VRAM: {vram:5.2f} GB | Throughput: {sps:4.2f} step/s ({smp:5.1f} smp/s)")
    except Exception as e:
        print(f"  Physical Batch={b:3d} | FAILED with error: {e}")
        break

th.cuda.empty_cache()

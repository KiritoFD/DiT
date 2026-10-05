import os, sys, time
import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

dev = th.device("cuda")

print("=================================================================")
print("【1-Step Joint Training 显存扩展至 20GB 测试】")
print("=================================================================")

from src.model.dit import DiT_2Cond_Sp_2, DiT_2Cond_S_2

# 1. Stage 1 Gen (Sp/2, 65.4M)
gen = DiT_2Cond_Sp_2(
    callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
    condition_fusion="factorized_cat", cond_fusion_norm="split",
    glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
    use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
    glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, rope_theta=100.0
).to(dev).train()

# 2. Stage 2 Bak (S/2, 33.2M)
bak = DiT_2Cond_S_2(
    callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
    condition_fusion="factorized_cat", cond_fusion_norm="split",
    glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
    use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
    glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, rope_theta=100.0
).to(dev).eval()
for p in bak.parameters():
    p.requires_grad_(False)

opt = th.optim.AdamW(gen.parameters(), lr=5e-5)
TIME_SCALE = 1000.0

for b in [128, 256, 384, 512, 640, 768, 896, 1024]:
    th.cuda.empty_cache()
    th.cuda.reset_peak_memory_stats()
    try:
        x0 = th.randn(b, 4, 32, 32, device=dev)
        g_std = th.randn(b, 4, 32, 32, device=dev)
        g_gt = th.randn(b, 4, 32, 32, device=dev)
        y = th.zeros(b, dtype=th.long, device=dev)
        
        t1 = th.sigmoid(th.randn(b, device=dev))
        eps1 = th.randn_like(g_gt)
        z_t1 = (1 - t1[:, None, None, None]) * g_gt + t1[:, None, None, None] * eps1
        v1_target = eps1 - g_gt

        t2 = th.sigmoid(th.randn(b, device=dev))
        eps2 = th.randn_like(x0)
        x_t2 = (1 - t2[:, None, None, None]) * x0 + t2[:, None, None, None] * eps2
        v2_target = eps2 - x0

        t0 = time.time()
        with th.autocast("cuda", dtype=th.bfloat16):
            # Stage 1: v1 + Tweedie g_pred
            v1 = gen(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_std)
            if isinstance(v1, tuple): v1 = v1[0]
            l_sflow = th.nn.functional.mse_loss(v1.float(), v1_target)
            g_pred = z_t1 - t1[:, None, None, None] * v1
            l_smse = th.nn.functional.mse_loss(g_pred.float(), g_gt)
            
            # Stage 2: backprop through Stage 2 into Stage 1
            v2 = bak(x_t2, t2 * TIME_SCALE, y_callig=y, y_char=None, g=g_pred)
            if isinstance(v2, tuple): v2 = v2[0]
            l_img = th.nn.functional.mse_loss(v2.float(), v2_target)
            
            loss = l_img + 1.0 * l_sflow + 0.3 * l_smse

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        th.cuda.synchronize()
        dt = time.time() - t0

        vram = th.cuda.max_memory_allocated() / (1024**3)
        sps = 1.0 / dt
        smp = b * sps
        print(f"  Batch={b:4d} | Peak VRAM: {vram:5.2f} GB | Throughput: {sps:4.2f} step/s ({smp:6.1f} smp/s)")
    except Exception as e:
        print(f"  Batch={b:4d} | FAILED with error: {e}")
        break

th.cuda.empty_cache()

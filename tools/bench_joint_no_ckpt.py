import os, sys, time
import torch as th

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

dev = th.device("cuda")

print("=================================================================")
print("【1-Step 联合微调 (冻结 Stage 2 Render) 20G 显存基准测试】")
print("=================================================================")

from src.model.dit import DiT_2Cond_S_2

# 1. SkelNet (可训)
gen = DiT_2Cond_S_2(
    callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
    condition_fusion="factorized_cat", cond_fusion_norm="split",
    glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
    use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
    glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, rope_theta=100.0,
    use_checkpoint=False
).to(dev).train()

# 2. Render (冻结)
render = DiT_2Cond_S_2(
    callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
    condition_fusion="factorized_cat", cond_fusion_norm="split",
    glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
    use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
    glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, rope_theta=100.0,
    use_checkpoint=False
).to(dev).eval()

for p in render.parameters():
    p.requires_grad_(False)

opt = th.optim.AdamW(gen.parameters(), lr=2e-5)
TIME_SCALE = 1000.0

for b in [80, 96, 112, 128, 144, 160]:
    th.cuda.empty_cache()
    th.cuda.reset_peak_memory_stats()
    try:
        g_std = th.randn(b, 4, 32, 32, device=dev)
        g_gt = th.randn(b, 4, 32, 32, device=dev)
        x_gt = th.randn(b, 4, 32, 32, device=dev)
        y = th.zeros(b, dtype=th.long, device=dev)

        t1 = th.sigmoid(th.randn(b, device=dev))
        eps1 = th.randn_like(g_gt)
        z_t1 = (1 - t1[:, None, None, None]) * g_gt + t1[:, None, None, None] * eps1
        v1_target = eps1 - g_gt

        t2 = th.sigmoid(th.randn(b, device=dev))
        eps2 = th.randn_like(x_gt)
        x_t2 = (1 - t2[:, None, None, None]) * x_gt + t2[:, None, None, None] * eps2
        v2_target = eps2 - x_gt

        # Warmup
        with th.autocast("cuda", dtype=th.bfloat16):
            v1 = gen(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_std)
            if isinstance(v1, tuple): v1 = v1[0]
            l_sflow = th.nn.functional.mse_loss(v1.float(), v1_target)

            g_pred = z_t1 - t1[:, None, None, None] * v1
            l_smse = th.nn.functional.mse_loss(g_pred.float(), g_gt)

            v2 = render(x_t2, t2 * TIME_SCALE, y_callig=y, y_char=None, g=g_pred)
            if isinstance(v2, tuple): v2 = v2[0]
            l_render = th.nn.functional.mse_loss(v2.float(), v2_target)

            loss = l_render + l_sflow + 0.3 * l_smse

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        th.cuda.synchronize()

        # Timing
        t0 = time.time()
        iters = 5
        for _ in range(iters):
            with th.autocast("cuda", dtype=th.bfloat16):
                v1 = gen(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_std)
                if isinstance(v1, tuple): v1 = v1[0]
                l_sflow = th.nn.functional.mse_loss(v1.float(), v1_target)

                g_pred = z_t1 - t1[:, None, None, None] * v1
                l_smse = th.nn.functional.mse_loss(g_pred.float(), g_gt)

                v2 = render(x_t2, t2 * TIME_SCALE, y_callig=y, y_char=None, g=g_pred)
                if isinstance(v2, tuple): v2 = v2[0]
                l_render = th.nn.functional.mse_loss(v2.float(), v2_target)

                loss = l_render + l_sflow + 0.3 * l_smse

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        th.cuda.synchronize()
        dt = (time.time() - t0) / iters

        vram = th.cuda.max_memory_allocated() / (1024**3)
        sps = 1.0 / dt
        smp = b * sps
        print(f"  Batch={b:4d} | Peak VRAM: {vram:5.2f} GB | Speed: {sps:4.2f} step/s ({smp:6.1f} smp/s)")
    except Exception as e:
        print(f"  Batch={b:4d} | FAILED: {e}")
        break
    finally:
        th.cuda.empty_cache()

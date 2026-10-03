import os, sys, time
import torch as th

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

dev = th.device("cuda")

print("=================================================================")
print("【DiT-2Cond-S/2 关闭 Gradient Checkpointing 真实算力基准测试】")
print("=================================================================")

from src.model.dit import DiT_2Cond_S_2

for ckpt_flag in [True, False]:
    print(f"\n--- 测试 use_checkpoint={ckpt_flag} ---")
    model = DiT_2Cond_S_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0,
        use_checkpoint=ckpt_flag  # 测试开关差异
    ).to(dev).train()

    opt = th.optim.AdamW(model.parameters(), lr=5e-4)
    TIME_SCALE = 1000.0

    test_batches = [320, 384, 512] if not ckpt_flag else [320, 1440]
    for b in test_batches:
        th.cuda.empty_cache()
        th.cuda.reset_peak_memory_stats()
        try:
            g_skel = th.randn(b, 4, 32, 32, device=dev)
            x_gt = th.randn(b, 4, 32, 32, device=dev)
            y = th.zeros(b, dtype=th.long, device=dev)

            t1 = th.sigmoid(th.randn(b, device=dev))
            eps1 = th.randn_like(x_gt)
            z_t1 = (1 - t1[:, None, None, None]) * x_gt + t1[:, None, None, None] * eps1
            v1_target = eps1 - x_gt

            # warmup
            with th.autocast("cuda", dtype=th.bfloat16):
                v1 = model(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_skel)
                if isinstance(v1, tuple): v1 = v1[0]
                loss = th.nn.functional.mse_loss(v1.float(), v1_target)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            th.cuda.synchronize()

            # timing
            t0 = time.time()
            iters = 5
            for _ in range(iters):
                with th.autocast("cuda", dtype=th.bfloat16):
                    v1 = model(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_skel)
                    if isinstance(v1, tuple): v1 = v1[0]
                    loss = th.nn.functional.mse_loss(v1.float(), v1_target)
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

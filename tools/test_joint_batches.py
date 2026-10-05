import os, sys, time
import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

dev = th.device("cuda")

from src.model.dit import DiT_2Cond_Sp_2, DiT_2Cond_S_2
from src.eval import model_io

gen = DiT_2Cond_Sp_2(
    callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
    condition_fusion="factorized_cat", cond_fusion_norm="split",
    glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
    use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
    glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, rope_theta=100.0
).to(dev).train()

bak_ckpt = "exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt"
bak, _ = model_io.load_model_from_ckpt(bak_ckpt, device=dev, use_ema=True)
bak.eval()
for p in bak.parameters():
    p.requires_grad_(False)

opt = th.optim.AdamW(gen.parameters(), lr=2e-5)
TIME_SCALE = 1000.0

print("Testing actual Joint Training forward + backward + step with different batch sizes:")
for b in [128, 256, 384, 448, 512]:
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
            v1 = gen(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_std)
            if isinstance(v1, tuple): v1 = v1[0]
            l_sflow = th.nn.functional.mse_loss(v1.float(), v1_target)

            g_pred = z_t1 - t1[:, None, None, None] * v1
            l_smse = th.nn.functional.mse_loss(g_pred.float(), g_gt)

            v2 = bak(x_t2, t2 * TIME_SCALE, y_callig=y, y_char=None, g=g_pred)
            if isinstance(v2, tuple): v2 = v2[0]
            l_img = th.nn.functional.mse_loss(v2.float(), v2_target)

            loss = l_img + 1.0 * l_sflow + 0.3 * l_smse

        loss.backward()
        th.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        th.cuda.synchronize()
        dt = time.time() - t0

        vram = th.cuda.max_memory_allocated() / (1024**3)
        sps = 1.0 / dt
        smp = b * sps
        print(f"  Physical Batch={b:3d} | Peak VRAM: {vram:5.2f} GB | Throughput: {sps:4.2f} step/s ({smp:5.1f} smp/s)")
    except Exception as e:
        print(f"  Physical Batch={b:3d} | FAILED: {e}")
        break

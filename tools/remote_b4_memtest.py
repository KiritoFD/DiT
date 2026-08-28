#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""远程 GPU: 实测 B/4 模型在不同 batch 下的显存峰值。
逐步增大 batch, 记录峰值显存, 找到 24G 下的最大可行 batch。
"""
import os, sys, gc
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn

BASE = "/root/Workspace/xy/DiT"
sys.path.insert(0, BASE)

from models import DiT_2Cond

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda")

def build_model():
    m = DiT_2Cond(
        input_size=64, patch_size=4, in_channels=3, hidden_size=768,
        depth=12, num_heads=12, num_calligraphers=1011, num_characters=35130,
        use_checkpoint=False, condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=768,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25)
    return m.to(device)

def build_opt(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)

def run_batch(model, opt, batch_size):
    """前向+反向+优化一步, 返回峰值显存 (MB)"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        z = torch.randn(batch_size, 3, 64, 64, device=device)
        t = torch.randint(0, 1000, (batch_size,), device=device)
        y_c = torch.randint(0, 1011, (batch_size,), device=device)
        y_ch = torch.randint(0, 35130, (batch_size,), device=device)
        noise = torch.randn_like(z)
        from diffusion import create_diffusion
        diffusion = create_diffusion(timestep_respacing="")
        x0 = z
        t_clamped = t.clamp(0, 999)
        x_t = diffusion.q_sample(x0, t_clamped, noise)
        model.train()
        out = model(x_t, t_clamped, y_c, y_ch)
        # model outputs 2*in_channels (learn_sigma=True), 取前半作为 eps 预测
        eps = out[:, :3] if out.shape[1] == 6 else out
        loss = nn.functional.mse_loss(eps, noise)
        loss.backward()
        opt.step()
        opt.zero_grad()
        peak = torch.cuda.max_memory_allocated() / 1e6  # MB
        del z, t, y_c, y_ch, noise, x0, x_t, out, eps, loss
        torch.cuda.empty_cache()
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

def main():
    print("=== B/4 (hidden=768, depth=12, patch=4, 158M params) 显存实测 ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print()

    model = build_model()
    opt = build_opt(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")
    print(f"Model VRAM (params+grad+opt+ema est): {n_params*20/1e9:.2f} GB")
    print()

    print(f"{'batch':>6} {'peak MB':>10} {'peak GB':>10} {'status':>8}")
    results = []
    for bs in [256, 224, 192, 160, 128, 96, 64, 48, 32]:
        peak = run_batch(model, opt, bs)
        if peak is None:
            print(f"{bs:>6} {'OOM':>10} {'':>10} {'FAIL':>8}")
            results.append((bs, None))
            # OOM 后下一个更小的可能也 OOM, 但清缓存后继续试
            torch.cuda.empty_cache()
            gc.collect()
            continue
        gb = peak / 1e3
        ok = "OK" if peak < 23000 else "OVER"
        print(f"{bs:>6} {peak:>10.0f} {gb:>10.2f} {ok:>8}")
        results.append((bs, peak))
        if peak > 24000:
            # 超了, 后面更大的都跳
            break

    # 找最大可行 batch
    ok_batches = [(bs, mb) for bs, mb in results if mb is not None and mb < 23000]
    if ok_batches:
        max_bs, max_mb = max(ok_batches, key=lambda x: x[0])
        print(f"\n=== 最大可行 batch: {max_bs} (peak {max_mb/1e3:.2f} GB) ===")
    else:
        print(f"\n=== 所有 batch 都 OOM ===")

    # 速度估算: 跑 10 步计时
    print(f"\n=== 速度测试 ===")
    import time
    for test_bs in [32, 48, 64, 96]:
        if not any(bs == test_bs for bs, _ in ok_batches):
            continue
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            run_batch(model, opt, test_bs)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        sps = 10 / elapsed
        print(f"batch={test_bs}: {sps:.1f} steps/s, 600k步 -> {600000/sps/3600:.1f}h")

    del model, opt
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

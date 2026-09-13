#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""估算 B/4 模型在不同 batch_size 下的显存需求。
基于 s8 (S/4) 的实际显存数据反推。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── s8 实测数据 (S/4, batch=224, latent 64x64, kl-f4) ──
S4_PARAMS = 60_072_673   # 60.1M
S4_HIDDEN = 384
S4_TOKENS = 256          # 16x16
S4_BATCH = 224
S4_MEM_USED = 19.88      # GB (from log)

# B/4
B4_PARAMS = 157_600_000  # ~157.6M (手动测)
B4_HIDDEN = 768
B4_TOKENS = 256           # 16x16, same as S/4
B4_HEADS = 12

# ── 显存分解 ──
# 1. 模型参数 (fp32): params * 4 bytes
# 2. 梯度: params * 4 bytes
# 3. Adam 优化器状态 (2x params): params * 8 bytes
# 4. EMA 副本: params * 4 bytes
# 5. 激活值 (正比于 batch * tokens * hidden)
# 6. Latent 缓存 (batch * latent_channels * 64 * 64)

# 从 S/4 实测反推激活值系数
# S/4 总显存 = 参数+梯度+优化器+EMA + 激活 + latent
param_mem_s4 = S4_PARAMS * 4 / 1e9        # 0.24 GB
grad_mem_s4 = S4_PARAMS * 4 / 1e9         # 0.24 GB
opt_mem_s4 = S4_PARAMS * 8 / 1e9          # 0.48 GB
ema_mem_s4 = S4_PARAMS * 4 / 1e9          # 0.24 GB
# 注意: s8 有 DINO char table (35131, 768) = 27M params, 已计入总参数
static_mem_s4 = param_mem_s4 + grad_mem_s4 + opt_mem_s4 + ema_mem_s4
print(f"=== S/4 (当前 s8) 显存分解 ===")
print(f"  参数+梯度+优化器+EMA = {static_mem_s4:.2f} GB")
print(f"  实测总显存 = {S4_MEM_USED:.2f} GB")
print(f"  激活+latent+杂项 = {S4_MEM_USED - static_mem_s4:.2f} GB")
print(f"  batch={S4_BATCH}, tokens={S4_TOKENS}, hidden={S4_HIDDEN}")

# 激活值正比于 batch * tokens * hidden (attention + MLP)
# 激活 = k * batch * tokens * hidden
act_s4 = S4_MEM_USED - static_mem_s4
k = act_s4 / (S4_BATCH * S4_TOKENS * S4_HIDDEN)
print(f"  激活系数 k = {k:.6e} GB / (batch*token*hidden)")

# ── B/4 估算 ──
print(f"\n=== B/4 估算 ===")
param_mem_b4 = B4_PARAMS * 4 / 1e9
grad_mem_b4 = B4_PARAMS * 4 / 1e9
opt_mem_b4 = B4_PARAMS * 8 / 1e9
ema_mem_b4 = B4_PARAMS * 4 / 1e9
static_mem_b4 = param_mem_b4 + grad_mem_b4 + opt_mem_b4 + ema_mem_b4
print(f"  参数+梯度+优化器+EMA = {static_mem_b4:.2f} GB")
print(f"  tokens={B4_TOKENS}, hidden={B4_HIDDEN}")

# B/4 的激活: attention 是 hidden^2 (4x), MLP 是 hidden*4 (4x)
# 激活系数正比于 hidden (attention layer norm etc) 和 hidden^2 (qkv)
# 粗略: 激活系数 k_b4 = k_s4 * (768/384) = k * 2 (因为 hidden 翻倍)
# 更准确: attention 激活 ~ batch*tokens*hidden + batch*heads*tokens^2
#   qkv: batch*tokens*hidden*3
#   attn weights: batch*heads*tokens^2
#   MLP: batch*tokens*hidden*4
# S/4: 3*256*384 + 6*256^2 + 4*256*384 = 294K + 393K + 393K = 1.08M per sample
# B/4: 3*256*768 + 12*256^2 + 4*256*768 = 589K + 786K + 786K = 2.16M per sample
# ratio = 2.16/1.08 = 2.0
ratio = 2.0
k_b4 = k * ratio
print(f"  激活系数 (B/4) = k * {ratio} = {k_b4:.6e}")

GPU_TOTAL = 24.0  # GB, RTX 4090
GPU_SAFE = 23.0   # 留 1G 余量
print(f"\n=== B/4 不同 batch 的显存预测 ===")
print(f"{'batch':>6} {'静态':>8} {'激活':>8} {'总计':>8} {'余量':>8} {'可行?':>6}")
for bs in [256, 224, 192, 160, 128, 96, 64, 48, 32, 16]:
    act = k_b4 * bs * B4_TOKENS * B4_HIDDEN
    total = static_mem_b4 + act
    margin = GPU_TOTAL - total
    ok = "OK" if total <= GPU_SAFE else "OVER"
    print(f"{bs:>6} {static_mem_b4:>8.2f} {act:>8.2f} {total:>8.2f} {margin:>8.2f} {ok:>6}")

# ── 速度估算 ──
# S/4 batch=224: 3.5 steps/s
# B/4 计算量正比于 params * tokens (forward+backward)
# S/4: 60M * 256 = 15.4B
# B/4: 158M * 256 = 40.4B  -> 2.6x
# 但 B/4 hidden=768, attention 每层更重
# 粗略: B/4 batch=128 ~ S/4 batch=224 * (128/224) * (158/60) = 3.5 * 0.57 * 2.6 = 5.2 -> 不对
# 更准确: 速度 ~ 1 / (params * batch)
# S/4: 60M * 224 = 13.4B ops -> 3.5 steps/s
# B/4 batch=128: 158M * 128 = 20.2B ops -> 3.5 * (13.4/20.2) = 2.3 steps/s
# B/4 batch=64: 158M * 64 = 10.1B ops -> 3.5 * (13.4/10.1) = 4.6 steps/s
print(f"\n=== B/4 速度估算 ===")
S4_OPS = S4_PARAMS * S4_BATCH  # 粗略正比
print(f"{'batch':>6} {'相对计算量':>10} {'预估steps/s':>12} {'600k步耗时':>12}")
for bs in [128, 96, 64, 48, 32]:
    b4_ops = B4_PARAMS * bs
    sps = 3.5 * (S4_OPS / b4_ops)
    hours = 600000 / sps / 3600
    print(f"{bs:>6} {b4_ops/S4_OPS:>10.2f}x {sps:>12.1f} {hours:>12.1f}h")

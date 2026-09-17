# -*- coding: utf-8 -*-
"""epoch 长度 A/B: 把 epoch 从 119 步拉到 2500 步, 到底有没有收益?

背景 (用户提问): 一个 epoch 只有 28569//240 = 119 步 (~21s @5.5step/s),
每 119 步就要 reset 一次 DataLoader 迭代器。想直接拉到 2500 步。

关键事实 (v12, train.py:808-826):
  sampler  = DistributedSampler(shuffle=True)  -> randperm, 无放回, epoch=28569 样本
  loader   = DataLoader(batch_size=240, num_workers=8, pin_memory=True,
                        drop_last=True, persistent_workers=True, prefetch_factor=4)

**本脚本的 A/B 是严格受控的**: 用同一个 sampler 类, 只有 k (拼接的 epoch 数) 不同。
  k=1  -> 每 119 步 reset 一次, 跑 21 个 epoch (2499 步)
  k=21 -> 每 2499 步 reset 一次, 跑 1 个 epoch  (2499 步)
两者的**索引流完全相同** (seed+epoch*k+j: k=1 时 seeds=0..20; k=21 时 seeds=0..20),
所以统计语义不变, 唯一变量 = reset 次数 (21 vs 1)。

判据: 若 k=21 更快 -> epoch reset 真的在拖慢 (且超出 prefetch 缓冲)。
      若两者相同 -> loader 完全被 GPU 隐藏, 拉长 epoch **零收益**。
GPU 侧判据: reset 代价 R 若 < prefetch 缓冲 (4 batch x 178ms = 712ms),
            GPU 根本不会停 -> 无收益。

跑法: /opt/conda/envs/cu121/bin/python _review/bench_epoch2.py   (纯 CPU, 不需要卡空)
"""
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

N = 28569
B = 240
CUR_STEPS = N // B          # 119
NEW_STEPS = 2500
K_NEW = -(-NEW_STEPS * B // N)   # ceil(2500*240/28569) = 21

# v12 每样本 payload: latent (4,32,32) f32 + img (256,256,3) u8 + skel (4,32,32) f32
PAYLOAD = 4 * 32 * 32 * 4 + 256 * 256 * 3 + 4 * 32 * 32 * 4


class FakeDS(Dataset):
    """payload 尺寸与真实 MCCDLatentDataset(preload=True) 逐项一致。"""

    def __init__(self, n):
        rng = np.random.default_rng(0)
        self.lat = rng.standard_normal((n, 4, 32, 32)).astype(np.float32)
        self.img = rng.integers(0, 255, (n, 256, 256, 3), dtype=np.uint8)
        self.skl = rng.standard_normal((n, 4, 32, 32)).astype(np.float32)

    def __len__(self):
        return len(self.lat)

    def __getitem__(self, i):
        return {"latent": torch.from_numpy(self.lat[i]),
                "img": torch.from_numpy(self.img[i]),
                "skel_latent": torch.from_numpy(self.skl[i]),
                "y_callig": int(i % 36), "y_char": int(i % 7765)}


class ConcatSampler(DistributedSampler):
    """epoch = k 个标准 randperm epoch 的拼接。

    索引流与「k 个连续的标准 epoch」逐元素相同, 因此统计语义不变,
    唯一区别是 DataLoader 迭代器 reset 的频率 (每 k*N 次取样一次, 而非每 N 次)。
    """

    def __init__(self, dataset, k=1, num_replicas=1, rank=0, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=True, seed=seed)
        self.k = int(k)
        self.num_samples = self.num_samples * self.k
        self.total_size = self.total_size * self.k

    def __iter__(self):
        g = torch.Generator()
        out = []
        for j in range(self.k):
            g.manual_seed(self.seed + self.epoch * self.k + j)
            out.extend(torch.randperm(len(self.dataset), generator=g).tolist())
        return iter(out)


def build(ds, k):
    s = ConcatSampler(ds, k=k, num_replicas=1, rank=0, seed=0)
    ld = DataLoader(ds, batch_size=B, shuffle=False, sampler=s, num_workers=8,
                    pin_memory=True, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)
    return s, ld


def main():
    print("=" * 78)
    print("epoch 长度 A/B  (fake dataset payload=%d B/sample, %.1f KB)"
          % (PAYLOAD, PAYLOAD / 1024))
    print("N=%d  B=%d  现状 epoch=%d 步   提议 epoch=%d 步 (k=%d)"
          % (N, B, CUR_STEPS, NEW_STEPS, K_NEW))
    print("torch=%s  cpu=%d  pin_memory=True persistent_workers=True prefetch=4"
          % (torch.__version__, os.cpu_count()))
    print("=" * 78)

    ds = FakeDS(N)
    n_batches_total = (N * K_NEW) // B      # 2499

    # ---------- 1. k=1 (现状): 21 个 epoch ----------
    n_ep_base = -(-n_batches_total // CUR_STEPS)      # 21
    s1, l1 = build(ds, 1)
    s1.set_epoch(900)
    for _ in l1:                       # 预热: 让 persistent workers 起来
        pass

    resets = 0
    t0 = time.perf_counter()
    for e in range(1000, 1000 + n_ep_base):
        s1.set_epoch(e)
        for _ in l1:
            pass
        resets += 1
    t_base = time.perf_counter() - t0
    n_base = n_ep_base * CUR_STEPS

    # ---------- 2. k=21 (提议): 1 个 epoch ----------
    s2, l2 = build(ds, K_NEW)
    s2.set_epoch(900)
    for _ in l2:
        pass

    t0 = time.perf_counter()
    s2.set_epoch(1000)
    n_new = 0
    for _ in l2:
        n_new += 1
    t_new = time.perf_counter() - t0

    print("\n-- 端到端 (相同步数, 只改 reset 频率) --")
    print("  k=1  (%3d step/epoch): %5d 步 / %6.2fs -> %6.3f step/s  (%3d 次 reset)"
          % (CUR_STEPS, n_base, t_base, n_base / t_base, resets))
    print("  k=%2d (%4d step/epoch): %5d 步 / %6.2fs -> %6.3f step/s  (%3d 次 reset)"
          % (K_NEW, NEW_STEPS, n_new, t_new, n_new / t_new, 1))
    ra, rb = n_base / t_base, n_new / t_new
    print("  -> 提速 = %.4fx (%+.2f%%)" % (rb / ra, (rb / ra - 1) * 100))

    # ---------- 3. reset 代价 R ----------
    print("\n-- reset 代价 (iter(loader) -> 第 1 个 batch) --")
    s3, l3 = build(ds, 1)
    s3.set_epoch(900)
    it = iter(l3)
    next(it)
    steady = []
    for _ in range(12):
        t0 = time.perf_counter()
        next(it)
        steady.append(time.perf_counter() - t0)
    st = float(np.median(steady))
    Rs = []
    for e in range(1000, 1008):
        t0 = time.perf_counter()
        s3.set_epoch(e)
        it = iter(l3)
        next(it)
        Rs.append(time.perf_counter() - t0)
    R = float(np.median(Rs))
    print("  稳态 batch 延迟        = %7.2f ms" % (st * 1e3))
    print("  reset 后首个 batch     = %7.2f ms  (R)" % (R * 1e3))
    print("  reset 净代价           = %7.2f ms" % ((R - st) * 1e3))
    PF = 4 * 178e-3
    print("  prefetch 缓冲 = 4 x 178ms = %.0f ms" % (PF * 1e3))
    net = max(0.0, (R - st) - PF)
    print("  -> 超出缓冲的暴露代价  = %7.2f ms  %s"
          % (net * 1e3, "(GPU 会停!)" if net > 0 else "(被完全隐藏 -> 零收益)"))
    print("  -> 每步摊薄 (现状/119) = %7.3f ms  -> 占 178ms/step 的 %.2f%%"
          % (net * 1e3 / CUR_STEPS, net * 1e3 / CUR_STEPS / 178 * 100))
    print("  -> 每步摊薄 (提议/2500)= %7.3f ms" % (net * 1e3 / NEW_STEPS))
    print("  => 拉长 epoch 的理论上限收益 = %.3f ms/step (%.2f%%)"
          % (net * 1e3 * (1 / CUR_STEPS - 1 / NEW_STEPS),
             net * 1e3 * (1 / CUR_STEPS - 1 / NEW_STEPS) / 178 * 100))


main()

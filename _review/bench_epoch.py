# -*- coding: utf-8 -*-
"""epoch 拉长的 A/B: 用**真实 payload 尺寸** + 与 v12 逐项一致的 loader 参数。

v12 事实 (resolved_config + train.py:816):
  sampler 未指定 -> DistributedSampler(shuffle=True)   (randperm, 亚毫秒)
  global_batch_size=240, num_workers=8, pin_memory=True,
  drop_last=True, persistent_workers=True, prefetch_factor=4
  数据 preload 后每样本 ≈ latent 4x32x32 f32 (16KB)
                        + img 3x256x256 uint8 (192KB)
                        + skel 4x32x32 f32 (16KB)  ≈ 224KB
  -> 每 batch ≈ 240*224KB = 53.8MB, prefetch 4 -> 215MB 在途

跑法: /opt/conda/envs/cu121/bin/python /tmp/bench_epoch.py   (纯 CPU)
"""
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

N = 28569
B = 240
CUR = N // B            # 119 步/epoch (现状)
NEW = 2500              # 提议
NFE_EPOCHS = 20         # 现状跑 20 个 epoch ≈ 2380 步, 与 NEW 的 2500 步可比


class FakeDS(Dataset):
    """payload 尺寸与真实 MCCDDataset(preload) 一致。"""

    def __init__(self, n):
        rng = np.random.default_rng(0)
        self.lat = [rng.standard_normal((4, 32, 32), dtype=np.float32) for _ in range(n)]
        self.img = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(n)]
        self.skel = [rng.standard_normal((4, 32, 32), dtype=np.float32) for _ in range(n)]
        self.callig = rng.integers(0, 36, n)
        self.char = rng.integers(0, 7765, n)

    def __len__(self):
        return len(self.lat)

    def __getitem__(self, i):
        return {
            "latent": torch.from_numpy(self.lat[i]),
            "img": torch.from_numpy(self.img[i]),
            "skel_latent": torch.from_numpy(self.skel[i]),
            "y_callig": int(self.callig[i]),
            "y_char": int(self.char[i]),
        }


ds = FakeDS(N)
_payload = ds.lat[0].nbytes + ds.img[0].nbytes + ds.skel[0].nbytes
print(f"dataset N={N}, B={B}, epoch(now)={CUR} 步, epoch(new)={NEW} 步")
print(f"每样本 payload = {_payload/1024:.1f} KB -> 每 batch {_payload*B/1e6:.1f} MB "
      f"(prefetch 4 = {_payload*B*4/1e6:.0f} MB 在途)")
print(f"torch {torch.__version__}, cpu count={len(torch.get_num_threads() and [0]) and __import__('os').cpu_count()}")

# ---- 1. DistributedSampler(randperm) 的 pure-draw 代价 ----
print("\n== 1. DistributedSampler.__iter__ (v12 实际用的) ==")
for label, total in (("28569 draws (现状/epoch)", N),
                     ("600000 draws (2500 步/epoch)", NEW * B)):
    s = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    ts = []
    for e in range(5):
        s.set_epoch(e)
        t0 = time.perf_counter()
        _ = len(list(iter(s)))
        ts.append(time.perf_counter() - t0)
    print(f"  {label}: {min(ts)*1000:7.2f} ms (min of 5)")


def build(epoch_steps):
    """返回 (sampler, loader); epoch 长度由 sampler.total_size/num_samples 控制。"""
    s = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    total = epoch_steps * B
    s.num_samples = total
    s.total_size = total
    ld = DataLoader(ds, batch_size=B, shuffle=False, sampler=s, num_workers=8,
                    pin_memory=True, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)
    return s, ld


# ---- 2. 端到端 A/B: 相同步数下的每步成本 ----
print("\n== 2. 端到端 A/B (相同总步数, 只改 epoch 长度) ==")
res = {}
for label, epoch_steps, n_ep in ((f"epoch={CUR} (现状, {NFE_EPOCHS} epoch)", CUR, NFE_EPOCHS),
                                 (f"epoch={NEW} (提议, 1 epoch)", NEW, 1)):
    s, loader = build(epoch_steps)
    # 预热 (让 persistent workers 起来)
    s.set_epoch(1000)
    for _ in loader:
        pass
    t0 = time.perf_counter()
    n_steps = 0
    for e in range(2000, 2000 + n_ep):
        s.set_epoch(e)
        for batch in loader:
            n_steps += 1
    dt = time.perf_counter() - t0
    res[label] = (dt, n_steps)
    print(f"  {label}: {n_steps} 步 / {dt:.2f}s -> {n_steps/dt:.3f} step/s  "
          f"({dt/n_steps*1000:.2f} ms/step)")

if len(res) == 2:
    a, b = list(res.values())
    ra, rb = a[1] / a[0], b[1] / b[0]
    print(f"\n  提速 = {rb/ra:.4f}x  ({(rb/ra-1)*100:+.2f}%)")
    print(f"  -> 若训练 wall 的 {1/ra:.1f}s/... 按 3.1h/56k 步外推: "
          f"节省 {(1/ra - 1/rb)*56300/3600:.2f} h / 56k 步")

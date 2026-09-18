# -*- coding: utf-8 -*-
"""DataLoader 吞吐基准: preload 模式下 num_workers 的影响。

复现 MCCDLatentDataset(preload=True) 的**数据返回形态**(合成, 不碰磁盘/GPU):
  - latent       float32 (4,32,32)
  - skel_latent  float32 (4,32,32)   (skel_as_glyph_cond)
  - image        uint8   (3,256,256) (REPA on, 返回 uint8)
  - y_callig/script/char long 标量, img_id int, 其余 empty(0)
__getitem__ 从**一整块预加载 numpy** 里切片(与 preload 分支一致), 从而只测
worker IPC + default_collate 的开销 —— 正是 preload 路径下唯一与 worker 数相关的成本。

关键判据: worker=0 时主线程 collate 每批耗时 T_main, 若 T_main << 单步 DiT 时间,
则 GPU 不会被饿 -> worker=0 省掉 IPC 是净收益。
"""
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

BATCH = 360          # global_batch_size (单卡 = per-GPU)
N = 12000            # 合成样本数 (覆盖 ~33 批, 足够计时)
DEV = "cpu"          # 绝不碰 GPU


class PreloadLikeDataset(Dataset):
    def __init__(self, n):
        self.n = n
        # 一次性建好 in-RAM 大块 (模拟 _preload_all 的结果), 零拷贝切片
        self._latents = np.zeros((n, 4, 32, 32), dtype=np.float32)
        self._skel = np.zeros((n, 4, 32, 32), dtype=np.float32)
        self._imgs = np.zeros((n, 256, 256, 3), dtype=np.uint8)
        # 少量标量条件
        self._yc = (np.random.RandomState(0).randint(0, 45, n)).astype(np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "latent": torch.from_numpy(self._latents[i]),
            "skel_latent": torch.from_numpy(self._skel[i]),
            "image": torch.from_numpy(self._imgs[i]).permute(2, 0, 1),  # uint8 CHW
            "img_id": int(i),
            "y_callig": torch.tensor(self._yc[i], dtype=torch.long),
            "y_script": torch.tensor(0, dtype=torch.long),
            "y_char": torch.tensor(int(i) % 4429, dtype=torch.long),
            "g": torch.empty(0),
            "aux_latents": torch.empty(0),
            "inst_skel": torch.empty(0),
            "canny": torch.empty(0),
            "skeleton": torch.empty(0),
        }


def time_iter(num_workers, warm_batches=3, measure_batches=25):
    ds = PreloadLikeDataset(N)
    dl = DataLoader(
        ds, batch_size=BATCH, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    it = iter(dl)
    # 预热: 让 worker 池启动 / pin 线程就位, 不计入
    for _ in range(warm_batches):
        next(it)
    # 计时
    t0 = time.perf_counter()
    got = None
    for _ in range(measure_batches):
        got = next(it)
    dt = time.perf_counter() - t0
    bps = measure_batches / dt
    sps = measure_batches * BATCH / dt
    per_batch_ms = dt / measure_batches * 1000.0
    # 校验 batch 形态 (只一次)
    shapes = {k: tuple(v.shape) for k, v in got.items()
              if torch.is_tensor(v) and v.numel() > 0}
    return bps, sps, per_batch_ms, shapes


def main():
    print(f"torch={torch.__version__}  BATCH={BATCH}  N={N}  device=CPU(no GPU)")
    print("每个 worker 进程会 COW 共享主进程的 numpy; worker=0 时 collate 在主线程做。\n")
    header = f"{'num_workers':>11} | {'batch/s':>9} | {'samples/s':>10} | {'ms/batch':>9}"
    print(header)
    print("-" * len(header))
    results = {}
    for nw in [0, 2, 4, 6, 8, 12]:
        bps, sps, ms, shapes = time_iter(nw)
        results[nw] = ms
        print(f"{nw:>11} | {bps:>9.2f} | {sps:>10.1f} | {ms:>9.1f}")
    print("\nbatch 内非空张量形状 (校验与 train 期望一致):", shapes)

    # 与当前 DiT 步时对比: 让用户填 / 或从日志估。这里给出 worker=0 相对 worker=8 的 ms/batch
    w0, w8 = results[0], results[8]
    print(f"\n[结论辅助] worker=0 每批 {w0:.1f}ms, worker=8 每批 {w8:.1f}ms, "
          f"差 {w0-w8:+.1f}ms/批。")
    print("若当前 DiT 每步耗时 > max(w0, w8)，则两者都不会饿到 GPU，"
          "此时选 IPC 更低(吞吐更高)的一侧即为净收益。")


if __name__ == "__main__":
    main()

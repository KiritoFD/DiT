# -*- coding: utf-8 -*-
"""fast_image_pipeline.py — 图 -> GPU 的高吞吐数据管线 (给 DINO 抽取与 RGB 编码器共用)。

解决"nvidia-smi 里 util 0% 尖刺 / 功率上不去"的根因: 原来每批是
  **读图(CPU) -> 传(H2D) -> 前向(GPU)** 串成一条链, GPU 在等 CPU。
这里改成生产者/消费者 + pinned 预取, 三件事同时发生:
  [线程A] 读图(多线程 cv2, 每个 worker 关掉 cv2 自己的线程池) -> 写 pinned 缓冲
  [流B ] pinned -> GPU 异步拷贝 (h2d stream)
  [主流] 前向/反传

对外只暴露 FastImageFeeder.iter_batches(), 用法:

    feeder = FastImageFeeder(paths, size=256, gray=True, batch=1536, workers=32)
    for x_uint8_gpu, s in feeder:        # x 已在显存, uint8 (B,1 or 3,H,W)
        ...                              # 主流已 wait_event, 直接可用
"""
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch


def _worker_load(args):
    """单张读图 -> uint8 (C,H,W)。cv2 线程池必须关掉, 否则和 Python 线程池互抢核。"""
    path, size, gray, channels = args
    cv2.setNumThreads(1)
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if g is None:
        raise RuntimeError(f"读图失败: {path}")
    if g.shape[0] != size or g.shape[1] != size:
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(g))
    if channels == 1:
        return t.unsqueeze(0)
    return t.unsqueeze(0).repeat(3, 1, 1)          # 灰度 -> 3 通道 (dim0 是通道)


class FastImageFeeder:
    """pinned 双/三缓冲 + 后台读图线程。每批返回 (显存上的 uint8 张量, 起始下标)。"""

    def __init__(self, paths, size=256, gray=True, channels=3, batch=1536,
                 workers=32, buffers=3, prefetch_threads=0, verbose=True):
        self.paths = paths
        self.size = size
        self.gray = gray
        self.channels = channels
        self.batch = batch
        self.workers = workers
        self.buffers = buffers
        self.verbose = verbose
        self.n = len(paths)

    def __len__(self):
        return (self.n + self.batch - 1) // self.batch

    def __iter__(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        shape = (self.batch, self.channels, self.size, self.size)
        # ★ 槽位化预分配: pinned 与 **GPU** 缓冲都只分配一次, 每步只改切片。
        #   原实现每批都 buf[:n].to(dev) 新分配显存 -> 显存随时间单调上涨。
        slots = list(range(self.buffers))
        pin = [torch.empty(shape, dtype=torch.uint8, pin_memory=True) for _ in slots]
        gbuf = [torch.empty(shape, dtype=torch.uint8, device=dev) for _ in slots]
        evs = [torch.cuda.Event() for _ in slots]
        free = Queue()
        for i in slots:
            free.put(i)
        ready = Queue(maxsize=self.buffers)
        pool = ThreadPoolExecutor(self.workers)
        h2d = torch.cuda.Stream() if dev == "cuda" else None

        def producer():
            for k in range(0, self.n, self.batch):
                chunk = self.paths[k:k + self.batch]
                n_ok = len(chunk)
                i = free.get()
                imgs = list(pool.map(
                    _worker_load,
                    [(p, self.size, self.gray, self.channels) for p in chunk]))
                pin[i][:n_ok].copy_(torch.stack(imgs))
                if h2d is not None:
                    with torch.cuda.stream(h2d):
                        gbuf[i][:n_ok].copy_(pin[i][:n_ok], non_blocking=True)
                        evs[i].record(h2d)
                ready.put((k, n_ok, i))
            ready.put(None)

        th = Thread(target=producer, daemon=True)
        th.start()

        while True:
            item = ready.get()
            if item is None:
                break
            k, n_ok, i = item
            if h2d is not None:
                torch.cuda.current_stream().wait_event(evs[i])
            yield gbuf[i][:n_ok], k
            free.put(i)

    @staticmethod
    def to_float01(x):
        """uint8 显存张量 -> [0,1] float (在 GPU 上做, 不占 PCIe 带宽)。"""
        return x.float().div_(255.0)

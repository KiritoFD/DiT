# -*- coding: utf-8 -*-
"""
dino_cache.py — REPA teacher (DINOv2) 特征离线缓存。

动机: 训练每个 step 都对 batch GT 图跑一次 DINOv2 ViT-S/14 前向 (~40ms/step,
400k 步累计 76.8M 次冗余前向)。GT 图像集合固定 (csv 每行一个文件), 教师特征
只依赖图像 → 可离线预提取一次, 训练时查表。

磁盘格式 (cache_dir):
    feats.f16   — (N, P, D) float16 连续二进制 (np.memmap 可读)
    ids.npy     — (N,) int64 img_id (与 feats 行序一一对应)
    meta.json   — {"n": N, "patches": P, "dim": D, "backbone": ...}

典型体积: 85k 张 x 256 patch x 384 dim x 2B ≈ 16.7 GB (随机读走 OS page cache,
不需要全部驻留 RAM)。
"""
import json
import os

import numpy as np
import torch


class DinoFeatureCache:
    """img_id → DINOv2 patch-token (256,384) 查表 (memmap, 随机访问)。

    gather(img_ids) 返回 (feats_fp32[B,P,D], missing_positions list[int]);
    missing 行填 0, 调用方 (REPALoss) 用 teacher 前向兜底。
    """

    def __init__(self, cache_dir, dtype=torch.float32):
        self.cache_dir = cache_dir
        self.feats_path = os.path.join(cache_dir, "feats.f16")
        self.ids_path = os.path.join(cache_dir, "ids.npy")
        self.meta_path = os.path.join(cache_dir, "meta.json")
        if not (os.path.isfile(self.feats_path) and os.path.isfile(self.ids_path)):
            raise FileNotFoundError(
                f"[dino-cache] cache incomplete in {cache_dir!r}: need feats.f16 + ids.npy "
                f"(build with tools/build_dino_cache.py)")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.patches = int(self.meta["patches"])
        self.dim = int(self.meta["dim"])
        self.dtype = dtype
        self._feats = np.memmap(self.feats_path, dtype=np.float16, mode="r")
        if self._feats.size % (self.patches * self.dim) != 0:
            raise ValueError(f"[dino-cache] feats.f16 size {self._feats.size} is not a "
                             f"multiple of {self.patches}x{self.dim}")
        self._feats = self._feats.reshape(-1, self.patches, self.dim)
        self._ids = np.load(self.ids_path)
        self._id2row = {int(i): r for r, i in enumerate(self._ids)}
        if len(self._id2row) != len(self._ids):
            raise ValueError(f"[dino-cache] duplicate img_ids in {self.ids_path}")

    def __len__(self):
        return len(self._ids)

    def has(self, img_id):
        return int(img_id) in self._id2row

    def coverage(self, img_ids):
        hit = sum(1 for i in img_ids if int(i) in self._id2row)
        return hit, len(img_ids)

    def stats(self):
        return (f"cache={self.cache_dir} n={len(self)} patches={self.patches} "
                f"dim={self.dim} ({os.path.getsize(self.feats_path) / 2**30:.1f} GiB)")

    @torch.no_grad()
    def gather(self, img_ids, device="cpu"):
        """img_ids: 任何可迭代的 int (tensor/list)。返回 (feats(B,P,D) on device, missing 位置列表)。"""
        rows = torch.empty(len(img_ids), dtype=torch.long)
        missing = []
        for k, i in enumerate(img_ids):
            r = self._id2row.get(int(i))
            if r is None:
                missing.append(k)
                rows[k] = 0
            else:
                rows[k] = r
        arr = self._feats[rows.numpy()]          # (B,P,D) fp16 fancy-index copy
        feats = torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device=device, dtype=self.dtype)
        return feats, missing

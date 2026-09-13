# -*- coding: utf-8 -*-
"""
dino_cache.py — REPA teacher (DINOv2) 特征离线缓存 (fp32, 常驻 pinned RAM)。

磁盘格式 (cache_dir):
    feats.f32  — (N, P, D) float32 (首选; 由 tools/convert_dino_cache_f32.py 从 f16 转换)
    feats.f16  — (N, P, D) float16 (fallback)
    ids.npy    — (N,) int64 img_id (与 feats 行序一一对应)
    meta.json  — {"n", "patches", "dim", "backbone", "csv"}

运行时: 全 cache 一次性载入 pinned RAM 常驻 (f32=31.2GB, f16=15.6GB)。
gather = 向量化 searchsorted 查行号 -> index_select (pinned) -> async H2D。
无 page-fault、无 CPU 端 dtype 转换、无同步等待 -> 训练步内 ~10ms (75MB PCIe)。
"""
import json
import os

import numpy as np
import torch


class DinoFeatureCache:
    """img_id → DINOv2 patch-token (P,D) 查表 (fp32 常驻 pinned RAM)。"""

    def __init__(self, cache_dir, preload="pinned"):
        """
        preload="pinned": 全 cache 常驻 pinned RAM (零 page-fault, async H2D)。
        RAM 不足或显式 preload="memmap" 时 fallback 磁盘 memmap (慢 ~77ms/step)。
        """
        self.cache_dir = cache_dir
        self.feats_path = os.path.join(cache_dir, "feats.f32")
        if not os.path.isfile(self.feats_path):
            self.feats_path = os.path.join(cache_dir, "feats.f16")
        self.ids_path = os.path.join(cache_dir, "ids.npy")
        self.meta_path = os.path.join(cache_dir, "meta.json")
        if not (os.path.isfile(self.feats_path) and os.path.isfile(self.ids_path)):
            raise FileNotFoundError(
                f"[dino-cache] cache incomplete in {cache_dir!r}: need feats.f32|feats.f16 + ids.npy "
                f"(build with tools/build_dino_cache.py)")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.patches = int(self.meta["patches"])
        self.dim = int(self.meta["dim"])
        self._feats = None
        self._feats_mm = None
        self._out_buf = None   # 预分配 pinned 输出缓冲 (index_select out= 避免 pageable 慢路径)

        _np_dtype = np.float32 if self.feats_path.endswith(".f32") else np.float16
        _mm = np.memmap(self.feats_path, dtype=_np_dtype, mode="r")
        if _mm.size % (self.patches * self.dim) != 0:
            raise ValueError(f"[dino-cache] feats size {_mm.size} is not a "
                             f"multiple of {self.patches}x{self.dim}")
        _mm = _mm.reshape(-1, self.patches, self.dim)

        self._ids = np.load(self.ids_path)
        if len(np.unique(self._ids)) != len(self._ids):
            raise ValueError(f"[dino-cache] duplicate img_ids in {self.ids_path}")
        # 向量化查表: sorted ids + argsort 行号 (searchsorted O(log N)/row, 无 python 循环)
        self._sorted_rows = np.argsort(self._ids)
        self._sorted_ids = self._ids[self._sorted_rows]

        if preload == "pinned":
            try:
                _t = torch.from_numpy(np.array(_mm))     # 全量 RAM copy (fp32=31.2GB)
                self._feats = _t.pin_memory()             # 常驻 pinned RAM
                del _t
            except Exception as _e:
                print(f"[dino-cache] pinned preload failed ({_e!r}) -> memmap fallback")
                self._feats = None
        if self._feats is None:
            self._feats_mm = _mm
        self.mode = "pinned" if self._feats is not None else "memmap"

    def __len__(self):
        return len(self._ids)

    def has(self, img_id):
        pos = np.searchsorted(self._sorted_ids, np.int64(img_id))
        return pos < len(self._sorted_ids) and self._sorted_ids[pos] == np.int64(img_id)

    def coverage(self, img_ids):
        q = np.asarray([int(i) for i in img_ids], dtype=np.int64)
        pos = np.searchsorted(self._sorted_ids, q)
        valid = pos < len(self._sorted_ids)
        valid[valid] = self._sorted_ids[pos[valid]] == q[valid]
        return int(valid.sum()), len(q)

    def stats(self):
        return (f"cache={self.cache_dir} mode={self.mode} n={len(self)} "
                f"patches={self.patches} dim={self.dim} "
                f"dtype={'fp32' if self.feats_path.endswith('.f32') else 'fp16'} "
                f"({os.path.getsize(self.feats_path) / 2**30:.1f} GiB)")

    @torch.no_grad()
    def gather(self, img_ids, device="cpu"):
        """img_ids: tensor/list of int。返回 (feats(B,P,D) fp32 on device, missing 位置列表)。

        全向量化: searchsorted 行号映射 -> index_select from pinned fp32 ->
        non_blocking H2D。无 CPU dtype 转换, 无 page-fault, 无同步。
        """
        q = np.asarray([int(i) for i in img_ids], dtype=np.int64)
        pos = np.searchsorted(self._sorted_ids, q)
        valid = (pos < len(self._sorted_ids))
        valid[valid] = self._sorted_ids[pos[valid]] == q[valid]
        rows = np.where(valid, self._sorted_rows[np.clip(pos, 0, len(self._sorted_ids) - 1)], 0)
        missing = np.nonzero(~valid)[0].tolist()
        rows_t = torch.from_numpy(rows)
        if self._feats is not None:
            # 预分配 pinned 输出缓冲 + index_select(out=) -> 8ms (默认 pageable 路径 40ms)
            if self._out_buf is None or self._out_buf.shape[0] < len(rows_t):
                self._out_buf = torch.empty(len(rows_t), self.patches, self.dim,
                                            dtype=self._feats.dtype, pin_memory=True)
            buf = self._out_buf[:len(rows_t)]
            torch.index_select(self._feats, 0, rows_t, out=buf)
            out = buf.to(device=device)   # 同步 H2D 75MB ~6ms (non_blocking 有 buf 复用竞态)
        else:
            out = torch.from_numpy(np.ascontiguousarray(self._feats_mm[rows])).to(device=device).float()
        return out, missing

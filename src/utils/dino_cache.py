# -*- coding: utf-8 -*-
"""
dino_cache.py — REPA teacher (DINOv2) 特征离线缓存（常驻 pinned RAM ✓）。

磁盘格式 (cache_dir):
    feats.f32  — (N, P, D) float32 (首选; 由 tools/convert_dino_cache_f32.py 从 f16 转换)
    feats.f16  — (N, P, D) float16 (备选)
    ids.npy    — (N,) int64 img_id (与 feats 行序一一对应)
    meta.json  — {"n", "patches", "dim", "backbone", "csv"}

★ 2026-10-14 [重构 PR2] 修正**文档与实现不符** ✗（与 P1-4/P1-5 同类缺陷 ✓）:
    旧文档称 "gather = … -> **async H2D**"、"**无同步等待**" ✗
    但实现（`gather` 末段）是 **同步 H2D** ✓ —— 代码里的行内注释自己就写着
    "同步 H2D 75MB ~6ms（**non_blocking 有 buf 复用竞态**）" ✓✓
    ⇒ 文档已改为**如实描述** ✓；并注明真正的异步需要**双缓冲 / CUDA Event** ✓
      （审查 §7.1 明确警告: **不能只打开 non_blocking 就复用同一块 pinned buffer** ✗，
        否则是难以复现的数据竞态 ✗）。

★ 2026-10-14 [重构 PR2] 消除**静默 fallback** ✗（PR2 完成标准: "无静默 fallback" ✓）:
    · 旧实现: pinned 预载失败 ⇒ 只 print 一句就退到 memmap ✗（静默降速 ~10× ✗）
    · 新语义: `strict=True`（默认 ✓）⇒ **直接报错** ✗；
              `preload="memmap"` 是**显式**选择慢路径 ✓（不算 fallback ✓）
    · 并修掉一个**静默 dtype 不一致** ✗: 旧实现 pinned 路径**不做 .float()**（返回磁盘
      dtype ✗）而 memmap 路径做 `.float()`（返回 fp32 ✓）⇒ 同一 cache 因内存是否够
      而返回**不同 dtype** ✗✗。新语义: **返回一律 fp32** ✓（与 REPA loss 的期望一致 ✓）。

可选内存优化（**默认关闭** ✓ 需实测后开启 ✓ 审查 §7.1 ✓）:
    `ram_dtype="fp16"` ⇒ 常驻副本用 fp16（31.2GB -> 15.6GB ✓ 显存/内存减半 ✓），
    gather 时在 H2D 拷贝中**升回 fp32** ✓（数值影响须先对比 ✓ 不得默认切换 ✗）。
"""
import json
import os

import numpy as np
import torch


class DinoFeatureCache:
    """img_id → DINOv2 patch-token (P,D) 查表（常驻 pinned RAM ✓ 返回 fp32 ✓）。"""

    def __init__(self, cache_dir, preload="pinned", *, ram_dtype=None, strict=True,
                 verbose=True):
        """
        preload : "pinned"（默认 ✓ 全 cache 常驻 pinned RAM ✓）
                  "memmap"（**显式**选慢路径 ✓ 约 77ms/step ✓ 不算 fallback ✓）
        ram_dtype: None（跟磁盘一致 ✓ 默认 ✓）| "fp32" | "fp16"（常驻副本减半 ✓ 需实测 ✓）
        strict  : True（默认 ✓）—— pinned 预载失败 ⇒ **报错** ✗（PR2: 无静默 fallback ✗）
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
        self.verbose = bool(verbose)

        # ── 磁盘 dtype（显式记录 ✓ 不再"看路径猜" ✗）──
        self.disk_dtype = "fp32" if self.feats_path.endswith(".f32") else "fp16"
        _np_dtype = np.float32 if self.disk_dtype == "fp32" else np.float16
        # ── 常驻 RAM dtype（默认同磁盘 ✓ 可显式选 fp16 省一半内存 ✓）──
        self.ram_dtype = str(ram_dtype or self.disk_dtype).lower()
        if self.ram_dtype not in ("fp32", "fp16"):
            raise ValueError(f"[dino-cache] ram_dtype 必须是 fp32|fp16，收到 {ram_dtype!r} ✗")

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

        if preload not in ("pinned", "memmap"):
            raise ValueError(f"[dino-cache] preload 必须是 pinned|memmap，收到 {preload!r} ✗")
        if preload == "pinned":
            _ram_np = np.float32 if self.ram_dtype == "fp32" else np.float16
            try:
                # 注意: np.array(_mm) 已经是全量 RAM copy ✓（f32=31.2GB / f16=15.6GB ✓）
                _arr = np.array(_mm)
                if _arr.dtype != _ram_np:
                    _arr = _arr.astype(_ram_np, copy=False)
                _t = torch.from_numpy(np.ascontiguousarray(_arr))
                self._feats = _t.pin_memory()             # 常驻 pinned RAM
                del _t, _arr
            except Exception as _e:                        # noqa: BLE001
                # ★ PR2: 默认**不静默降级** ✗ —— 内存不足必须让调用方知道 ✓
                if strict:
                    raise RuntimeError(
                        f"[dino-cache] pinned 预载失败（{type(_e).__name__}: {_e}）✗ —— "
                        f"strict=True 时**拒绝静默降级到 memmap** ✗（慢 ~10× ✗）。"
                        f"请显式传 preload='memmap'（知情选择 ✓）或 ram_dtype='fp16'"
                        f"（常驻减半 ✓ 需先实测数值影响 ✓）") from _e
                if self.verbose:
                    print(f"[dino-cache] ⚠ pinned 预载失败（{_e!r}）⇒ **显式允许的** memmap "
                          f"回退 ✓（strict=False ✓ 慢 ~10× ✗）", flush=True)
                self._feats = None
        if self._feats is None:
            self._feats_mm = _mm
        self.mode = "pinned" if self._feats is not None else "memmap"
        if self.verbose:
            print(f"[dino-cache] {self.stats()}", flush=True)

    def __len__(self):
        return len(self._ids)

    def has(self, img_id):
        pos = np.searchsorted(self._sorted_ids, np.int64(img_id))
        return pos < len(self._sorted_ids) and self._sorted_ids[pos] == np.int64(img_id)

    def _valid_mask(self, q):
        pos = np.searchsorted(self._sorted_ids, q)
        valid = pos < len(self._sorted_ids)
        valid[valid] = self._sorted_ids[pos[valid]] == q[valid]
        return pos, valid

    def coverage(self, img_ids):
        """返回 (命中数, 总数) ✓（非破坏性查询 ✓ 不报错 ✓）。"""
        q = np.asarray([int(i) for i in img_ids], dtype=np.int64)
        _pos, valid = self._valid_mask(q)
        return int(valid.sum()), len(q)

    def require_coverage(self, img_ids, what="REPA 特征", raise_on_missing=True):
        """★ PR2 ✓: **覆盖率契约** —— 缺任何一条即**报错** ✗（不再静默用第 0 行 ✗）。

        REPA 是**监督信号** ✓: 缺特征却继续训练 = 用**别的图**的特征当老师 ✗✗
        （与 P0-3 的 "clip 到第 0 行" 同类 ✗）。返回命中数 ✓。
        """
        q = np.asarray([int(i) for i in img_ids], dtype=np.int64)
        _pos, valid = self._valid_mask(q)
        n_hit = int(valid.sum())
        if n_hit != len(q) and raise_on_missing:
            bad = q[~valid]
            raise KeyError(
                f"[dino-cache] {what} 覆盖率不足: {n_hit}/{len(q)} ✓✗ —— 缺失 {len(bad)} 条 "
                f"（示例 {bad[:8].tolist()}）✗ cache={self.cache_dir}。"
                f"REPA 是监督信号, 缺行会喂**别的图**的特征 ✗ ⇒ 请重建 cache "
                f"（tools/build_dino_cache.py / tools/dino_expand_*.py）")
        return n_hit

    def stats(self):
        return (f"cache={self.cache_dir} mode={self.mode} n={len(self)} "
                f"patches={self.patches} dim={self.dim} "
                f"disk={self.disk_dtype} ram={self.ram_dtype if self.mode == 'pinned' else '-'} "
                f"out=fp32 ({os.path.getsize(self.feats_path) / 2**30:.1f} GiB)")

    @torch.no_grad()
    def gather(self, img_ids, device="cpu"):
        """img_ids: tensor/list of int。返回 (feats(B,P,D) **fp32** on device, missing 位置列表)。

        实现: searchsorted 行号映射 -> index_select（pinned ✓ 预分配 out= ✓）-> **同步** H2D ✓。
        无 page-fault ✓ 无 CPU 端 dtype 转换 ✓；**不是** async H2D ✗（见模块文档 ✓）。
        返回 dtype **恒为 fp32** ✓（pinned/memmap 两条路径一致 ✓ 修掉旧的静默不一致 ✗）。
        """
        q = np.asarray([int(i) for i in img_ids], dtype=np.int64)
        pos, valid = self._valid_mask(q)
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
            # 同步 H2D（约 75MB/6ms）—— non_blocking 会与 buf 复用产生竞态 ✗
            # 真异步需要**双缓冲 / CUDA Event**（审查 §7.1 ✓）⇒ 留待实测确认是瓶颈后再做 ✓
            out = buf.to(device=device, dtype=torch.float32)
        else:
            out = torch.from_numpy(np.ascontiguousarray(self._feats_mm[rows])).to(
                device=device, dtype=torch.float32)
        return out, missing

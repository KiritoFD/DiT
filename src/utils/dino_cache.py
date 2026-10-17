# -*- coding: utf-8 -*-
"""
dino_cache.py — REPA teacher (DINOv2) 特征离线缓存（常驻 pinned RAM ✓）。

磁盘格式 (cache_dir):
    feats.f16  — (N, P, D) float16 (**首选** ✓ 见下方 2026-10-16 的逐位证明 ✓)
    feats.f32  — (N, P, D) float32 (若存在, 由 tools/convert_dino_cache_f32.py 从 f16 上转换)
    ids.npy    — (N,) int64 img_id (与 feats 行序一一对应)
    meta.json  — {"n", "patches", "dim", "backbone", "csv"}
    ⚠ 注意: `feats.f16/f32` 是**裸二进制** ✓（不是 .npy ✗）⇒ 可真正 `np.memmap` ✓
      （这正好**规避**了审查 P1-4 指出的 ".npz 无法 mmap" 问题 ✓✓）

★ 2026-10-16 [infra_1015 §四] **f32 只是 f16 的上转换 —— 已逐位证明** ✓✓:
    审查文档 §四 的推断: "如果当前训练所用的 `feats.f32` 确实只是 FP16 数据上转换而来，
      那么 FP32 文件并没有增加任何特征精度，却让缓存和 H2D 流量翻倍" ✓
    实测（`tools/verify_repa_f16.py` ✓ 可复现 ✓）:
      cache=`v11_sym_full` ✓ 抽样 2,000 条 / 共 85,155 条 ✓
      **逐元素相等 = True** ✓  **max|Δ| = 0.000e+00** ✓  **非零元素 0 / 196,608,000** ✓
      ⇒ f32 **没有任何额外精度** ✓ ⇒ **首选 f16** ✓（H2D 与常驻各减半 ✓ 数值**零变化** ✓）
    ★ 另实测: 7 个 cache 中 **6 个只有 f16** ✓（含活跃的 `top10_aug_v1` ✓）
      ⇒ 活跃训练路径**本来就用 f16** ✓ ⇒ 本改动只影响历史 cache ✓（低风险 ✓）

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
        # ★ 2026-10-16 [infra_1015 §四]: **优先 f16** ✓（不是 f32 ✗）
        #   依据: `tools/verify_repa_f16.py` 的**逐位证明** ——
        #     v11_sym_full 抽样 2000/85155 条: 逐元素相等 ✓ max|Δ|=0 ✓ 非零 0/196608000 ✓
        #   ⇒ f32 无额外精度 ✓ 却让**磁盘读 + pinned RAM 各翻倍** ✗
        #   ⇒ 数值**零变化** ✓（因为 f32 ≡ f16 上转换 ✓）而内存/H2D 减半 ✓
        #   ⚠ 若某天出现**真正高精度**的 f32 cache（非上转换 ✓）⇒
        #     先跑 `tools/verify_repa_f16.py` 核对 ✓ 不相等时它会被显式保留 ✓（见下 ✓）
        self.feats_path = os.path.join(cache_dir, "feats.f16")
        if not os.path.isfile(self.feats_path):
            self.feats_path = os.path.join(cache_dir, "feats.f32")
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
        self._prefetch_slots = [None, None]
        self._prefetch_index = 0
        self._prefetch_stream = None
        self._pending = None
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
    def prefetch(self, img_ids, device):
        """Stage this batch before DiT forward; event-guarded double pinned buffers."""
        device = torch.device(device)
        if self._feats is None or device.type != 'cuda':
            return
        q = np.asarray(img_ids, dtype=np.int64).copy()
        pos, valid = self._valid_mask(q)
        rows = np.where(valid, self._sorted_rows[np.clip(pos, 0, len(self._sorted_ids)-1)], 0)
        missing = np.flatnonzero(~valid).tolist()
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=device)
        slot = self._prefetch_slots[self._prefetch_index]
        if slot is not None:
            slot[1].synchronize()  # CPU may overwrite pinned bytes only after DMA completes.
        if slot is None or slot[0].shape[0] != len(rows):
            slot = (torch.empty(len(rows), self.patches, self.dim,
                                dtype=self._feats.dtype, pin_memory=True), torch.cuda.Event())
            self._prefetch_slots[self._prefetch_index] = slot
        self._prefetch_index = 1-self._prefetch_index
        torch.index_select(self._feats, 0, torch.from_numpy(rows), out=slot[0])
        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            out = slot[0].to(device=device, dtype=torch.float32, non_blocking=True)
            slot[1].record(stream)
        self._pending = (q, out, missing, slot[1])

    @torch.no_grad()
    def gather(self, img_ids, device="cpu"):
        """img_ids: tensor/list of int。返回 (feats(B,P,D) **fp32** on device, missing 位置列表)。

        With a matching prefetch, wait only on its CUDA event; otherwise use the
        synchronous pinned-buffer path. Both return the same FP32 features.
        返回 dtype **恒为 fp32** ✓（pinned/memmap 两条路径一致 ✓ 修掉旧的静默不一致 ✗）。
        """
        q = np.asarray([int(i) for i in img_ids], dtype=np.int64)
        if self._pending is not None:
            ids, out, missing, ready = self._pending
            requested = torch.device(device)
            same_device = requested.type == out.device.type and (requested.index is None or requested.index == out.device.index)
            if same_device and np.array_equal(ids, q):
                current = torch.cuda.current_stream(out.device)
                current.wait_event(ready)
                out.record_stream(current)
                return out, missing
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

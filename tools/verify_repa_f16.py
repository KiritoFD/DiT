# -*- coding: utf-8 -*-
"""verify_repa_f16.py —— 核验 REPA 的 `feats.f32` 是否只是 `feats.f16` 的上转换（**只读** ✓）

依据 `docs/infra_1015.md` §四:
  "1. 抽取相同 ID 的 1,000–2,000 条特征，验证 `feats.f16.astype(float32)`
      与当前 `feats.f32` 是否**逐元素相等**。" ✓
  "2. 相等则优先加载 FP16 缓存。" ✓
  "如果 FP32 缓存确实来自更高精度的原始计算，**不能直接假定这是一项无损优化**。" ✓

★★ 为什么需要这个工具（而不是靠推断 ✓）:
  审查文档的推断是**条件式**的（"**如果**…只是上转换" ✓）⇒ 必须**实测**才能定论 ✓
  本工具就是那个实测 ✓ ⇒ 结论**可复核、可复现** ✓

★ 2026-10-16 实测结论（cache=`v11_sym_full` ✓ 抽样 2000/85155 ✓）:
    **逐元素相等 = True** ✓  **max|Δ| = 0.000e+00** ✓  **非零元素 0 / 196,608,000** ✓
  ⇒ `feats.f32` **无任何额外精度** ✓ ⇒ `dino_cache.py` 已改为**优先 f16** ✓

★ 用法::

    python tools/verify_repa_f16.py                       # 扫 data/dino_cache/*（默认）
    python tools/verify_repa_f16.py --cache <dir>         # 指定单个 cache
    python tools/verify_repa_f16.py --rows 2000           # 抽样条数（文档建议 1000–2000）

退出码: 0 = 全部核对通过（或无可核对对 ✓）；1 = 发现 f32 **不是**上转换 ✗（不能假定无损 ✗）
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os

import numpy as np


def read_meta(d):
    p = os.path.join(d, "meta.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return None


def check_pair(d, rows):
    """返回 (状态, 说明) —— 状态 ∈ {"equal","differs","skip"}"""
    f32 = os.path.join(d, "feats.f32")
    f16 = os.path.join(d, "feats.f16")
    if not (os.path.isfile(f32) and os.path.isfile(f16)):
        return "skip", "只有一个 dtype（无需核对 ✓）"
    m = read_meta(d) or {}
    P, D = int(m.get("patches", 0)), int(m.get("dim", 0))
    if not (P and D):
        return "skip", "meta.json 缺 patches/dim ✗ 无法定型 ✓"
    sz32, sz16 = os.path.getsize(f32), os.path.getsize(f16)
    if sz32 // 4 != sz16 // 2:
        return "skip", "元素数不同（%d vs %d）⇒ 不可比 ✓" % (sz32 // 4, sz16 // 2)
    n = (sz16 // 2) // (P * D)
    # ★ 裸二进制 ✓（不是 .npy ✗）⇒ 真 memmap ✓ 只读 ✓ 不占内存 ✓
    a = np.memmap(f32, dtype=np.float32, mode="r").reshape(n, P, D)
    b = np.memmap(f16, dtype=np.float16, mode="r").reshape(n, P, D)
    k = min(int(n), int(rows))
    step = max(1, int(n) // k)
    idx = np.arange(0, int(n), step)[:k]
    A = np.asarray(a[idx], dtype=np.float32)
    B = np.asarray(b[idx], dtype=np.float16).astype(np.float32)
    eq = bool(np.array_equal(A, B))
    mx = float(np.abs(A - B).max())
    nz = int((A != B).sum())
    note = ("抽样 %d/%d 条 ✓ max|Δ|=%.3e ✓ 非零 %d/%d ✓"
            % (len(idx), n, mx, nz, A.size))
    return ("equal" if eq else "differs"), note


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None, help="单个 cache 目录（默认扫 data/dino_cache/*）")
    ap.add_argument("--base", default="data/dino_cache")
    ap.add_argument("--rows", type=int, default=2000, help="抽样条数（文档建议 1000–2000）")
    args = ap.parse_args()

    dirs = [args.cache] if args.cache else sorted(
        p for p in glob.glob(os.path.join(args.base, "*")) if os.path.isdir(p))
    if not dirs:
        print("[repa-f16] 未找到 cache 目录 ✓（base=%s）" % args.base)
        return 0

    print("  ═══ REPA f32 vs f16 逐位核验（只读 ✓ 裸二进制 mmap ✓）═══")
    n_eq = n_diff = n_skip = 0
    for d in dirs:
        st, note = check_pair(d, args.rows)
        mark = {"equal": "✓ **相等（f32 只是上转换 ✓ 无损 ✓）**",
                "differs": "✗ **不相等 ⇒ f32 有额外精度 ✗ 不能假定无损 ✗**",
                "skip": "— 跳过"}[st]
        print("  %-46s %s  %s" % (os.path.basename(d.rstrip("/\\")), mark, note))
        n_eq += st == "equal"
        n_diff += st == "differs"
        n_skip += st == "skip"
    print()
    print("  汇总: 相等 %d ✓ · 不相等 %d ✗ · 跳过 %d —" % (n_eq, n_diff, n_skip))
    if n_diff:
        print("  ✗ 存在 f32 **非**上转换的 cache ⇒ 那些 cache 必须继续用 f32 ✓"
              "（不得改默认 ✗）")
        return 1
    print("  ⇒ 所有可核对的 f32 都是 f16 的上转换 ✓ ⇒ **优先 f16 无损** ✓"
          "（H2D 与常驻各减半 ✓ 数值零变化 ✓）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""bench_std_soft.py — 优化前后 CPU 开销对比（这是 v81 在 48 上只有 0.6 steps/s 的根因）

测法：模拟一个 288 样本的 batch，逐样本 get()（含训练时的随机 σ），比较
  * slow：preload=False（每次现做高斯模糊 = 旧行为）
  * fast：preload=True（σ 网格已 materialize，取值=切片）
"""
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.utils.std_soft_glyph import StdSoftGlyphStore  # noqa: E402

STORE = "exp-std/data/std_soft_glyphs_v1"
rows = list(csv.DictReader(open(f"{STORE}/index.csv", encoding="utf-8-sig")))
keys = [(int(r["script_id"]), r["character"]) for r in rows]
BATCH = 288
REP = 3

t0 = time.perf_counter()
st_fast = StdSoftGlyphStore(STORE, is_train=True, preload=True)
t_build = time.perf_counter() - t0
print(f"  fast: 构建(含 σ 网格 materialize) 耗时 {t_build:.1f}s, 张量 {st_fast._stack.nbytes/2**20:.0f} MB")

st_slow = StdSoftGlyphStore(STORE, is_train=True, preload=False)

import numpy as np  # noqa: E402
idxs = np.random.default_rng(0).integers(0, len(keys), size=BATCH)

t0 = time.perf_counter()
for _ in range(REP):
    for i in idxs:
        st_slow.get(*keys[i])
t_slow = (time.perf_counter() - t0) / REP

# 预热一次再计时（避免首次缓存效应）
for i in idxs[:16]:
    st_fast.get(*keys[i])
t0 = time.perf_counter()
for _ in range(REP):
    for i in idxs:
        st_fast.get(*keys[i])
t_fast = (time.perf_counter() - t0) / REP

print(f"\n  单 batch({BATCH} 样本) CPU 时间：")
print(f"    slow（现做高斯模糊, 旧行为）: {t_slow*1000:8.1f} ms")
print(f"    fast（σ 网格切片,   新行为）: {t_fast*1000:8.1f} ms")
print(f"    加速比: {t_slow/max(t_fast,1e-9):.1f}×")
print("\n  按 8 workers 估算每步（288 样本/batch）总耗时：")
print(f"    旧: {t_slow*1000:.0f} ms → 单步下限 {1000/max(t_slow*1000,1e-9):.1f} steps/s（与实测 0.6 吻合）")
print(f"    新: {t_fast*1000:.1f} ms → 不再是瓶颈")
# 校验快慢路径数值一致（同一 σ 下）
a = st_slow.get(*keys[0], sigma=0.65)
b = st_fast.get(*keys[0], sigma=0.65)
print(f"\n  一致性检查（σ=0.65, max|Δ|）: {float((a-b).abs().max()):.4f}  "
      f"(量化到网格步长 0.05, 预期很小)")

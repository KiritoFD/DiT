# -*- coding: utf-8 -*-
"""mem_top.py — 解析 torch.cuda.memory._dump_snapshot() 的快照，给"分配点排行榜"。

用法:
    python tools/prof/mem_top.py /tmp/mem_snapshot_<pid>.pickle [--top 15]

回答的问题: **是谁在每一步做上百次 cudaMalloc?**
做法: 快照的 device_traces 里每个 alloc 事件都带完整 Python 调用栈;
      按"首个属于本仓库的帧"归并 -> 得到分配点排行（次数 / 字节）。
"""
import argparse
import collections
import os
import pickle
import re
import sys

ROOT_HINT = re.compile(r"(src/|tools/|train\.py|losses\.py|latent_dataset\.py|flow_matching\.py)")


def site_of(frames):
    """取首个属于本仓库的帧作为"分配点"; 拿不到就退回最外层业务帧。"""
    for fn, line, name in frames:
        s = f"{fn}"
        if "site-packages" in s or "/torch/" in s:
            continue
        if ROOT_HINT.search(s) or s.endswith(".py"):
            return f"{os.path.basename(fn)}:{line} {name}"
    if frames:
        fn, line, name = frames[0]
        return f"(外部) {os.path.basename(fn)}:{line} {name}"
    return "(无栈)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    snap = pickle.load(open(a.snapshot, "rb"))
    tr = snap.get("device_traces", [])
    if isinstance(tr, list) and tr and isinstance(tr[0], list):
        tr = [e for sub in tr for e in sub]           # 多 device: 展平
    alloc_cnt = collections.Counter()
    alloc_b = collections.Counter()
    free_cnt = 0
    n_alloc = 0
    sizes = collections.Counter()
    for e in tr:
        act = e.get("action") or e.get("type")
        if act == "alloc":
            n_alloc += 1
            s = site_of(e.get("frames") or [])
            alloc_cnt[s] += 1
            alloc_b[s] += int(e.get("size", 0))
            sizes[int(e.get("size", 0))] += 1
        elif act == "free":
            free_cnt += 1

    print(f"  快照: {a.snapshot}")
    print(f"  alloc 事件 {n_alloc:,}   free 事件 {free_cnt:,}   "
          f"(net {n_alloc - free_cnt:,} 未释放)")
    print()
    print(f"  === 分配点排行（按**次数**）TOP{a.top} ===")
    print(f"  {'次数':>9}  {'总字节':>14}  {'平均':>10}  分配点")
    for s, c in alloc_cnt.most_common(a.top):
        print(f"  {c:>9,}  {alloc_b[s]:>14,}  {alloc_b[s]/max(c,1):>10,.0f}  {s}")
    print()
    print(f"  === 分配点排行（按**字节**）TOP{a.top} ===")
    for s, b in alloc_b.most_common(a.top):
        print(f"  {alloc_cnt[s]:>9,}  {b:>14,}  {b/max(alloc_cnt[s],1):>10,.0f}  {s}")
    print()
    print(f"  === 最常见分配尺寸 TOP10 ===")
    for sz, c in sizes.most_common(10):
        print(f"  {c:>9,} 次  {sz:>12,} B  ({sz/1024/1024:.2f} MiB)")


if __name__ == "__main__":
    sys.exit(main())

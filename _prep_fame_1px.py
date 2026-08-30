# -*- coding: utf-8 -*-
"""为 fame 准备 1px 骨架：合并 train+eval 成一个全量 csv，并核对现有覆盖。

为什么要合并成一个 csv
----------------------
build_skel_latents.py 的 build_latents 用「区间命名」写 shard
(shard_{first}_{last}.npz)，并靠已存在 shard 的 img_ids 做断点续跑。
若分两次跑（train 一次、eval 一次）指向同一 latent-out，
第二次的 id 区间会与第一批 shard 重叠 → 同一个 shard 文件被覆盖，
导致 train 侧 latent 静默丢失。因此必须一次跑完全量。

输出
----
- 5script/fame_all_ids.csv  （train + eval 去重合并，供骨架/编码一次性使用）
- 控制台打印：现有 3px PNG / latent 的覆盖情况
"""
import os, sys, csv, re, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np


def ids_of(csv_path):
    ids = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)\.png", row.get("image_path", ""))
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))


def main():
    train = "5script/train_fame.csv"
    evalc = "5script/eval_fame_strict.csv"

    tr_ids = ids_of(train)
    ev_ids = ids_of(evalc)
    all_ids = sorted(set(tr_ids) | set(ev_ids))
    print(f"train ids = {len(tr_ids)}")
    print(f"eval  ids = {len(ev_ids)}")
    print(f"union     = {len(all_ids)}  (overlap={len(set(tr_ids) & set(ev_ids))})")

    # 现有覆盖核对
    sk3 = set()
    for p in glob.glob("final_skel3_fame/*.png"):
        sk3.add(int(os.path.splitext(os.path.basename(p))[0]))
    print(f"\nfinal_skel3_fame PNG = {len(sk3)}")
    print(f"  eval ids 已覆盖: {len(set(ev_ids) & sk3)}/{len(ev_ids)}")
    print(f"  train ids 已覆盖: {len(set(tr_ids) & sk3)}/{len(tr_ids)}")

    lat_ids = set()
    for sp in glob.glob("final_skel_latents_fame/shard_*.npz"):
        with np.load(sp) as d:
            lat_ids.update(int(x) for x in d["img_ids"])
    print(f"\nfinal_skel_latents_fame latent ids = {len(lat_ids)}")
    print(f"  eval ids 已覆盖: {len(set(ev_ids) & lat_ids)}/{len(ev_ids)}")

    # 合并 csv（保留 train 的列结构，追加 eval 行）
    with open(train, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        rows = list(rd)
    with open(evalc, encoding="utf-8") as f:
        rows_ev = list(csv.DictReader(f))

    seen = set()
    merged = []
    for r in rows + rows_ev:
        m = re.search(r"(\d+)\.png", r.get("image_path", ""))
        key = int(m.group(1)) if m else r.get("image_path")
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)

    out = "5script/fame_all_ids.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(merged)
    print(f"\nmerged csv -> {out}  rows={len(merged)}")

    # 诊断：shard 区间是否重叠（会引发覆盖 bug）
    spans = []
    for sp in sorted(glob.glob("final_skel_latents_fame/shard_*.npz")):
        b = os.path.basename(sp)
        m = re.match(r"shard_(\d+)_(\d+)\.npz", b)
        if m:
            spans.append((int(m.group(1)), int(m.group(2)), b))
    spans.sort()
    overlaps = [(spans[i][2], spans[i+1][2]) for i in range(len(spans)-1)
                if spans[i][1] >= spans[i+1][0]]
    print(f"shard 区间重叠对数 = {len(overlaps)}")
    if overlaps:
        print("  !! 存在重叠，分批写同一目录会互相覆盖")
        for a, b in overlaps[:5]:
            print(f"    {a} <-> {b}")


if __name__ == "__main__":
    main()

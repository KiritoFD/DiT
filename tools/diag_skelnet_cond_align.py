# -*- coding: utf-8 -*-
"""diag_skelnet_cond_align.py — 查 SkelNet 的**条件对齐**: 输入标准骨架是否真对应目标字。

动机 (2026-09-30): 目视发现 val 上 8 个不同字符拿到**同一张**标准骨架 ->
  cond shard 的 img_id 查表可能对不上 (仓库有过 2026-09-21 few-shot 同类事故)。
本脚本只读, 打印:
  1) val csv 的 id 段 / 字符, 与 cond shards 的 id 段是否同源
  2) 前 N 行: char, gt 目标 ink, std 输入 ink, std 输入的 md5 (看是否重复)
  3) 关键判据: 同一字符的多行是否拿到同一 std; **不同字符**是否拿到不同 std
"""
import argparse
import csv
import glob
import hashlib
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_map(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            lat = z["latents"]
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = np.asarray(lat[j], dtype=np.float32)
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/val_skelnet.csv")
    ap.add_argument("--cond", default="data/top10_style23/shards_std")
    ap.add_argument("--tgt", default="data/top10_style23/shards_gtskel_w3")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    print(f"val csv: {len(rows)} 行, 列={list(rows[0].keys())[:12]}")
    ids = []
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            ids.append(int(m.group(1)))
    print(f"  csv id: n={len(ids)} min={min(ids)} max={max(ids)} 唯一={len(set(ids))}")
    print(f"  例: {rows[0]['image_path']} | char={rows[0].get('character')} "
          f"| std_path={rows[0].get('std_path')}")

    cond = load_map(a.cond)
    tgt = load_map(a.tgt)
    ci = sorted(cond)
    ti = sorted(tgt)
    print(f"\ncond {a.cond}: {len(cond)} ids, min={ci[0]} max={ci[-1]}")
    print(f"tgt  {a.tgt}: {len(tgt)} ids, min={ti[0]} max={ti[-1]}")
    print(f"  csv∩cond = {len(set(ids) & set(cond))}/{len(set(ids))}")
    print(f"  csv∩tgt  = {len(set(ids) & set(tgt))}/{len(set(ids))}")

    # 同字符 -> 是否同一 std; 不同字符 -> 是否不同 std
    by_char = defaultdict(list)
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            by_char[r.get("character", "?")].append(int(m.group(1)))

    ch2std = {}
    for ch, iids in by_char.items():
        hs = {hashlib.md5(cond[i].tobytes()).hexdigest()[:10]
              for i in iids if i in cond}
        ch2std[ch] = hs
    multi = {c: h for c, h in ch2std.items() if len(h) > 1}
    print(f"\n字符数 {len(ch2std)}; **同一字符映射到多个 std** 的字符数 = {len(multi)}"
          f"  (应为 0)")
    uniq_sig = defaultdict(list)
    for c, h in ch2std.items():
        uniq_sig[tuple(sorted(h))].append(c)
    dup = {k: v for k, v in uniq_sig.items() if len(v) > 1}
    print(f"**不同字符共用同一 std** 的组数 = {len(dup)}  (应为 0!)")
    for k, v in list(dup.items())[:5]:
        print(f"   sig={k[:1]} 被 {len(v)} 个不同字符共用: {v[:10]}")

    print(f"\n前 {a.n} 行明细:")
    print(f"{'idx':>4} {'char':>4} {'id':>9} {'gt_ink':>8} {'std_ink':>8} {'std_md5':>11}")
    for i, r in enumerate(rows[:a.n]):
        m = re.search(r"(\d+)\.png", r["image_path"])
        iid = int(m.group(1)) if m else -1
        gi = float((tgt[iid] < 0).mean()) if iid in tgt else float("nan")
        si = float((cond[iid] < 0).mean()) if iid in cond else float("nan")
        h = hashlib.md5(cond[iid].tobytes()).hexdigest()[:10] if iid in cond else "-"
        print(f"{i:4d} {r.get('character','?'):>4} {iid:9d} {gi:8.4f} {si:8.4f} {h:>11}")


if __name__ == "__main__":
    main()

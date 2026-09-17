# -*- coding: utf-8 -*-
"""从 50k 训练集里**切出**评测集：strict(250) + seen(20)。

## 语义
- **strict**：从 50k 里挑 250 张，并**从训练集里拿掉** -> 真正的"未见图"。
  判据（沿用旧 strict 集的口径）：
    - (script, calligrapher) 在**剩余训练集**里出现过  -> 书家要素有覆盖
    - (script, calligrapher, character) 三元组**没出现过** -> 组合没覆盖
    - 按 script 均衡
- **seen**：挑 20 张，**保留在训练集里** -> 模型见过的图。
  按书家轮转，避免偏某几个人。

## 为什么不从别处找
新加的数据（hcsu_wild/bei/tie、calli_tongji）只有进了 50k 的那部分
才有配套的 latent / std，别处找不到同源候选。**从 50k 里切最省事也最一致。**
（`archive/final_manifest.json` 是**旧**清单，不含新数据，实测选出 0 条。）

## 产物
    assets/train_50k_v2.csv     训练集（= 原 50k - strict 的 250 行）
    assets/eval_v13_strict.csv  250 行，未见图
    assets/eval_v13_seen.csv    20 行，见过图

⚠ strict 的 250 张 latent 仍在 `data/50k/shards_*` 里（不删），
  评测端按 img_id 查得到，**不需要重新编码**。

用法:
    python tools/split_eval_from_50k.py --strict 250 --seen 20
"""
import argparse
import csv
import os
import random
import re
import sys
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())


def iid_of(p):
    m = re.search(r"(\d+)\.png$", str(p))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="assets/train_50k.csv")
    ap.add_argument("--out-train", default="assets/train_50k_v2.csv")
    ap.add_argument("--out-strict", default="assets/eval_v13_strict.csv")
    ap.add_argument("--out-seen", default="assets/eval_v13_seen.csv")
    ap.add_argument("--strict", type=int, default=250)
    ap.add_argument("--seen", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = list(csv.DictReader(open(a.train, encoding="utf-8")))
    print(f"[split] {a.train}: {len(rows)} 行")

    # ── 1. 先选 seen（保留在训练集里）────────────────────────────────
    by_cal = defaultdict(list)
    for r in rows:
        by_cal[int(r["calligrapher_id"])].append(r)
    cals = sorted(by_cal)
    rng.shuffle(cals)
    seen, pool_used = [], set()
    i = 0
    while len(seen) < a.seen and i < 10000:
        live = [c for c in cals if by_cal[c]]
        if not live:
            break
        c = live[i % len(live)]
        p = by_cal[c].pop(rng.randrange(len(by_cal[c])))
        seen.append(p)
        i += 1
    print(f"[split] seen {len(seen)} 张（覆盖 "
          f"{len(set(int(r['calligrapher_id']) for r in seen))} 个书家），保留在训练集")

    # ── 2. 选 strict：三元组未见 + (script,callig) 见过 ──────────────
    remaining = [r for r in rows if r not in seen]     # dict 比较，够用
    seen_sc = {(int(r["script_id"]), int(r["calligrapher_id"])) for r in remaining}
    # (script, callig, char) 三元组 -> 该组有几张
    trip = defaultdict(list)
    for r in remaining:
        trip[(int(r["script_id"]), int(r["calligrapher_id"]),
              int(r.get("character_id", 0)))].append(r)

    # 只挑"该三元组只有 1 张"的（抽走它，三元组就真的不在训练集里了）
    solo = {k: v[0] for k, v in trip.items()
            if len(v) == 1 and k[:2] in seen_sc}
    print(f"[split] 可选 strict 候选（三元组唯一 + 书家见过）: {len(solo)} 张")

    by_script = defaultdict(list)
    for k in sorted(solo):
        by_script[k[0]].append(k)
    for v in by_script.values():
        rng.shuffle(v)

    strict, used_trip = [], set()
    while len(strict) < a.strict and any(by_script.values()):
        for sc in sorted(by_script):
            if by_script[sc] and len(strict) < a.strict:
                k = by_script[sc].pop(0)
                strict.append(solo[k])
                used_trip.add(k)
    print(f"[split] strict {len(strict)} 张（从训练集移除 -> 真正的未见图）")

    strict_ids = {iid_of(r["image_path"]) for r in strict}
    strict_paths = {r["image_path"] for r in strict}
    train2 = [r for r in rows
              if r["image_path"] not in strict_paths
              and iid_of(r["image_path"]) not in strict_ids]
    print(f"[split] 训练集 {len(rows)} -> {len(train2)} 行")

    # ── 3. 落盘 ──────────────────────────────────────────────────────
    with open(a.out_train, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(train2)
    for path, data in ((a.out_strict, strict), (a.out_seen, seen)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(data)

    # ── 4. 自检 ──────────────────────────────────────────────────────
    print()
    print("  === 自检 ===")
    tr_cal = {int(r["calligrapher_id"]) for r in train2}
    for name, data in (("strict", strict), ("seen", seen)):
        ev_cal = {int(r["calligrapher_id"]) for r in data}
        ev_sc = {(int(r["script_id"]), int(r["calligrapher_id"])) for r in data}
        print(f"  {name}: {len(data)} 张, {len(ev_cal)} 书家, "
              f"书家都在训练集里? {ev_cal <= tr_cal}, "
              f"(script,callig) 都在训练集里? {ev_sc <= seen_sc}")
    overlap = {r['image_path'] for r in strict} & {r['image_path'] for r in train2}
    print(f"  strict 与训练集重合: {len(overlap)} 张（应为 0）")
    print(f"  产物: {a.out_train} / {a.out_strict} / {a.out_seen}")


if __name__ == "__main__":
    main()

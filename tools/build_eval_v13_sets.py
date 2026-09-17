# -*- coding: utf-8 -*-
"""构建 v13 的评测集：strict(250) + seen(20)。

## strict（未见图，跨书家泛化）
判据沿用 `build_eval_strict_midclean.py`：
  - (script, calligrapher) 在 train csv 中出现过   -> 书家要素有覆盖
  - (script, calligrapher, character) 三元组未出现 -> 组合没覆盖
  - img_id 不在 train csv 中                       -> 图片本身未训练
候选池 = `archive/final_manifest.json`（含未进入任何 train csv 的图）。

## seen（模型见过的图）
直接从 train csv 里**均匀抽 20 张**（按书家轮转，避免偏某几个人）。

## ⚠ 与旧 eval 集的关键差异
旧集用的是旧 train csv（36 书家）建的，里面有 5 个书家在任何训练数据里都是
0 张（base 词表的"空号"）-> v13 的 45 人词表覆盖不了 -> 越界崩溃。
本脚本用 **train_50k.csv** 建，所以书家**必然都在 45 人词表内** ✓

用法:
    python tools/build_eval_v13_sets.py --strict 250 --seen 20
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

TRAIN = "assets/train_50k.csv"
MANIFEST = "archive/final_manifest.json"
CMAP = "assets/callig_id_map_50k.json"


def iid_of(p):
    m = re.search(r"(\d+)\.png$", str(p))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--cmap", default=CMAP)
    ap.add_argument("--strict", type=int, default=250)
    ap.add_argument("--seen", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cmap = {int(k) for k in json.load(open(a.cmap, encoding="utf-8"))["id_map"]}
    train = list(csv.DictReader(open(a.train, encoding="utf-8")))
    train_ids = {iid_of(r["image_path"]) for r in train}
    train_paths = {r["image_path"] for r in train}
    # (script, calligrapher) 见过；(script, calligrapher, character) 没见过
    seen_sc = {(int(r["script_id"]), int(r["calligrapher_id"])) for r in train}
    seen_scc = {(int(r["script_id"]), int(r["calligrapher_id"]),
                 int(r.get("character_id", 0))) for r in train}
    print(f"[eval-v13] 训练集 {len(train)} 行, 词表 {len(cmap)} 书家")

    # ── strict ────────────────────────────────────────────────────────
    man = json.load(open(a.manifest, encoding="utf-8"))
    items = man if isinstance(man, list) else man.get("items", man.get("images", []))
    print(f"[eval-v13] manifest 候选 {len(items)} 条")

    cand = defaultdict(list)          # (script, character) -> [(row, ...)]
    for it in items:
        p = it.get("image_path") or it.get("path") or ""
        iid = iid_of(p)
        if iid is None or iid in train_ids or p in train_paths:
            continue
        try:
            sc = int(it["script_id"]); cg = int(it["calligrapher_id"])
            ch = int(it.get("character_id", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if cg not in cmap:                       # 书家必须在 45 人词表内
            continue
        if (sc, cg) not in seen_sc:              # 书家要素要有覆盖
            continue
        if (sc, cg, ch) in seen_scc:             # 三元组不能见过
            continue
        cand[(sc, ch)].append(it)

    keys = sorted(cand)
    rng.shuffle(keys)
    by_script = defaultdict(list)
    for k in keys:                               # 按 script 均衡
        by_script[k[0]].append(k)

    picked, order = [], []
    while len(picked) < a.strict and any(by_script.values()):
        for sc in sorted(by_script):
            if by_script[sc] and len(picked) < a.strict:
                k = by_script[sc].pop(0)
                picked.append(rng.choice(cand[k]))
    print(f"[eval-v13] strict 选出 {len(picked)} 条 "
          f"(按 (script,character) 唯一, script 均衡)")

    with open("assets/eval_v13_strict.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "calligrapher", "script", "character",
                    "calligrapher_id", "script_id", "character_id", "glyph_id"])
        for it in picked:
            w.writerow([it.get("image_path", ""), it.get("calligrapher", ""),
                        it.get("script", ""), it.get("character", ""),
                        it.get("calligrapher_id", ""), it.get("script_id", ""),
                        it.get("character_id", ""), it.get("glyph_id", "")])
    print("    -> assets/eval_v13_strict.csv")

    # ── seen ──────────────────────────────────────────────────────────
    by_cal = defaultdict(list)
    for r in train:
        by_cal[int(r["calligrapher_id"])].append(r)
    cals = sorted(by_cal)
    rng.shuffle(cals)
    seen_rows = []
    i = 0
    while len(seen_rows) < a.seen and cals:
        c = cals[i % len(cals)]
        pool = by_cal[c]
        if pool:
            seen_rows.append(pool.pop(rng.randrange(len(pool))))
        cals = [x for x in cals if by_cal[x]] or cals
        i += 1
        if i > 10000:
            break
    with open("assets/eval_v13_seen.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(train[0].keys()))
        w.writeheader()
        w.writerows(seen_rows)
    print(f"[eval-v13] seen 选出 {len(seen_rows)} 条（按书家轮转，覆盖 "
          f"{len(set(int(r['calligrapher_id']) for r in seen_rows))} 个书家）")
    print("    -> assets/eval_v13_seen.csv")


if __name__ == "__main__":
    main()

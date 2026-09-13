#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切 top6 人名书家数据集: 楷(0)+隶(4) 每书体各 top6 人名书家(共11位唯一, 赵孟頫横跨两书体)。

样本数口径: eval100 = 100 张图(楷隶各50), show2 = 2 张图(楷1隶1), seen2 = 2 张图(楷1隶1)。

输出:
  train_top6.csv    : 11 位书家楷隶全部样本
  eval100_top6.csv  : 100 张图 (优先 eval.csv unseen, 不足从 train 补)
  show2_top6.csv    : 2 张图 (eval 侧 unseen, 楷1隶1)
  seen2_top6.csv    : 2 张图 (train 侧, 楷1隶1)
"""
import csv
import os
import random

random.seed(0)
BASE = "/root/Workspace/xy/DiT/assets"
SCRIPT_NAMES = {0: "楷", 1: "篆", 2: "草", 3: "行", 4: "隶"}

# 每书体 top6 人名书家 (行数排序, 人工确认)
KAILI_TOP6 = ["颜真卿", "赵孟𫖯", "褚遂良", "智永", "柳公权", "欧阳询"]
LI_TOP6    = ["赵孟𫖯", "王澍", "吴叡", "金农", "陈鸿寿", "邓石如"]
CALLIG_BY_SCRIPT = {0: KAILI_TOP6, 4: LI_TOP6}

def load(path):
    with open(os.path.join(BASE, path), encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    train = load("train.csv")
    print(f"train.csv 总行数: {len(train)}")

    # 按 (script_id, calligrapher_name) 筛选
    ok_ids = {}
    for sid, names in CALLIG_BY_SCRIPT.items():
        for nm in names:
            ok_ids[(sid, nm)] = 0
    sub6 = []
    for r in train:
        key = (int(r["script_id"]), r["calligrapher"])
        if key in ok_ids:
            ok_ids[key] += 1
            sub6.append(r)
    print(f"\ntop6 书家楷隶样本: {len(sub6)} 行")
    for (sid, nm), cnt in sorted(ok_ids.items(), key=lambda kv: (-kv[1], kv[0][0])):
        print(f"  {SCRIPT_NAMES[sid]} {nm}: {cnt} 行")

    # 字数统计(按书体)
    for sid in sorted(CALLIG_BY_SCRIPT):
        sc = [r for r in sub6 if int(r["script_id"]) == sid]
        print(f"\n{SCRIPT_NAMES[sid]}: {len(sc)} 行, 字数 {len({r['character'] for r in sc})}")
    print(f"总字数(去重 character): {len({r['character'] for r in sub6})}")

    hdr = list(train[0].keys())
    with open(os.path.join(BASE, "train_top6.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(sub6)
    print(f"\n写入 train_top6.csv ({len(sub6)} 行)")

    # eval100: 每书体 50 张, 优先 eval.csv unseen, 不足从 train 补
    eval_csv = load("eval.csv")
    eval_unseen = [r for r in eval_csv
                   if (int(r["script_id"]), r["calligrapher"]) in ok_ids
                   and r["image_path"] not in {x["image_path"] for x in train}]
    print(f"\neval.csv 中 top6书家楷隶 unseen 样本: {len(eval_unseen)}")

    def per_script_pool(pool, sid):
        return [r for r in pool if int(r["script_id"]) == sid]

    ev = []
    used_eval = []
    for sid in sorted(CALLIG_BY_SCRIPT):
        pool = per_script_pool(eval_unseen, sid)
        random.shuffle(pool)
        take = min(50, len(pool))
        ev.extend(pool[:take])
        used_eval.extend(pool[:take])
    missing = 100 - len(ev)
    if missing > 0:
        for sid in sorted(CALLIG_BY_SCRIPT):
            have = sum(1 for r in ev if int(r["script_id"]) == sid)
            need = 50 - have
            if need <= 0:
                continue
            pool = [r for r in per_script_pool(sub6, sid)
                    if r["image_path"] not in {x["image_path"] for x in ev}]
            random.shuffle(pool)
            ev.extend(pool[:need])
    with open(os.path.join(BASE, "eval100_top6.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(ev)
    e_src = f"eval.csv {len(used_eval)} + train补充 {100 - len(used_eval)}"
    print(f"写入 eval100_top6.csv ({len(ev)} 张图: 楷{sum(1 for r in ev if int(r['script_id'])==0)} 隶{sum(1 for r in ev if int(r['script_id'])==4)}) 来源 {e_src}")

    # show2: 楷1隶1 (eval side)
    show = []
    for sid in sorted(CALLIG_BY_SCRIPT):
        pool = per_script_pool(eval_unseen, sid)
        if pool:
            show.append(random.choice(pool))
        else:
            pool = per_script_pool(sub6, sid)
            show.append(random.choice(pool))
    with open(os.path.join(BASE, "show2_top6.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(show)
    for r in show:
        print(f"  show2: {r['calligrapher']} {r['script']} {r['character']} {os.path.basename(r['image_path'])}")

    # seen2: 楷1隶1 (train side)
    seen = []
    for sid in sorted(CALLIG_BY_SCRIPT):
        pool = per_script_pool(sub6, sid)
        seen.append(random.choice(pool))
    with open(os.path.join(BASE, "seen2_top6.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(seen)
    for r in seen:
        print(f"  seen2: {r['calligrapher']} {r['script']} {r['character']} {os.path.basename(r['image_path'])}")

    print("\n完成。")
    # 简单校验: 各 csv 行数
    for f in ["train_top6.csv", "eval100_top6.csv", "show2_top6.csv", "seen2_top6.csv"]:
        n = sum(1 for _ in open(os.path.join(BASE, f), encoding="utf-8"))
        print(f"  {f}: {n-1} 数据行")

if __name__ == "__main__":
    main()
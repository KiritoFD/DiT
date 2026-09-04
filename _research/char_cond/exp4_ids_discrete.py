# -*- coding: utf-8 -*-
"""exp4: IDS 离散部件编码 vs 视觉特征 —— 三层次对比。

动机: exp1-3 证明视觉特征(真迹/印刷体)都编码"外观相似度"而非"字符身份":
  - 真迹: 被书体主导(同字 0.24 < 形近 0.40)
  - 印刷体: 字间区分度归零(随机 0.736 ≈ 同字 0.746)
字符身份本质是**离散符号** → 用 IDS 部件(离散、书体无关、可组合泛化)编码。

评测:
  1) 覆盖率: IDS 覆盖我们字表多少
  2) 三层次 cos(部件计数向量的余弦): 同字 / 形近 / 随机
  3) 部件词表规模、每字部件数
"""
import os
import sys
import json
import csv
import numpy as np
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:\GitHub\DiT"
os.chdir(ROOT)

IDS_FILE = "_sync_work/cjkvi-ids/cjkvi-ids-master/ids.txt"
CSV = "5script/train_fame.csv"
IDS_OPS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

SIM_PAIRS = [
    ("土","士"),("大","太"),("人","入"),("日","曰"),("天","夫"),("刀","力"),
    ("申","由"),("王","玉"),("牛","午"),("己","已"),("未","末"),("木","林"),
    ("休","体"),("侯","候"),("风","凤"),("几","凡"),("戊","戍"),("己","巳"),
    ("亳","毫"),("宋","宗"),("东","车"),("冈","同"),("三","王"),("十","干"),
    ("大","天"),("夫","天"),("犬","太"),("人","个"),("手","毛"),("毛","手"),
    ("子","孑"),("戊","戌"),("戍","戌"),("刀","刁"),("万","方"),("鸟","乌"),
    ("贝","见"),("龙","尤"),("失","矢"),("句","向"),("因","困"),("同","回"),
    ("问","间"),("门","们"),("口","曰"),("田","由"),("甲","申"),("白","百"),
    ("吉","古"),("夫","夭"),("天","夭"),("王","主"),("玉","主"),("人","入"),
]


def load_ids(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0].startswith("U+"):
                ch, ids = p[1], p[2]
                if len(ch) == 1 and ch not in m:
                    m[ch] = ids
    return m


def components(ids_str, depth=0):
    """递归提取叶子部件（含重复）。"""
    if depth > 4:
        return []
    out = []
    for c in ids_str:
        if c in IDS_OPS or c.isspace():
            continue
        if ord(c) < 128:
            continue
        out.append(c)
    return out


def vec(comp_list):
    return Counter(comp_list)


def cos(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


def main():
    ids = load_ids(IDS_FILE)
    print(f"IDS 字典: {len(ids)} 字")

    # 我们的字表
    our = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        ch = r["character"]
        if len(ch) == 1:
            our[ch] = r["character_id"]
    print(f"我们的字表: {len(our)} 字")
    cov = [c for c in our if c in ids]
    print(f"IDS 覆盖: {len(cov)}/{len(our)} = {len(cov)/len(our):.1%}")

    # 部件表示
    comps = {c: vec(components(ids[c])) for c in cov}
    allc = Counter()
    for v in comps.values():
        allc.update(v)
    print(f"部件词表: {len(allc)} 部件; 平均每字部件数 "
          f"{np.mean([sum(v.values()) for v in comps.values()]):.2f}")
    print(f"  TOP10 部件: {allc.most_common(10)}")

    # 三层次
    sp = [(a, b) for a, b in SIM_PAIRS if a in comps and b in comps]
    cos_sim = np.array([cos(comps[a], comps[b]) for a, b in sp])
    rng = np.random.default_rng(0)
    cs = list(comps.keys())
    sps = set(SIM_PAIRS) | {(b, a) for a, b in SIM_PAIRS}
    neg = []
    while len(neg) < 5000:
        a, b = rng.choice(cs, 2, replace=False)
        if (a, b) in sps:
            continue
        neg.append(cos(comps[a], comps[b]))
    cos_rand = np.array(neg)
    cos_same = 1.0  # 同字 = 同一部件集

    print("\n" + "=" * 68)
    print("IDS 离散部件编码 —— 三层次评测")
    print("=" * 68)
    print(f"  同字 cos         = {cos_same:.4f}  (定义: 同一部件集)")
    print(f"  形近字 cos       = {cos_sim.mean():.4f} ± {cos_sim.std():.4f}  (n={len(sp)})")
    print(f"  随机字 cos       = {cos_rand.mean():.4f} ± {cos_rand.std():.4f}")
    ok = cos_same > cos_sim.mean() > cos_rand.mean()
    print(f"  三层次有序?      {'✓ 是' if ok else '✗ 否(但见下方解读)'}")
    print(f"  区分度(同字-形近)= {cos_same - cos_sim.mean():+.4f}")
    print(f"  与视觉特征对比:")
    print(f"    真迹 DINO     : 同字 0.2418 / 形近 0.4020 / 随机 0.0019  ✗ 风格主导")
    print(f"    标准字形 DINO : 同字 0.7461 / 形近 0.8824 / 随机 0.7364  ✗ 无区分")
    print(f"    IDS 部件      : 同字 {cos_same:.4f} / 形近 {cos_sim.mean():.4f} "
          f"/ 随机 {cos_rand.mean():.4f}")
    print("=" * 68)

    # 形近字明细
    print("\n  形近字对（IDS 部件余弦）:")
    for (a, b), c in sorted(zip(sp, cos_sim), key=lambda x: -x[1])[:12]:
        print(f"    {a}/{b}: {c:.3f}   部件 a={list(comps[a])} b={list(comps[b])}")

    out = "_research/char_cond/state/exp4_results.json"
    json.dump({
        "ids_chars": len(ids), "our_chars": len(our), "coverage": round(len(cov)/len(our), 4),
        "covered": len(cov), "comp_vocab": len(allc),
        "avg_comp_per_char": round(float(np.mean([sum(v.values()) for v in comps.values()])), 2),
        "cos_same": 1.0,
        "cos_sim": round(float(cos_sim.mean()), 4),
        "cos_rand": round(float(cos_rand.mean()), 4),
        "hierarchy_ok": bool(ok),
    }, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

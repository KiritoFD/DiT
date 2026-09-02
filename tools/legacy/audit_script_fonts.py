# -*- coding: utf-8 -*-
"""audit_script_fonts.py — 书体标签 / 字体覆盖审计.

回答:
  * fame 训练集实际有哪几种书体? 各多少样本/唯一字?
  * 有没有 SCRIPT_ID 未覆盖的异常标签 (会导致 KeyError)?
  * MCCD 原始库里有哪些书体 (有没有没放进 fame 的)?
  * gradio 只暴露 3 种书体 -> 隐藏了多少训练书体?
  * zero-shot 骨架渲染字体: 草/篆/六体 是否错用楷体?
"""
import os
import sys
import csv
import collections

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SCRIPT_ID = {"楷": 0, "行": 3, "隶": 4, "草": 2, "篆": 1, "六体": 5}
SCRIPT_FONT = {"楷": "simkai.ttf", "行": "STXINGKA.TTF", "隶": "SIMLI.TTF",
               "草": "simkai.ttf", "篆": "simkai.ttf", "六体": "simkai.ttf"}
GRADIO_SCRIPTS = ["楷", "行", "隶"]

rows = list(csv.DictReader(open("5script/train_fame.csv", encoding="utf-8")))
print(f"[fame] 样本 {len(rows)}")

# --- 1. 书体分布 ---
n = collections.Counter(r["script"] for r in rows)
uniq = collections.defaultdict(set)
for r in rows:
    uniq[r["script"]].add(r["character"])
print("\n=== 1. fame 书体分布 ===")
print(f"{'书体':<6s}{'样本':>8s}{'占比':>8s}{'唯一字':>8s}{'SCRIPT_ID':>10s}")
for k, v in n.most_common():
    sid = SCRIPT_ID.get(k)
    flag = "" if sid is not None else "  <<< 未映射!"
    sid_s = "—" if sid is None else str(sid)
    print(f"{k:<6s}{v:8d}{v/len(rows)*100:7.1f}%{len(uniq[k]):8d}"
          f"{sid_s:>10s}{flag}")

# --- 2. 异常标签 ---
bad = sorted(set(n) - set(SCRIPT_ID))
print(f"\n=== 2. SCRIPT_ID 未覆盖的书体标签: {bad if bad else '无'} ===")
for b in bad:
    ex = [r for r in rows if r["script"] == b][:3]
    for r in ex:
        print(f"  样本: char={r['character']!r} callig={r['calligrapher']!r} "
              f"path={r['image_path']} cid={r.get('character_id')}")
    print(f"  -> 共 {n[b]} 条, 生成时会 KeyError")

# --- 3. gradio 暴露度 ---
print(f"\n=== 3. gradio 书体暴露度 (UI={GRADIO_SCRIPTS}) ===")
tot = sum(n[k] for k in GRADIO_SCRIPTS)
print(f"  已暴露: {tot} ({tot/len(rows)*100:.1f}%)")
hidden = [(k, n[k]) for k in n if k not in GRADIO_SCRIPTS]
print(f"  被隐藏: {len(hidden)} 种, {sum(v for _,v in hidden)} "
      f"({sum(v for _,v in hidden)/len(rows)*100:.1f}%)")
for k, v in sorted(hidden, key=lambda x: -x[1]):
    print(f"    {k}: {v} ({v/len(rows)*100:.1f}%)")

# --- 4. zero-shot 字体错配 ---
print("\n=== 4. zero-shot 骨架渲染字体 ===")
for k in n:
    f = SCRIPT_FONT.get(k, "simkai.ttf")
    ok = "" if k in ("楷", "行", "隶") else "  <<< 无专用字体, 用楷体代替"
    print(f"  {k:<6s} -> {f}{ok}")
mis = [k for k in n if k not in ("楷", "行", "隶")]
if mis:
    print(f"  -> {len(mis)} 种书体 zero-shot 时错用楷体骨架: {mis}")
    print(f"     受影响唯一字: {sum(len(uniq[k]) for k in mis)}")

# --- 5. MCCD 原始书体 ---
p = "5script/mccd_image_map.csv"
if os.path.isfile(p):
    mrows = list(csv.DictReader(open(p, encoding="utf-8")))
    sc = collections.Counter()
    fail = 0
    for r in mrows:
        b = os.path.basename(r["filepath"]).rsplit("-", 1)[0]
        parts = b.split("-")
        if len(parts) >= 3:
            sc[parts[1]] += 1
        else:
            fail += 1
    print(f"\n=== 5. MCCD 原始库书体 (源自文件名, {len(mrows)} 张) ===")
    for k, v in sc.most_common(15):
        mark = ""
        if k in n:
            mark = "  [fame 已收]"
        else:
            mark = "  <<< fame 未收"
        print(f"  {k:<8s}{v:8d}{mark}")
    print(f"  parse-fail: {fail}")

    # 书家
    ca = collections.Counter()
    for r in mrows:
        b = os.path.basename(r["filepath"]).rsplit("-", 1)[0]
        parts = b.split("-")
        if len(parts) >= 3:
            ca[parts[-1]] += 1
    print(f"\n  MCCD 书家数: {len(ca)}")
    print(f"  top5: {ca.most_common(5)}")
    fc = set(r["calligrapher"] for r in rows)
    print(f"  fame 书家数: {len(fc)}")
    only_mccd = set(ca) - fc
    print(f"  MCCD 有但 fame 未收的书家: {len(only_mccd)}")
    if only_mccd:
        top = sorted(((k, ca[k]) for k in only_mccd),
                     key=lambda x: -x[1])[:10]
        print(f"    top10: {top}")

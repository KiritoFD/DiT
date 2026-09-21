"""用「多书家一致性」过滤 7B 的错配发现 —— 降低 VLM 自身错认的影响。

原理:
  如果同一个 (script, char_csv) 的**多个样本**（不同书家/不同图）都被 VLM
  识别成**同一个繁体字**，那大概率是真错配；
  如果各样本识别结果五花八门，那是 VLM 认不准。

输出: assets/mismatch_final.csv
"""
import csv
from collections import Counter, defaultdict

from opencc import OpenCC

s2t = OpenCC("s2t")

rows = []
for f in ("assets/vlm_7b_lowconf.csv", "assets/vlm_7b_mismatch.csv"):
    try:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            r["_src"] = f
            rows.append(r)
    except FileNotFoundError:
        pass
print(f"  合并 VLM 结果: {len(rows)}")

# 只看"csv 简体 -> VLM 繁体"的
cand = defaultdict(list)
for r in rows:
    a, b = r["char_csv"], r["char_vlm"]
    if a == b:
        continue
    if s2t.convert(a) == b and s2t.convert(a) != a:
        cand[(r["script"], a)].append(r)

print(f"  候选 (script, char) 组数: {len(cand)}")
tot = sum(len(v) for v in cand.values())
print(f"  候选样本数: {tot}")

# 一致性: 组内 VLM 输出是否一致
consistent = {}
inconsistent = {}
for k, lst in cand.items():
    outs = Counter(r["char_vlm"] for r in lst)
    top, n = outs.most_common(1)[0]
    if len(outs) == 1 or n / len(lst) >= 0.6:
        consistent[k] = (top, n, len(lst))
    else:
        inconsistent[k] = (outs, len(lst))

n_con = sum(v[2] for v in consistent.values())
n_inc = sum(v[1] for v in inconsistent.values())
print(f"\n  === 一致性过滤 ===")
print(f"    ✓ 组内一致(>=60% 同一输出): {len(consistent)} 组, {n_con} 样本")
print(f"    ✗ 组内不一致:               {len(inconsistent)} 组, {n_inc} 样本")

print(f"\n  === 一致的组（前 30）===")
for (sc, ch), (top, n, tot_) in sorted(consistent.items(),
                                       key=lambda x: -x[1][2])[:30]:
    print(f"    [{sc}] {ch} -> {top}   {n}/{tot_} 一致")

print(f"\n  === 不一致的组（前 12，VLM 认不准）===")
for (sc, ch), (outs, tot_) in list(inconsistent.items())[:12]:
    print(f"    [{sc}] {ch}: {dict(outs)} (共{tot_})")

# 导出最终清单：只保留一致的
with open("assets/mismatch_final.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_true", "vlm_agree", "vlm_total"])
    for (sc, ch), (top, n, tot_) in consistent.items():
        for r in cand[(sc, ch)]:
            if r["char_vlm"] == top:
                w.writerow([r["idx"], r["image_path"], r["script"],
                            r["calligrapher"], r["char_csv"], top, n, tot_])
print(f"\n  ✓ written assets/mismatch_final.csv ({n_con} 条)")
print(f"    占全量 {n_con/50786*100:.2f}%")

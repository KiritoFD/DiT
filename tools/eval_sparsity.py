# -*- coding: utf-8 -*-
"""eval coverage + sparsity analysis vs train_base_noaug."""
import collections as C
import csv
import re

train = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
out = open("eval_sparsity.txt", "w", encoding="utf-8")

# train 分布
by_script = C.Counter(r["script"] for r in train)
by_callig = C.Counter(r["calligrapher"] for r in train)
cc_pairs = C.Counter()          # (callig, char) 实例数
callig_chars = {}               # callig -> unique chars
char_calligs = {}               # char -> calligraphers having it
for r in train:
    k = (r["calligrapher"], r["character"])
    cc_pairs[k] += 1
    callig_chars.setdefault(r["calligrapher"], set()).add(r["character"])
    char_calligs.setdefault(r["character"], set()).add(r["calligrapher"])
out.write(f"train: {len(train)} rows\nscripts: {dict(by_script)}\n\n")
out.write(f"(callig,char) unique pairs: {len(cc_pairs)}\n")
multi = sum(1 for v in cc_pairs.values() if v > 1)
out.write(f"(callig,char) with >1 instance: {multi}\n")
per_cc = sorted(cc_pairs.values())
out.write(f"instances per (callig,char): med={per_cc[len(per_cc)//2]}, "
          f"p10={per_cc[len(per_cc)//10]}, max={max(per_cc)}\n")
# 每书家 unique chars
u = sorted((len(v), c) for c, v in callig_chars.items())
out.write(f"\nunique chars per calligrapher: min={u[0]}, med={u[len(u)//2]}, max={u[-1]}\n")

# strict eval 覆盖
strict = list(csv.DictReader(open("assets/eval_fame3_strict_clean_v9.csv", encoding="utf-8")))
seen = list(csv.DictReader(open("assets/eval_seen_v10.csv", encoding="utf-8")))

def check(evals, name):
    n_in_train = 0            # (callig,script,char) 泄漏在 train (应为 0)
    callig_ok = 0             # 书家在 train 里
    callig_style_chars = []   # 该书家在 train 的字数 (风格学习依据)
    char_other = 0            # 该字在其他书家 train 里 (结构共性)
    n_new_callig = 0          # eval 书家已不在 train (被剔除的)
    for r in evals:
        c, s, ch = r["calligrapher"], r["script"], r["character"]
        if (c, s, ch) in {(r2["calligrapher"], r2["script"], r2["character"]) for r2 in train}:
            n_in_train += 1
        if c in callig_chars:
            callig_ok += 1
            callig_style_chars.append(len(callig_chars[c]))
            if ch not in callig_chars[c] and ch in char_calligs:
                char_other += 1
        else:
            n_new_callig += 1
    out.write(f"\n[{name}] n={len(evals)}\n")
    out.write(f"  (callig,script,char) 泄漏进 train: {n_in_train} (应为 0)\n")
    out.write(f"  书家在 train: {callig_ok}/{len(evals)} (不在的 {n_new_callig} 家)\n")
    if callig_style_chars:
        cs = sorted(callig_style_chars)
        out.write(f"  eval 书家的 train 字数: min={cs[0]} med={cs[len(cs)//2]} max={cs[-1]}\n")
    out.write(f"  eval 字有其他书家实例 (结构迁移): {char_other}\n")

check(strict, "strict n=237 (可部署主指标)")
check(seen, "seen n=10")

# 新书家 (UniCalli/tongji 加入的) 是否值得新 eval
fame3_calligs = set()
for r in csv.DictReader(open("assets/train_fame3_clean_v8.csv", encoding="utf-8")):
    fame3_calligs.add(r["calligrapher"])
new_calligs = {c: v for c, v in by_callig.items() if c not in fame3_calligs}
out.write(f"\nbase 新增书家 (fame3 之外): {len(new_calligs)} 家, 共 {sum(new_calligs.values())} 行\n")
for c, v in sorted(new_calligs.items(), key=lambda x: -x[1])[:12]:
    out.write(f"  {c}: {v} (unique chars {len(callig_chars[c])})\n")
out.close()
print("done")

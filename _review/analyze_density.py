"""分析 50k 的数据密度，判断 seen 高 / strict 低是稀疏还是过拟合。

关键量：
  1. 每书家样本数、每书家字符数
  2. (书家, 字) 对的样本数分布 —— 有多少对只有 1 张？
  3. strict 集的 (script, callig, char) 三元组在训练集里是否真的为 0
  4. strict 集里那些**字**在**其它书家**下出现过吗（决定模型能否借力）
"""
import csv
import os
import statistics as S
from collections import Counter, defaultdict

os.chdir("/root/Workspace/xy/DiT")

tr = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
st = list(csv.DictReader(open("assets/eval_v13_strict.csv", encoding="utf-8")))
print(f"  训练集 {len(tr)} 行 / 评测 strict {len(st)} 行")

cal_of = lambda r: int(r["calligrapher_id"])
char_of = lambda r: int(r.get("character_id", 0))
sc_of = lambda r: int(r["script_id"])

# ── 1. 每书家 ────────────────────────────────────────────────────────
by_cal = Counter(cal_of(r) for r in tr)
chars_per_cal = defaultdict(set)
for r in tr:
    chars_per_cal[cal_of(r)].add(char_of(r))
all_chars = set(char_of(r) for r in tr)

print()
print("  === 每书家 ===")
print(f"    样本数: min={min(by_cal.values())} med={int(S.median(by_cal.values()))} "
      f"max={max(by_cal.values())}")
cp = [len(v) for v in chars_per_cal.values()]
print(f"    覆盖字符数: min={min(cp)} med={int(S.median(cp))} max={max(cp)}")
print(f"    全局不同字符: {len(all_chars)}")
print(f"    -> 平均每个书家只覆盖 {S.mean(cp)/len(all_chars)*100:.1f}% 的字符")

# ── 2. (书家, 字) 对的密度 ───────────────────────────────────────────
pair = Counter((cal_of(r), char_of(r)) for r in tr)
print()
print("  === (书家, 字) 对的样本数分布 ===")
d = Counter(pair.values())
tot = sum(d.values())
for k in sorted(d)[:6]:
    print(f"    恰好 {k} 张: {d[k]:>6} 对  ({d[k]/tot*100:5.1f}%)")
print(f"    总对数 {tot}（= 训练样本数，每行一个对）")
print(f"    **只有 1 张样本的 (书家,字) 对: {d.get(1,0)} ({d.get(1,0)/tot*100:.1f}%)**")
print(f"    中位数 = {int(S.median(list(pair.values())))} 张/对")

# ── 3. strict 三元组是否真的为 0 ─────────────────────────────────────
trip = set((sc_of(r), cal_of(r), char_of(r)) for r in tr)
zero = sum(1 for r in st if (sc_of(r), cal_of(r), char_of(r)) not in trip)
print()
print(f"  === strict 的 (script,callig,char) 在训练集里为 0 的: "
      f"{zero}/{len(st)} ({zero/len(st)*100:.0f}%) ===")

# ── 4. strict 的字在**其它书家**下出现过吗 ────────────────────────────
char_any = Counter(char_of(r) for r in tr)
n_other = []
for r in st:
    c = char_of(r)
    # 该字在训练集里出现在多少个**不同书家**下
    cals = set(cal_of(x) for x in tr if char_of(x) == c)
    n_other.append(len(cals))
print()
print("  === strict 的每个「字」在训练集里被多少个不同书家写过 ===")
print(f"    min={min(n_other)} med={int(S.median(n_other))} max={max(n_other)}")
for t in (1, 2, 5, 10, 20):
    n = sum(1 for x in n_other if x <= t)
    print(f"    <= {t:>2} 个书家: {n:>3} / {len(n_other)} ({n/len(n_other)*100:5.1f}%)")

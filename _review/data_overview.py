"""数据全貌: 规模 / 多样性 / 均衡性, 老数据 vs HCSU vs 合并。"""
import csv, os, collections, math
os.chdir("/root/Workspace/xy/DiT")

OLD = "assets/train_fame-kxl-tj-px60.csv"
NEW = "assets/train_hcsu_kxl.csv"


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def gini(counts):
    v = sorted(counts)
    n = len(v)
    if n == 0 or sum(v) == 0:
        return 0.0
    cum = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(v))
    return cum / (n * sum(v))


def norm_entropy(counts):
    """归一化熵 ∈ [0,1]; 1 = 完全均匀。"""
    n = sum(counts)
    if n == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts if c > 0)
    return h / math.log(len(counts))


def report(tag, rows):
    n = len(rows)
    c = collections.Counter(r["calligrapher"] for r in rows)
    s = collections.Counter(r["script"] for r in rows)
    ch = collections.Counter(r["character"] for r in rows)
    print("=" * 78)
    print(f"{tag}   共 {n} 张")
    print(f"  书家 {len(c)}   书体 {len(s)}   字 {len(ch)}")
    print(f"  书体分布: " + "  ".join(f"{k}={v}({100*v/n:.1f}%)" for k, v in s.most_common()))
    print(f"\n  --- 均衡性 ---")
    print(f"  书家: Gini={gini(list(c.values())):.3f}  归一化熵={norm_entropy(list(c.values())):.3f}"
          f"  min={min(c.values())} med={int(sorted(c.values())[len(c)//2])} max={max(c.values())}")
    print(f"  字  : Gini={gini(list(ch.values())):.3f}  归一化熵={norm_entropy(list(ch.values())):.3f}"
          f"  min={min(ch.values())} med={int(sorted(ch.values())[len(ch)//2])} max={max(ch.values())}")
    # 每书家的字数分布 (决定"这个书家能学多少字形")
    cch = collections.defaultdict(set)
    for r in rows:
        cch[r["calligrapher"]].add(r["character"])
    per = sorted(len(v) for v in cch.values())
    print(f"  每书家字数: min={per[0]} med={per[len(per)//2]} max={per[-1]}")
    # 书家 × 书体 覆盖
    cs = collections.defaultdict(set)
    for r in rows:
        cs[r["calligrapher"]].add(r["script"])
    multi = sum(1 for v in cs.values() if len(v) > 1)
    print(f"  书家覆盖书体数: " + str(dict(collections.Counter(len(v) for v in cs.values()))))
    print(f"  -> {multi}/{len(cs)} 位书家跨多个书体 (跨书体越多, 风格-书体越难解耦)")
    # 书家 × 字 的稀疏度
    pair = collections.Counter((r["calligrapher"], r["character"]) for r in rows)
    print(f"  (书家,字) 组合 {len(pair)} 个, 平均每组合 {n/len(pair):.1f} 张")
    rep = sum(1 for v in pair.values() if v > 1)
    print(f"  其中重复(>1张)的组合 {rep} 个 ({100*rep/len(pair):.1f}%)")
    return dict(n=n, c=len(c), s=len(s), ch=len(ch))


o = load(OLD); w = load(NEW)
ro = report("【老数据】train_fame-kxl-tj-px60", o)
rw = report("【HCSU】train_hcsu_kxl", w)

# 合并
merged = o + w
rm = report("【合并】老 + HCSU", merged)

print("=" * 78)
print("合并视角")
oc = {r["calligrapher"] for r in o}
wc = {r["calligrapher"] for r in w}
och = {r["character"] for r in o}
wch = {r["character"] for r in w}
print(f"  书家: 老 {len(oc)} + HCSU {len(wc)} -> 合并 {len(oc | wc)}  (新增 {len(wc - oc)})")
print(f"  字  : 老 {len(och)} + HCSU {len(wch)} -> 合并 {len(och | wch)}  (新增 {len(wch - och)})")
print(f"  张数: {len(o)} + {len(w)} = {len(merged)}  ({len(merged)/len(o):.2f}x)")

# 新书家的"可训练性": 字数够不够
print()
print("=" * 78)
print("新增书家的样本/字数 (样本量决定 embedding 能否训出来)")
cch_w = collections.defaultdict(set)
cnt_w = collections.Counter()
for r in w:
    cch_w[r["calligrapher"]].add(r["character"])
    cnt_w[r["calligrapher"]] += 1
newc = sorted(wc - oc, key=lambda c: -cnt_w[c])
print(f"  {'书家':<12}{'张数':>7}{'字数':>7}{'张/字':>8}")
for c in newc:
    print(f"  {c:<12}{cnt_w[c]:>7}{len(cch_w[c]):>7}{cnt_w[c]/len(cch_w[c]):>8.1f}")
print(f"  新增书家合计 {sum(cnt_w[c] for c in newc)} 张")

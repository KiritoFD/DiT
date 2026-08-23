"""分析 top30 数据集的稀疏度与 GT 覆盖情况。"""
import csv, json, os
from collections import defaultdict, Counter

CSV = os.path.join(os.path.dirname(__file__), "..", "5script", "train_top30.csv")
CALLIGS = os.path.join(os.path.dirname(__file__), "..", "5script", "top30_calligs.json")

# 加载 callig 名单
with open(CALLIGS, encoding="utf-8") as f:
    calligs = json.load(f)
# 每个 script_id -> list of (id, name)
script_names = {0: "篆", 1: "隶", 2: "楷", 3: "行", 4: "草"}

# 读取 CSV
rows = []
with open(CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"总行数: {len(rows)}")

# 统计每个 (script_id, calligrapher_id) 的样本数
pair_count = defaultdict(int)  # (script_id, callig_id) -> count
pair_chars = defaultdict(set)  # (script_id, callig_id) -> set of character_id

for r in rows:
    sid = int(r["script_id"])
    cid = int(r["calligrapher_id"])
    char_id = int(r["character_id"])
    pair_count[(sid, cid)] += 1
    pair_chars[(sid, cid)].add(char_id)

# 全局唯一的 character_id 数量
all_chars = set()
for r in rows:
    all_chars.add(int(r["character_id"]))
print(f"全局唯一 character 数: {len(all_chars)}")

# 每个 script 的 character 数
script_chars = defaultdict(set)
for r in rows:
    script_chars[int(r["script_id"])].add(int(r["character_id"]))
for sid in range(5):
    print(f"  script {sid} ({script_names[sid]}): {len(script_chars[sid])} 个唯一 character")

print()
# 每个 script 下，top30 calligrapher 各有多少样本
for sid in range(5):
    print(f"\n=== Script {sid} ({script_names[sid]}) ===")
    callig_list = calligs[str(sid)]
    print(f"  名单: {len(callig_list)} 位书家")
    counts = []
    for c in callig_list:
        cid = int(c["id"])
        n = pair_count.get((sid, cid), 0)
        nc = len(pair_chars.get((sid, cid), set()))
        counts.append((c["name"], n, nc))
    counts.sort(key=lambda x: -x[1])
    for name, n, nc in counts:
        print(f"    {name:8s}: {n:5d} 样本, {nc:4d} 字")
    total_samples = sum(x[1] for x in counts)
    print(f"  合计: {total_samples} 样本")

# 全局 (callig_id, script_id) pair 覆盖率
print("\n\n=== GT 覆盖矩阵分析 ===")
# 每个 script 内，所有 callig 的字符交集 vs 并集
for sid in range(5):
    callig_list = calligs[str(sid)]
    # 计算并集
    union = set()
    char_sets = {}
    for c in callig_list:
        cid = int(c["id"])
        cs = pair_chars.get((sid, cid), set())
        char_sets[c["name"]] = cs
        union |= cs
    print(f"\nScript {sid} ({script_names[sid]}): 并集 {len(union)} 字")
    # 每个 callig 覆盖了多少
    for c in callig_list:
        cs = char_sets[c["name"]]
        if len(union) > 0:
            print(f"  {c['name']:8s}: {len(cs):4d}/{len(union)} = {len(cs)/len(union)*100:.1f}%")
    # 计算 pair-wise 重叠（前5位书家之间）
    top5 = [c["name"] for c in callig_list[:5]]
    print(f"  前5位书家两两重叠:")
    for i in range(len(top5)):
        for j in range(i+1, len(top5)):
            ci, cj = char_sets[top5[i]], char_sets[top5[j]]
            if ci and cj:
                overlap = len(ci & cj)
                print(f"    {top5[i]} ∩ {top5[j]}: {overlap} 字")

# 稀疏度: 每个 (script, callig, char) 三元组的样本数分布
print("\n\n=== 样本数分布（每个 glyph） ===")
glyph_counts = Counter()
for r in rows:
    gid = int(r["glyph_id"])
    glyph_counts[gid] += 1
counts_sorted = sorted(glyph_counts.values(), reverse=True)
print(f"  总 glyph 数: {len(glyph_counts)}")
print(f"  每个 glyph 平均样本数: {sum(counts_sorted)/len(counts_sorted):.2f}")
print(f"  最大: {counts_sorted[0]}, 最小: {counts_sorted[-1]}")
# 分布
dist = Counter(counts_sorted)
print(f"  分布 (样本数: glyph数):")
for n in sorted(dist.keys()):
    print(f"    {n:3d} 样本: {dist[n]:5d} glyphs ({dist[n]/len(glyph_counts)*100:.1f}%)")

# 每个字符有多少个 (script, callig) 组合的 GT
print("\n\n=== 每个字符的 GT 组合数 ===")
char_pairs = defaultdict(set)  # char_id -> set of (script_id, callig_id)
for r in rows:
    char_pairs[int(r["character_id"])].add((int(r["script_id"]), int(r["calligrapher_id"])))
combo_counts = [len(v) for v in char_pairs.values()]
combo_counts.sort(reverse=True)
print(f"  每个字符平均组合数: {sum(combo_counts)/len(combo_counts):.2f}")
print(f"  最大: {combo_counts[0]}, 最小: {combo_counts[-1]}")
cdist = Counter(combo_counts)
print(f"  分布 (组合数: 字符数):")
for n in sorted(cdist.keys()):
    print(f"    {n:3d} 组合: {cdist[n]:5d} 字 ({cdist[n]/len(combo_counts)*100:.1f}%)")

# -*- coding: utf-8 -*-
"""远程统计: 书法家样本量分布 / topN 覆盖 / 著名书法家子集切分覆盖, 并导出 famous 子集 csv。"""
import csv, sys, io, os
from collections import defaultdict

OUT = "/root/Workspace/xy/DiT"
SYS = "/opt/conda/bin/python"
inp = open(sys.argv[1], encoding="utf-8") if len(sys.argv) > 1 else open("train.csv", encoding="utf-8")
rows = list(csv.DictReader(inp))
total = len(rows)
print(f"总样本(行)数: {total}")

cname = {}
c_count = defaultdict(int)
pair_count = defaultdict(int)          # (cid, char) -> count
char_count = defaultdict(lambda: defaultdict(int))
c_script_dist = defaultdict(lambda: defaultdict(int))
script_name = {}

for r in rows:
    cid = r["calligrapher_id"]
    cname[cid] = r["calligrapher"]
    c_count[cid] += 1
    ch = r["character"]
    pair_count[(cid, ch)] += 1
    char_count[cid][ch] += 1
    c_script_dist[cid][r["script_id"]] += 1
    script_name.setdefault(r["script_id"], r["script"])

print(f"总书法家数: {len(c_count)}")
print("=== 书法家样本量分布 ===")
buckets = [(1, 1), (2, 9), (10, 49), (50, 99), (100, 499), (500, 999), (1000, 4999), (5000, 9999), (10000, 10**9)]
for lo, hi in buckets:
    n = sum(1 for c in c_count.values() if lo <= c <= hi)
    if n:
        s = sum(c for c in c_count.values() if lo <= c <= hi)
        print(f"  {lo}-{hi}张: {n}个书法家 (含{s}样本)")

by_samples = sorted(c_count.items(), key=lambda x: -x[1])
print("=== top50书法家名单 ===")
for i, (cid, n) in enumerate(by_samples[:50], 1):
    print(f"  {i}. {cname[cid]}({cid}) {n}张")

print("=== top-N 书法家(按样本量, 全部作品合并) 覆盖 ===")
for N in (10, 20, 50, 60, 100, 200, 500):
    top_cs = set(cid for cid, _ in by_samples[:N])
    cov = sum(c for cid, c in c_count.items() if cid in top_cs)
    print(f"  top{N} 书法家: {cov}张 ({cov/total*100:.2f}%)")

print("=== 每个书法家各取 top60/top100 高频字 独立覆盖 ===")
for K in (60, 100):
    cover = 0
    for cid in c_count:
        top_chars = {c for c, _ in sorted(char_count[cid].items(), key=lambda x: -x[1])[:K]}
        cover += sum(cnt for (cc, ch), cnt in pair_count.items() if cc == cid and ch in top_chars)
    print(f"  每人取 top{K} 字: {cover}张 ({cover/total*100:.2f}%)")

# ---- 著名书法家名单 ----
FAMOUS = [
    # 二王 + 楷书四大家
    "王羲之", "王献之", "颜真卿", "柳公权", "欧阳询", "虞世南", "褚遂良", "薛稷", "欧阳通",
    "赵孟頫", "颜真卿", "柳公绰", "李邕", "钟繇", "张芝",
    # 宋四家
    "苏轼", "黄庭坚", "米芾", "蔡襄",
    # 草书
    "怀素", "张旭", "孙过庭", "智永", "王铎", "傅山",
    # 明清
    "董其昌", "文徵明", "唐寅", "祝允明", "徐渭", "何绍基", "刘墉", "郑板桥", "金农",
    # 篆隶
    "李斯", "程邈", "蔡邕", "邓石如", "伊秉绶", "赵之谦", "吴昌硕",
    # 近现代
    "于右任", "启功", "林散之", "沈尹默", "沙孟海", "康有为", "郭沫若",
    # 其他
    "杨凝式", "鲜于枢", "康里巎巎", "朱耷", "石涛", "翁同龢", "张之洞",
    "怀仁", "李阳冰", "颜延之", "黄易",
]
FAMOUS = list(dict.fromkeys(FAMOUS))
print(f"\n候选名单(去重后): {len(FAMOUS)} 人")

name_to_ids = defaultdict(list)
for cid, n in cname.items():
    name_to_ids[n].append(cid)

found = {}      # name -> cid
not_found = []
for nm in FAMOUS:
    if nm in name_to_ids:
        found[nm] = name_to_ids[nm][0]
    else:
        not_found.append(nm)

if not_found:
    print(f"[名单中未找到 {len(not_found)} 人]: {', '.join(not_found)}")
    # 尝试包含式匹配(去空格/同音别名), 报告候选
    print("  尝试子串匹配候选:")
    for nm in not_found:
        cands = [(n, sum(c_count[i] for i in ids)) for n, ids in name_to_ids.items()
                 if len(nm) >= 2 and nm[:2] in n or len(n) >= 2 and n in nm]
        cands = [(n, c) for n, c in cands if len(n) >= 2][:5]
        if cands:
            print(f"    {nm} -> {cands}")

cov = sum(c_count[cid] for cid in found.values())
print(f"\n实际精确匹配: {len(found)} 人, 覆盖样本: {cov}张 ({cov/total*100:.2f}%)")

print("\n=== 匹配到的著名书法家明细 ===")
for nm, cid in sorted(found.items(), key=lambda x: -c_count[x[1]]):
    chars = len(char_count[cid])
    sc = sorted(c_script_dist[cid].items(), key=lambda x: -x[1])[:3]
    scs = ",".join(f"{script_name.get(s, s)}:{n}" for s, n in sc)
    print(f"  {nm}({cid}) {c_count[cid]}张 独字{chars} 书体[{scs}]")

# ---- 导出 famous 子集 csv ----
keep_cids = set(found.values())
subset = [r for r in rows if r["calligrapher_id"] in keep_cids]
out_csv = os.path.join(OUT, "famous_calligrapher.csv")
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(subset)
print(f"\n已导出: {out_csv} ({len(subset)} 行)")

# 每人的字覆盖
name_rows = {}
for cid in keep_cids:
    name_rows[cname[cid]] = (cid, c_count[cid], len(char_count[cid]))
cov_char = sum(len(char_count[cid]) for cid in keep_cids)
print(f"独字总数合计: {cov_char}")
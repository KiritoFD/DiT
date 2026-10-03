"""分析 HCSU wild.zip 的目录结构（不解压），并与我们现有数据集对比。

目录/文件名用 #Uxxxx 转义: `{书家}-{书体}/{字}.png`
"""
import re, os, csv, collections, sys

LIST = "/root/Workspace/xy/HCSU/wild_listing.txt"
OUR_CSV = "/root/Workspace/xy/DiT/assets/train_fame-kxl-tj-px60.csv"

U = re.compile(r"#U([0-9a-fA-F]{4})")


def dec(s):
    return U.sub(lambda m: chr(int(m.group(1), 16)), s)


rows = []
with open(LIST, encoding="utf-8", errors="replace") as f:
    for line in f:
        p = line.split()
        if len(p) < 4 or not p[0].isdigit():
            continue
        size, name = int(p[0]), p[3]
        if name.endswith("/"):
            continue
        rows.append((name, size))

print(f"文件总数: {len(rows)}")
print(f"总大小  : {sum(s for _, s in rows)/1e9:.2f} GB")

per_dir = collections.Counter()
per_callig = collections.Counter()
per_script = collections.Counter()
chars = collections.Counter()
tiny = []
for name, size in rows:
    d, _, fn = name.rpartition("/")
    per_dir[d] += 1
    if "-" in d:
        c, _, s = dec(d).rpartition("-")
    else:
        c, s = dec(d), "?"
    per_callig[c] += 1
    per_script[s] += 1
    ch = dec(os.path.splitext(fn)[0])
    chars[ch] += 1
    if size < 5000:
        tiny.append((name, size))

print(f"\n目录(书家-书体)数: {len(per_dir)}")
print(f"书家数: {len(per_callig)}   书体数: {len(per_script)}")
print(f"不重复字数: {len(chars)}")

print("\n=== 书体分布 ===")
for s, n in per_script.most_common():
    print(f"  {s}: {n}")

print("\n=== 书家分布 (top 20) ===")
for c, n in per_callig.most_common(20):
    print(f"  {c}: {n}")

print("\n=== 每个 书家-书体 组合的样本数分布 ===")
cnt = collections.Counter(per_dir.values())
for k in sorted(cnt):
    print(f"  {k:>4} 张: {cnt[k]} 个组合")

print(f"\n=== 可疑小文件 (<5KB, 可能是空白图) ===")
print(f"  共 {len(tiny)} 个 ({100*len(tiny)/len(rows):.1f}%)")
for n, s in tiny[:10]:
    print(f"    {s:>7} B  {n}")

# ---- 与我们现有数据集对比 ----
if os.path.exists(OUR_CSV):
    with open(OUR_CSV, encoding="utf-8") as f:
        our = list(csv.DictReader(f))
    print(f"\n=== 与我们现有数据对比 ({os.path.basename(OUR_CSV)}, {len(our)} 行) ===")
    cols = our[0].keys()
    print(f"  csv 列: {list(cols)[:12]}")
    oc = collections.Counter()
    ochar = collections.Counter()
    for r in our:
        oc[r.get("calligrapher", "") or r.get("calligrapher_id", "")] += 1
        ochar[r.get("character", r.get("glyph", ""))] += 1
    print(f"  我们的书家数: {len(oc)}   字数: {len(ochar)}")
    print(f"  我们的书家: {sorted(oc)[:40]}")
    inter_c = set(per_callig) & set(oc)
    inter_ch = set(chars) & set(ochar)
    print(f"\n  书家交集: {len(inter_c)} 个 -> {sorted(inter_c)}")
    print(f"  字交集  : {len(inter_ch)} 个 (wild {len(chars)} / 我们 {len(ochar)})")
    print(f"  **新增书家**: {len(set(per_callig) - set(oc))} 个 -> {sorted(set(per_callig)-set(oc))}")
    print(f"  **新增字**  : {len(set(chars) - set(ochar))} 个")

"""精确找出"std 与真迹不是同一个字"的行 —— 不需要 OCR。

原理: 标注管线把 繁/简/异体 合并成了同一个 `glyph_id`(如 54116 -> {復, 复}),
      而 std 只能渲染其中一个字形 -> 同组里与渲染字形不同的那些行, 条件就是错的。
判据: 按 glyph_id 分组, 组内 `character` 取值 > 1 的组 = 被合并的字;
      组内取"多数派字形"当作**std 实际渲染的字形**(若渲染用的是少数派则反过来,
      故再按 src_image_path 的文件名字形交叉验证一次)。
纯 CPU, 秒级。
"""
import csv
import os
import re
from collections import Counter, defaultdict

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

rows = list(csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8")))
print(f"[in] train.csv n={len(rows)}")

grp = defaultdict(list)
for r in rows:
    grp[str(r["glyph_id"])].append(r)

collapse = {g: v for g, v in grp.items() if len({x["character"] for x in v}) > 1}
n_rows = sum(len(v) for v in collapse.values())
print(f"[合并字组] {len(collapse)} 个 glyph_id 对应多个 character, 涉及 {n_rows} 行 "
      f"({n_rows / len(rows):.2%})")

print("\n[样例] 前 12 组:")
for g, v in list(collapse.items())[:12]:
    cc = Counter(x["character"] for x in v)
    # 文件名里的字形 (src_image_path 的 stem) —— 大概率就是渲染 std 时用的字形
    base = Counter(re.sub(r"\.png$", "", os.path.basename(x["src_image_path"]))
                   for x in v)
    print(f"  glyph_id={g:<7} rows={len(v):<4} character={dict(cc)}  "
          f"文件名={dict(base)}")

# 与文件名不一致的行 = 几乎肯定拿到了错的 std
bad = []
for g, v in collapse.items():
    for r in v:
        stem = re.sub(r"\.png$", "", os.path.basename(r["src_image_path"]))
        if stem and stem != r["character"]:
            bad.append(r)
print(f"\n[可疑行] character != src 文件名 的 = {len(bad)} 行 "
      f"({len(bad) / len(rows):.2%}) —— 这些行的 std 基本可以确定是错的字形")
print("[可疑行] 按槽位:")
for s, c in Counter(r["slot_name"] for r in bad).most_common(12):
    print(f"   {s:<14} {c}")

with open("_ot_scratch/glyph_collapse_bad_rows.csv", "w", newline="",
          encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(bad)
print(f"\n[落盘] _ot_scratch/glyph_collapse_bad_rows.csv ({len(bad)} 行)")

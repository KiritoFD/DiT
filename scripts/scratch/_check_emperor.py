# -*- coding: utf-8 -*-
"""只加入帝王 + 合并，输出各书家张数，标出 <300 需考虑淘汰"""
import csv
from collections import Counter, defaultdict

MERGE = {"赵孟": "赵孟頫", "郑燮": "郑板桥", "孫過庭": "孙过庭"}
DROP = {"柳公權"}

ORIG = {
    "王羲之", "苏轼", "欧阳询", "颜真卿", "智永", "米芾", "褚遂良", "黄庭坚", "文徵明",
    "孙过庭", "怀素", "李阳冰", "祝允明", "何绍基", "赵之谦", "邓石如", "吴昌硕", "李邕",
    "王铎", "金农", "沈尹默", "柳公权", "鲜于枢", "王献之", "赵孟頫", "董其昌", "唐寅",
    "欧阳通", "蔡襄", "虞世南", "李斯", "薛稷", "朱耷", "张旭", "伊秉绶", "钟繇",
    "于右任", "黄易", "徐渭", "刘墉", "杨凝式", "傅山", "张芝", "怀仁", "翁同龢",
    "石涛", "沙孟海", "郑板桥", "康有为", "启功", "林散之",
}
EMPEROR = {"赵佶", "赵构", "李世民", "乾隆", "武则天", "康熙"}
TARGET = ORIG | EMPEROR

cnt = Counter()
uniq = defaultdict(set)
with open("train.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        c = row["calligrapher"]
        if c in DROP:
            continue
        c = MERGE.get(c, c)
        if c in TARGET:
            cnt[c] += 1
            uniq[c].add(row["character"])

total = 298281
print(f"目标书家数: {len(cnt)}  覆盖样本: {sum(cnt.values()):,} ({sum(cnt.values())/total*100:.2f}%)  独字: {len(set().union(*uniq.values())):,}")
print("=" * 60)
print(f"{'书家':<10}{'张数':>8}{'独字':>7}{'<200?':>8}")
for k, v in cnt.most_common():
    flag = "  <== 淘汰候选" if v < 200 else ""
    print(f"{k:<10}{v:>8}{len(uniq[k]):>7}{flag}")

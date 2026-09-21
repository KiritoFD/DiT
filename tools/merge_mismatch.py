"""汇总 7B 对低置信度样本的识别，找出更多简繁错配。

判据（同前）: s2t(char_csv) == char_vlm 且 != char_csv  -> csv 标简体但 GT 写繁体

同时统计"7B 与 csv 一致"的比例，说明 OCR 低置信度里有多少是 OCR 自己的问题。
"""
import csv
from collections import Counter

from opencc import OpenCC

s2t = OpenCC("s2t")
t2s = OpenCC("t2s")

vlm = list(csv.DictReader(open("assets/vlm_7b_lowconf.csv", encoding="utf-8")))
print(f"  7B 低置信度结果: {len(vlm)}")

ok = sum(1 for r in vlm if r["char_vlm"] == r["char_csv"])
print(f"\n  === 与 csv 一致 ===")
print(f"    一致: {ok}/{len(vlm)} = {ok/len(vlm)*100:.1f}%")
print(f"    (说明 OCR 低置信度里约 {ok/len(vlm)*100:.0f}% 是 OCR 自己认错了，csv 是对的)")

# 真错配: csv 简体 -> VLM 繁体
real = [r for r in vlm
        if r["char_vlm"] != r["char_csv"]
        and s2t.convert(r["char_csv"]) == r["char_vlm"]
        and s2t.convert(r["char_csv"]) != r["char_csv"]]
print(f"\n  ★ 低置信度里新发现的简繁错配: {len(real)}")

# 反向: csv 繁体 -> VLM 简体
rev = [r for r in vlm
       if r["char_vlm"] != r["char_csv"]
       and t2s.convert(r["char_csv"]) == r["char_vlm"]
       and t2s.convert(r["char_csv"]) != r["char_csv"]]
print(f"    反向(csv繁体/VLM简体): {len(rev)}")

print(f"\n  === 新发现错配的字符对 Top20 ===")
for (a, b), n in Counter((r["char_csv"], r["char_vlm"])
                         for r in real).most_common(20):
    print(f"    {a} -> {b}: {n}")

print(f"\n  === 新发现错配按书体 ===")
print(f"    {dict(Counter(r['script'] for r in real))}")

# 合并（已确认的 752 + 新发现的）
old = list(csv.DictReader(open("assets/mismatch_confirmed.csv",
                               encoding="utf-8")))
old_ip = {r["image_path"] for r in old}
new_only = [r for r in real if r["image_path"] not in old_ip]
print(f"\n  === 合并 ===")
print(f"    之前确认: {len(old)}")
print(f"    低置信度新增: {len(new_only)}")
print(f"    合计: {len(old) + len(new_only)}")

# 导出合并后的总清单
allfix = {}
for r in old:
    allfix[r["image_path"]] = {"idx": r["idx"], "image_path": r["image_path"],
                               "script": r["script"],
                               "calligrapher": r["calligrapher"],
                               "char_csv": r["char_csv"],
                               "char_true": r["char_true"],
                               "source": "ocr_hi_conf+vlm7b"}
for r in new_only:
    allfix[r["image_path"]] = {"idx": r["idx"], "image_path": r["image_path"],
                               "script": r["script"],
                               "calligrapher": r["calligrapher"],
                               "char_csv": r["char_csv"],
                               "char_true": r["char_vlm"],
                               "source": "ocr_lo_conf+vlm7b"}

with open("assets/mismatch_all.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_true", "source"])
    for v in allfix.values():
        w.writerow([v["idx"], v["image_path"], v["script"], v["calligrapher"],
                    v["char_csv"], v["char_true"], v["source"]])
print(f"\n  ✓ written assets/mismatch_all.csv ({len(allfix)} 条)")
print(f"    占全量 {len(allfix)/50786*100:.2f}%")

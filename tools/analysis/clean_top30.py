"""清洗 top30 数据集：去掉 others / 墨迹 / 唐 三个聚合标签。
保留碑刻、篆刻（它们是真实的风格标签）。

输出：
  - train_top30_clean.csv
  - eval100_top30_clean.csv
  - top30_calligs_clean.json （更新书家名单）
  - 清洗报告
"""
import csv, json, os
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

TRAIN_CSV = os.path.join(ROOT, "5script", "train_top30.csv")
EVAL_CSV = os.path.join(ROOT, "5script", "eval100_top30.csv")
CALLIGS = os.path.join(ROOT, "5script", "top30_calligs.json")

TRAIN_OUT = os.path.join(ROOT, "5script", "train_top30_clean.csv")
EVAL_OUT = os.path.join(ROOT, "5script", "eval100_top30_clean.csv")
CALLIGS_OUT = os.path.join(ROOT, "5script", "top30_calligs_clean.json")

SCRIPT_NAMES = {0: "楷", 1: "篆", 2: "草", 3: "行", 4: "隶"}

# 要清除的标签
DIRTY = {"others", "墨迹", "唐"}

# 读取原始数据
with open(CALLIGS, encoding="utf-8") as f:
    calligs = json.load(f)

train_rows = []
with open(TRAIN_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        train_rows.append(r)

eval_rows = []
with open(EVAL_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        eval_rows.append(r)

# 过滤
clean_train = [r for r in train_rows if r["calligrapher"] not in DIRTY]
clean_eval = [r for r in eval_rows if r["calligrapher"] not in DIRTY]

# 更新 calligs.json：移除 dirty 标签
clean_calligs = {}
removed_per_script = {}
for sid_str, lst in calligs.items():
    sid = int(sid_str)
    kept = []
    removed = []
    for c in lst:
        if c["name"] in DIRTY:
            removed.append(c)
        else:
            kept.append(c)
    clean_calligs[sid_str] = kept
    removed_per_script[sid] = removed

# 写文件
with open(TRAIN_OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=train_rows[0].keys())
    w.writeheader()
    w.writerows(clean_train)

with open(EVAL_OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
    w.writeheader()
    w.writerows(clean_eval)

with open(CALLIGS_OUT, "w", encoding="utf-8") as f:
    json.dump(clean_calligs, f, ensure_ascii=False, indent=2)

# 报告
print("=" * 60)
print("清洗报告")
print("=" * 60)
print(f"\n清除标签: {DIRTY}")
print(f"保留标签: 碑刻, 篆刻 (真实风格)\n")

# 原始统计
orig_glyphs = set(int(r["glyph_id"]) for r in train_rows)
clean_glyphs = set(int(r["glyph_id"]) for r in clean_train)
print(f"train CSV:")
print(f"  原始: {len(train_rows):>6} 样本, {len(orig_glyphs):>5} glyphs")
print(f"  清洗: {len(clean_train):>6} 样本, {len(clean_glyphs):>5} glyphs")
print(f"  删除: {len(train_rows)-len(clean_train):>6} 样本 "
      f"({(len(train_rows)-len(clean_train))/len(train_rows)*100:.1f}%), "
      f"{len(orig_glyphs)-len(clean_glyphs):>4} glyphs")

print(f"\neval100 CSV:")
print(f"  原始: {len(eval_rows):>4} 样本")
print(f"  清洗: {len(clean_eval):>4} 样本")
print(f"  删除: {len(eval_rows)-len(clean_eval):>4} 样本")

print(f"\ncalligs.json 更新:")
for sid in range(5):
    orig_n = len(calligs[str(sid)])
    clean_n = len(clean_calligs[str(sid)])
    removed = removed_per_script[sid]
    removed_names = [c["name"] for c in removed]
    print(f"  Script {sid} ({SCRIPT_NAMES[sid]}): {orig_n} -> {clean_n} 书家, "
          f"移除: {removed_names}")

# 清洗后的 eval GT 覆盖
clean_index = set()
for r in clean_train:
    clean_index.add((int(r["script_id"]), int(r["calligrapher_id"]), int(r["character_id"])))

if clean_eval:
    found = sum(1 for r in clean_eval
                if (int(r["script_id"]), int(r["calligrapher_id"]), int(r["character_id"]))
                in clean_index)
    print(f"\neval100_clean GT 覆盖: {found}/{len(clean_eval)} "
          f"({found/len(clean_eval)*100:.1f}%)")

# 各 script 清洗后样本数
print(f"\n各书体清洗后:")
for sid in range(5):
    orig = sum(1 for r in train_rows if int(r["script_id"]) == sid)
    clean = sum(1 for r in clean_train if int(r["script_id"]) == sid)
    print(f"  {SCRIPT_NAMES[sid]}: {orig} -> {clean} "
          f"(删除 {orig-clean}, {(orig-clean)/orig*100:.1f}%)")

print(f"\n输出文件:")
print(f"  {TRAIN_OUT}")
print(f"  {EVAL_OUT}")
print(f"  {CALLIGS_OUT}")

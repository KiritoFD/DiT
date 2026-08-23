"""聚焦分析：脏数据识别 + eval100 GT 覆盖率"""
import csv, json, os
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "..", "5script", "train_top30.csv")
EVAL_CSV = os.path.join(HERE, "..", "5script", "eval100_top30.csv")
CALLIGS = os.path.join(HERE, "..", "5script", "top30_calligs.json")

SCRIPT_NAMES = {0: "楷", 1: "篆", 2: "草", 3: "行", 4: "隶"}

# 非具体书家的聚合标签
VAGUE_LABELS = {"others", "墨迹", "碑刻", "篆刻", "唐"}

with open(CALLIGS, encoding="utf-8") as f:
    calligs = json.load(f)

rows = []
with open(TRAIN_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# 1. 聚合标签分析
print("=" * 60)
print("1. 聚合/模糊标签统计（非具体书家）")
print("=" * 60)
vague_count = defaultdict(lambda: defaultdict(int))
vague_chars = defaultdict(lambda: defaultdict(set))
clean_count = defaultdict(int)
for r in rows:
    name = r["calligrapher"]
    sid = int(r["script_id"])
    char_id = int(r["character_id"])
    if name in VAGUE_LABELS:
        vague_count[sid][name] += 1
        vague_chars[sid][name].add(char_id)
    else:
        clean_count[sid] += 1

total_vague = 0
total_clean = 0
for sid in range(5):
    print(f"\n  Script {sid} ({SCRIPT_NAMES[sid]}):")
    v_sum = 0
    for name in sorted(vague_count[sid].keys()):
        n = vague_count[sid][name]
        nc = len(vague_chars[sid][name])
        print(f"    {name:6s}: {n:5d} 样本, {nc:4d} 字  ← 模糊")
        v_sum += n
    print(f"    具体书家: {clean_count[sid]:5d} 样本")
    total_vague += v_sum
    total_clean += clean_count[sid]
print(f"\n  总计: 模糊={total_vague}, 具体={total_clean}, "
      f"模糊占比={total_vague/(total_vague+total_clean)*100:.1f}%")

# 2. eval100 GT 覆盖率
print("\n" + "=" * 60)
print("2. eval100 GT 覆盖率")
print("=" * 60)
eval_rows = []
with open(EVAL_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        eval_rows.append(r)

# 构建 train 索引: (script_id, calligrapher_id, character_id) -> count
train_index = defaultdict(int)
for r in rows:
    key = (int(r["script_id"]), int(r["calligrapher_id"]), int(r["character_id"]))
    train_index[key] += 1

# eval100 中每个样本的 GT 是否存在于 train set
gt_found = 0
gt_missing = 0
gt_missing_detail = []
for r in eval_rows:
    key = (int(r["script_id"]), int(r["calligrapher_id"]), int(r["character_id"]))
    if key in train_index:
        gt_found += 1
    else:
        gt_missing += 1
        gt_missing_detail.append(r)

print(f"  eval100 总样本: {len(eval_rows)}")
print(f"  GT 在 train 中找到: {gt_found} ({gt_found/len(eval_rows)*100:.1f}%)")
print(f"  GT 缺失: {gt_missing} ({gt_missing/len(eval_rows)*100:.1f}%)")
if gt_missing_detail:
    print(f"\n  缺失 GT 明细:")
    for r in gt_missing_detail:
        print(f"    {r['calligrapher']:8s} {SCRIPT_NAMES[int(r['script_id'])]} "
              f"{r['character']} (callig_id={r['calligrapher_id']}, "
              f"script_id={r['script_id']}, char_id={r['character_id']})")

# 3. eval100 中模糊标签占比
print("\n" + "=" * 60)
print("3. eval100 中的模糊标签占比")
print("=" * 60)
eval_vague = 0
eval_clean = 0
for r in eval_rows:
    if r["calligrapher"] in VAGUE_LABELS:
        eval_vague += 1
    else:
        eval_clean += 1
print(f"  模糊标签: {eval_vague} ({eval_vague/len(eval_rows)*100:.1f}%)")
print(f"  具体书家: {eval_clean} ({eval_clean/len(eval_rows)*100:.1f}%)")

# 4. 单样本 glyph 的分析
print("\n" + "=" * 60)
print("4. 单样本 glyph（只有1个 GT 的字）")
print("=" * 60)
glyph_samples = defaultdict(list)
for r in rows:
    gid = int(r["glyph_id"])
    glyph_samples[gid].append(r)

single_glyph = {gid: rs[0] for gid, rs in glyph_samples.items() if len(rs) == 1}
print(f"  单样本 glyph: {len(single_glyph)} / {len(glyph_samples)} "
      f"({len(single_glyph)/len(glyph_samples)*100:.1f}%)")
# 这些单样本中有多少是模糊标签
single_vague = sum(1 for r in single_glyph.values() if r["calligrapher"] in VAGUE_LABELS)
print(f"  其中模糊标签: {single_vague} ({single_vague/len(single_glyph)*100:.1f}%)")
print(f"  这些单样本无法构成有效的 style transfer pair")

# 5. 如果去掉模糊标签，数据量和覆盖率变化
print("\n" + "=" * 60)
print("5. 去掉模糊标签后的影响")
print("=" * 60)
clean_rows = [r for r in rows if r["calligrapher"] not in VAGUE_LABELS]
print(f"  原始: {len(rows)} 样本, {len(glyph_samples)} glyphs")
clean_glyphs = set()
for r in clean_rows:
    clean_glyphs.add(int(r["glyph_id"]))
print(f"  去模糊后: {len(clean_rows)} 样本, {len(clean_glyphs)} glyphs")

# eval100 去模糊后覆盖率
clean_train_index = defaultdict(int)
for r in clean_rows:
    key = (int(r["script_id"]), int(r["calligrapher_id"]), int(r["character_id"]))
    clean_train_index[key] += 1

clean_gt_found = sum(1 for r in eval_rows
                     if (int(r["script_id"]), int(r["calligrapher_id"]), int(r["character_id"]))
                     in clean_train_index)
print(f"  eval100 GT 覆盖: {clean_gt_found}/{len(eval_rows)} "
      f"({clean_gt_found/len(eval_rows)*100:.1f}%)")

# 6. 每个书家的"有效字数"（字数>=2才可能用于 pairwise 训练）
print("\n" + "=" * 60)
print("6. 各书家有效字数分布（>=2样本才算有效）")
print("=" * 60)
for sid in range(5):
    callig_list = calligs[str(sid)]
    print(f"\n  Script {sid} ({SCRIPT_NAMES[sid]}):")
    for c in callig_list:
        cid = int(c["id"])
        # 统计该 (script, callig) 下每个 char 的样本数
        char_sample = defaultdict(int)
        for r in rows:
            if int(r["script_id"]) == sid and int(r["calligrapher_id"]) == cid:
                char_sample[int(r["character_id"])] += 1
        valid_chars = sum(1 for n in char_sample.values() if n >= 2)
        total_chars = len(char_sample)
        if total_chars > 0:
            flag = " ←模糊" if c["name"] in VAGUE_LABELS else ""
            print(f"    {c['name']:8s}: {total_chars:4d}字, "
                  f"有效(≥2): {valid_chars:4d} ({valid_chars/total_chars*100:.0f}%){flag}")

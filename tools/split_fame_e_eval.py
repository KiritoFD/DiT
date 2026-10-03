"""从 fame_e 训练 csv 切一份 **真 held-out** 评测集 (供 v46 采样课程用)。

三个约束, 缺一不可:
  1. 书家必须在 callig_id_map["id_map"] 词表内 (否则图缓存构建会硬拦, 或静默用错书家);
  2. held-out 行必须**从训练 csv 里剔除** (否则是训练集内成绩, 没有说服力);
  3. 不按任何质量/难度筛选 —— 只按 img_id 的稳定哈希分桶 (可复现, 与模型无关)。

注: 训练 csv 无 img_id 列, id 从 image_path 的 basename 取 (与 MCCDLatentDataset 同规则)。
"""
import csv
import hashlib
import json
import os
import re

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
SRC = "assets/train_fame3_e_full.csv"
CMAP = "assets/callig_id_map.json"
TR_OUT = "assets/train_fame3_e_split.csv"
EV_STRICT = "assets/eval_v46_strict200.csv"
EV_SEEN = "assets/eval_v46_seen20.csv"
N_STRICT, N_SEEN = 200, 20
SEED = "v46"


def img_id(r):
    m = re.search(r"(\d+)\.png$", r.get("image_path", "") or "")
    return int(m.group(1)) if m else -1


cmap = json.load(open(CMAP, encoding="utf-8"))
vocab = {str(k) for k in cmap["id_map"]}
with open(SRC, encoding="utf-8") as f:
    rd = csv.DictReader(f)
    fns = rd.fieldnames
    rows = list(rd)
print(f"[in] {SRC} rows={len(rows)} | 词表 {len(vocab)} 类 (num={cmap.get('num_calligraphers')})")

in_vocab = [r for r in rows if str(r.get("calligrapher_id", "")).strip() in vocab]
print(f"[vocab] in-vocab rows = {len(in_vocab)} / {len(rows)}")
assert in_vocab, "词表内一条都没有 —— 检查 calligrapher_id 与 id_map 的对应"


def h(r):
    return int(hashlib.md5((SEED + str(img_id(r))).encode()).hexdigest()[:8], 16)


in_vocab.sort(key=h)
held = in_vocab[:N_STRICT + N_SEEN]
held_ids = {img_id(r) for r in held}
train = [r for r in rows if img_id(r) not in held_ids]

seen, strict = held[:N_SEEN], held[N_SEEN:]
for path, data in ((EV_SEEN, seen), (EV_STRICT, strict), (TR_OUT, train)):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(data)
    print(f"[out] {path} rows={len(data)}")

cov = {}
for r in held:
    cov[r.get("calligrapher")] = cov.get(r.get("calligrapher"), 0) + 1
print(f"[cov] held-out {len(held)} 条覆盖书家 {len(cov)} 类: "
      + ", ".join(f"{k}:{v}" for k, v in sorted(cov.items(), key=lambda x: -x[1])[:10]))
print(f"[cov] 训练集 {len(train)} 条 / 书家 {len({r.get('calligrapher') for r in train})} 类")
print(f"[chk] held-out 与训练集 img_id 重叠 = {len(held_ids & {img_id(r) for r in train})}")

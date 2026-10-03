"""切 exp-std 的 seen-20 集: 从**训练集** train.csv 里按 23 槽位分层取 20 条(已见样本)。

用途: 与 eval200(留出集) **同构造、同判分**, 给出 train/held-out 落差。
  - 若 seen 指标 >> eval200 指标 -> 纯泛化缺口;
  - 若 seen 指标也很低   -> 欠拟合/条件通路本身没学到 (比泛化问题更严重)。
构造与 eval200 对齐的 4 条:
  1) 同一份 train.csv(26,002) 里取, 与 eval200 的 img_id 不重叠;
  2) 槽位分层: 按 eval200 的 23 槽位分布做**比例分配**(余数给样本最多的槽), 保证覆盖;
  3) 同一 source 家族优先(eval200 里出现过的 source 才用), 保证难度可比;
  4) 优先 augmented 为空的行(原图), 不足时再用增强行。
输出: exp-std/csv/seen20.csv (列与 eval200.csv 一致, 可直接喂 build_eval_cache.py)
"""
import csv
import json
import os
import random
from collections import Counter, defaultdict

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
SEED = 20261003
N = 20

TRAIN = "exp-std/csv/train.csv"
EVAL = "exp-std/csv/eval200.csv"
SLOTMAP = "exp-std/csv/callig_script_id_map_top10.json"
OUT = "exp-std/csv/seen20.csv"

tr = list(csv.DictReader(open(TRAIN, encoding="utf-8")))
ev = list(csv.DictReader(open(EVAL, encoding="utf-8")))
ev_ids = {r["img_id"] for r in ev}
csmap = json.load(open(SLOTMAP, encoding="utf-8"))
pair_map = csmap["pair_map"]

print(f"[in] train={len(tr)}  eval200={len(ev)}  eval200_ids={len(ev_ids)}")
print(f"[in] train ∩ eval200 = {len({r['img_id'] for r in tr} & ev_ids)}")

# ── 1. eval200 的槽位分布 + source 白名单 ─────────────────────────────
ev_slot = Counter(r["slot_name"] for r in ev)
ev_src = set(r["source"] for r in ev)
print(f"[eval200] 槽位 {len(ev_slot)} 个; source 白名单 {sorted(ev_src)}")

# ── 2. 候选池: 训练集里、非 eval200、source 在白名单 的行 ──────────────
pool = defaultdict(list)
for r in tr:
    if r["img_id"] in ev_ids:
        continue
    if r["source"] not in ev_src:
        continue
    pool[r["slot_name"]].append(r)
print(f"[pool] 覆盖槽位 {len(pool)} 个, 共 {sum(len(v) for v in pool.values())} 条候选")

# ── 3. 按 eval200 分布做比例分配 (最大余数法), 保证 20 条 ──────────────
slots = sorted(ev_slot, key=lambda s: (-ev_slot[s], s))
quota, rem = {}, []
for s in slots:
    exact = N * ev_slot[s] / len(ev)
    quota[s] = int(exact)
    rem.append((exact - int(exact), s))
for _, s in sorted(rem, reverse=True)[:N - sum(quota.values())]:
    quota[s] += 1
quota = {s: q for s, q in quota.items() if q > 0}
print(f"[quota] 目标分配 (共 {sum(quota.values())}): " +
      ", ".join(f"{s}:{q}" for s, q in quota.items()))

# ── 4. 抽样: 优先原图(aug 空), 再按 source 与 eval200 的分布对齐 ──────
rng = random.Random(SEED)
picked = []
for s in slots:
    q = quota.get(s, 0)
    if q == 0:
        continue
    cand = pool.get(s, [])
    if not cand:
        print(f"[warn] 槽位 {s} 无候选, 跳过")
        continue
    orig = [r for r in cand if not (r.get("aug") or "").strip()]
    use = orig if len(orig) >= q else cand
    # 按 source 分布加权: 优先 eval200 里该槽位出现过的 source
    pick = rng.sample(use, min(q, len(use)))
    picked += pick
    print(f"   {s:<12} 取 {len(pick)}/{q}  (候选 {len(cand)}, 其中原图 {len(orig)})")

# ── 5. 写盘: 列与 eval200.csv 完全一致 ────────────────────────────────
cols = list(ev[0].keys())
out_rows = []
for r in picked:
    row = {c: r.get(c, "") for c in cols}
    if not row.get("pair_id"):
        key = f"{r['calligrapher_id']}:{r['script_id']}"
        row["pair_id"] = str(pair_map.get(key, 0))
    out_rows.append(row)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(out_rows)

# ── 6. 自检 ──────────────────────────────────────────────────────────
tr_ids = {r["img_id"] for r in tr}
o_ids = [r["img_id"] for r in out_rows]
sl = Counter(r["slot_name"] for r in out_rows)
print(f"\n[out] {OUT}: {len(out_rows)} 条, 槽位覆盖 {len(sl)}/{len(ev_slot)}")
print(f"[check] 全部来自训练集 = {sum(i in tr_ids for i in o_ids)}/{len(o_ids)}"
      f"  与 eval200 重叠 = {len(set(o_ids) & ev_ids)}  去重后 = {len(set(o_ids))}")
print(f"[check] source 分布: {dict(Counter(r['source'] for r in out_rows))}")
print(f"[check] eval200 source 分布: {dict(Counter(r['source'] for r in ev))}")
print(f"[check] 槽位: {dict(sl)}")

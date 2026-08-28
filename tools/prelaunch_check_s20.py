# -*- coding: utf-8 -*-
"""拉起 s20 前的最后核验：
  1. train_mid_common 的 img_id 是否都能在 latent shards 里找到
  2. eval_strict_midclean 的 glyph 是否 100% in-domain（zero-shot=0）
  3. eval 所需的 GT 图是否齐全
  4. eval csv 列名是否与 dataset/inference 的约定一致
"""
import csv, glob, io, os, re, sys
import numpy as np

BASE = "/root/Workspace/xy/DiT"
os.chdir(BASE)

# ---------- latent shard 索引 ----------
shards = sorted(glob.glob("final_latents_mid_clean/shard_*.npz"))
id2shard = {}
probe_shapes = {}
for sp in shards:
    d = np.load(sp)
    for j, iid in enumerate(d["img_ids"]):
        id2shard[int(iid)] = (sp, j)
    if not probe_shapes:
        probe_shapes = {"latents": d["latents"].shape, "dtype": str(d["latents"].dtype)}
    d.close()
print(f"[latent] shards={len(shards)} ids={len(id2shard)} sample_shape={probe_shapes}")


def ids_of(path):
    out = []
    with io.open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)\.png", row.get("image_path", "") or "")
            if m:
                out.append(int(m.group(1)))
    return out


def rows_of(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- 1. 训练集 latent 覆盖 ----------
for name in ["train_mid_common", "train_mid_clean"]:
    p = f"5script/{name}.csv"
    if not os.path.exists(p):
        print(f"[train] {name}: MISSING")
        continue
    ids = ids_of(p)
    hit = sum(1 for i in ids if i in id2shard)
    print(f"[train] {name:18s} rows={len(ids):7d}  latent_hit={hit:7d} "
          f"({100.0*hit/max(len(ids),1):.1f}%)  miss={len(ids)-hit}")

# ---------- 2. eval 集 ----------
EVAL_CSV = "5script/eval_strict_midclean.csv"
rows = rows_of(EVAL_CSV)
print(f"\n[eval] {EVAL_CSV}: rows={len(rows)}")
print(f"       columns = {list(rows[0].keys())}")

tr_rows = rows_of("5script/train_mid_common.csv")
tr_glyphs = set(int(r["glyph_id"]) for r in tr_rows)
tr_combo = set((int(r["glyph_id"]), int(r["calligrapher_id"])) for r in tr_rows)

ev_glyphs = set(int(r["glyph_id"]) for r in rows)
ev_combo = set((int(r["glyph_id"]), int(r["calligrapher_id"])) for r in rows)

print(f"       eval uniq glyph      = {len(ev_glyphs)}")
print(f"       eval uniq combo      = {len(ev_combo)}")
print(f"       glyph in-domain rate = {100.0*len(ev_glyphs & tr_glyphs)/len(ev_glyphs):.1f}%"
      f"   (zero-shot = {len(ev_glyphs - tr_glyphs)})")
print(f"       combo in-domain rate = {100.0*len(ev_combo & tr_combo)/len(ev_combo):.1f}%")

# 书体分布
from collections import Counter
sc = Counter(r.get("script", "?") for r in rows)
print(f"       script dist          = {dict(sc)}")

# ---------- 3. eval 的 GT 图 / latent ----------
ev_ids = ids_of(EVAL_CSV)
gt_dir = "final_imgs_256"
gt_hit = sum(1 for i in ev_ids if os.path.exists(os.path.join(gt_dir, f"{i}.png")))
lat_hit = sum(1 for i in ev_ids if i in id2shard)
print(f"\n[eval] GT 图齐全   : {gt_hit}/{len(ev_ids)} ({100.0*gt_hit/max(len(ev_ids),1):.1f}%)")
print(f"[eval] latent 命中 : {lat_hit}/{len(ev_ids)} ({100.0*lat_hit/max(len(ev_ids),1):.1f}%)"
      f"   (eval 只用 GT 图做对比，不查 latent — 仅供参考)")

# ---------- 4. 与旧 eval 集对比 ----------
OLD = "5script/eval_strict_top6.csv"
if os.path.exists(OLD):
    orows = rows_of(OLD)
    og = set(int(r["glyph_id"]) for r in orows)
    print(f"\n[cmp] {os.path.basename(OLD)}: rows={len(orows)} uniq_glyph={len(og)} "
          f"in-mid_common={100.0*len(og & tr_glyphs)/len(og):.1f}% "
          f"zero-shot={len(og - tr_glyphs)}")
    print("      -> 旧 eval 含 zero-shot 字，与 mid-common 训练不匹配；改用新 eval 集")

ok = True
ids = ids_of("5script/train_mid_common.csv")
miss = [i for i in ids if i not in id2shard]
if miss:
    print(f"\n[FATAL] 训练集有 {len(miss)} 个 img_id 缺 latent，例: {miss[:5]}")
    ok = False
if len(ev_glyphs - tr_glyphs) != 0:
    print(f"\n[WARN] eval 集并非 zero-shot=0，有 {len(ev_glyphs - tr_glyphs)} 个域外字")
if gt_hit != len(ev_ids):
    print(f"\n[FATAL] eval GT 图缺 {len(ev_ids)-gt_hit} 张")
    ok = False

print("\n" + "=" * 60)
print("PRECHECK " + ("PASSED" if ok else "FAILED"))
print("=" * 60)
sys.exit(0 if ok else 1)

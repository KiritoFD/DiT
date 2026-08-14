# -*- coding: utf-8 -*-
"""
生成 官方图 img_id -> 远程 latent 文件 的复用映射 + 缺失清单。
规则：
- 官方每图 (cal, sc, ch, 组合内第k个)
- cal 经变体映射 -> cal'（精确同名 或 (script,char)交集 best）
- 远程 latent: dataset/latents/cal'/sc/ch/{idx}.npy，取该目录第 k 个（idx 升序）
- 若远程该目录 latent 数 < 官方该组合图数，超出部分入缺失清单
输出:
  latent_reuse_map.json   {"img_id": "cal'/sc/ch/idx.npy"}
  latent_missing.json     [{"img_id", "orig_path", "calli_raw", "script", "char"}]
"""
import json, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

m = json.load(open("final_manifest_split.json", encoding="utf-8"))
rem_files = json.load(open("_remote_latent_files.json", encoding="utf-8"))  # "cal\tsc\tchar" -> [idx..]
key = lambda cal, sc, ch: f"{cal}\t{sc}\t{ch}"

# 远程: 组合 -> 目录是否含 latent
def rem_has(cal, sc, ch):
    return key(cal, sc, ch) in rem_files

# 官方: calli -> Counter((sc,ch)) 和 每组合内样本列表
off_calli_sc = defaultdict(list)
for r in m:
    off_calli_sc[r["orig_calli_raw"]].append((r["orig_script"], r["orig_char"]))

# 远程 calli 全集
rem_calli = set(k.split("\t")[0] for k in rem_files)

def calli_map(cal):
    if cal in rem_calli:
        return cal
    scset = set(off_calli_sc[cal])
    best, best_ov = None, 0
    for tgt in rem_calli:
        # 计算 tgt 与 cal 的 (sc,ch) 交集
        ov = 0
        # 用远程 files 的 key 前缀
        prefix = tgt + "\t"
        # 简单：遍历远程该 calli 的 (sc,ch)
        # 优化：从 rem_files 筛选
        ov = sum(1 for (sc,ch) in scset if key(tgt, sc, ch) in rem_files)
        if ov > best_ov:
            best_ov, best = ov, tgt
    return best

# 统计每个组合内官方样本的序号
# 官方按 (cal, sc, ch) 分组，记录原始 manifest 顺序
by_combo = defaultdict(list)
for r in m:
    by_combo[(r["orig_calli_raw"], r["orig_script"], r["orig_char"])].append(r)

reuse = {}       # img_id -> "cal'/sc/ch/idx"
missing = []     # 列表
n_reuse = 0
n_missing = 0
map_detail = Counter()

for (cal, sc, ch), recs in by_combo.items():
    tgt = calli_map(cal)
    rk = key(tgt, sc, ch) if tgt else None
    r_idxs = rem_files.get(rk, []) if rk else []
    # 每个官方样本取远程该目录第 k 个 idx
    for k, r in enumerate(recs):
        if k < len(r_idxs):
            idx = r_idxs[k]
            reuse[str(r["img_id"])] = f"{tgt}/{sc}/{ch}/{idx:05d}.npy"
            n_reuse += 1
        else:
            missing.append({
                "img_id": r["img_id"], "orig_path": r["orig_path"],
                "calli_raw": r["orig_calli_raw"], "calli_map": tgt,
                "script": r["orig_script"], "char": r["orig_char"],
                "k": k, "remote_available": len(r_idxs),
            })
            n_missing += 1
            map_detail[f"no_latent({cal}->{tgt})"] += 1

print(f"reuse: {n_reuse}, missing(需重encode): {n_missing}")
print("missing breakdown:")
for k, v in map_detail.most_common(10):
    print(f"  {k}: {v}")

with open("latent_reuse_map.json", "w", encoding="utf-8") as f:
    json.dump(reuse, f, ensure_ascii=False)
with open("latent_missing.json", "w", encoding="utf-8") as f:
    json.dump(missing, f, ensure_ascii=False)
print("written latent_reuse_map.json, latent_missing.json")

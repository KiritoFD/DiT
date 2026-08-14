# -*- coding: utf-8 -*-
"""生成缺失官方图清单（需重 encode）并输出 reuse 映射（复用远程 latent）。"""
import json, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("loading...", flush=True)
m = json.load(open("final_manifest_split.json", encoding="utf-8"))
rem_files = json.load(open("_remote_latent_files.json", encoding="utf-8"))
key = lambda cal, sc, ch: f"{cal}\t{sc}\t{ch}"
print("loaded", len(m), "official,", len(rem_files), "remote dirs", flush=True)

# 远程: calli -> set((script,char))   一次建好
rem_calli_sc = defaultdict(set)
for k in rem_files:
    cal, sc, ch = k.split("\t")
    rem_calli_sc[cal].add((sc, ch))
rem_calli = set(rem_calli_sc)
print("remote calli:", len(rem_calli), flush=True)

# 官方: calli -> set((script,char))
off_calli_sc = defaultdict(set)
for r in m:
    off_calli_sc[r["orig_calli_raw"]].add((r["orig_script"], r["orig_char"]))

def calli_map(cal):
    if cal in rem_calli_sc:
        return cal
    scset = off_calli_sc[cal]
    best, best_ov = None, 0
    for tgt, rsc in rem_calli_sc.items():
        ov = len(scset & rsc)
        if ov > best_ov:
            best_ov, best = ov, tgt
    return best

# 官方按组合分组
by_combo = defaultdict(list)
for r in m:
    by_combo[(r["orig_calli_raw"], r["orig_script"], r["orig_char"])].append(r)
print("official combos:", len(by_combo), flush=True)

reuse = {}
missing = []
for i, ((cal, sc, ch), recs) in enumerate(by_combo.items()):
    tgt = calli_map(cal)
    r_idxs = rem_files.get(key(tgt, sc, ch), []) if tgt else []
    for k, r in enumerate(recs):
        if k < len(r_idxs):
            reuse[str(r["img_id"])] = f"{tgt}/{sc}/{ch}/{r_idxs[k]:05d}.npy"
        else:
            missing.append(r)
    if (i + 1) % 20000 == 0:
        print(f"  progress {i+1}/{len(by_combo)}", flush=True)

with open("latent_reuse_map.json", "w", encoding="utf-8") as f:
    json.dump(reuse, f, ensure_ascii=False)
with open("latent_missing.json", "w", encoding="utf-8") as f:
    json.dump(missing, f, ensure_ascii=False)
print("reuse:", len(reuse), "missing(需重encode):", len(missing), flush=True)
mc = Counter(x["orig_calli_raw"] for x in missing)
print("missing top calli:", mc.most_common(8), flush=True)

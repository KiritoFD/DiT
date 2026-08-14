# -*- coding: utf-8 -*-
"""构建官方最终 latent 计划：每张官方图 img_id -> latent 来源。
复用部分(317301): 用 fp 查远程去重映射 _remote_dedup.json -> 远程 latent 路径。
本地部分(12414): 本地 latent_missing/{img_id}.npy（待打包上传）。
输出 final_latent_plan.json: {"img_id": {"src":"remote"/"local", "path":...}}
"""
import json, sys, glob, hashlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

# 远程 index: path -> fp
rem = json.load(open("_remote_vae_index.json", encoding="utf-8"))
rem_path_fp = {r["path"]: r["latent_fp"] for r in rem}
# 远程去重: fp -> 代表 latent 路径
dedup = json.load(open("_remote_dedup.json", encoding="utf-8"))
print("remote index entries:", len(rem), "dedup fp:", len(dedup))

# 复用映射: img_id -> "cal/sc/ch/idx.npy"
reuse = json.load(open("latent_reuse_map.json", encoding="utf-8"))
# 本地缺失清单: img_id
missing = json.load(open("latent_missing.json", encoding="utf-8"))
missing_ids = set(str(x["img_id"]) for x in missing)
print("reuse count:", len(reuse), "missing count:", len(missing_ids))

plan = {}
n_remote = n_local = 0
n_remote_nofp = 0
# 复用部分
for img_id, rel in reuse.items():
    img_path = "dataset/images/" + rel[:-4] + ".png"
    fp = rem_path_fp.get(img_path)
    if fp and fp in dedup:
        plan[img_id] = {"src": "remote", "path": dedup[fp]}
        n_remote += 1
    else:
        # fp 没在去重映射（理论上不该发生），标记
        plan[img_id] = {"src": "remote_unknown", "path": "dataset/latents/" + rel}
        n_remote_nofp += 1

# 本地部分
for img_id in missing_ids:
    plan[img_id] = {"src": "local", "path": f"latent_missing/{img_id}.npy"}
    n_local += 1

print(f"remote: {n_remote}, remote_unknown: {n_remote_nofp}, local: {n_local}, total: {len(plan)}")

with open("final_latent_plan.json", "w", encoding="utf-8") as f:
    json.dump(plan, f, ensure_ascii=False)
print("written final_latent_plan.json")

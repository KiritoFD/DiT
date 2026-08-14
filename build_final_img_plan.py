# -*- coding: utf-8 -*-
"""构建官方图片计划：每张官方图 img_id -> 图片来源。
remote: 远程去重 latent 对应图片（dataset/images/...）
local:  本地 final/{split}/{img_id}.png（需上传）
输出 final_img_plan.json: {"img_id": {"src":"remote"/"local","path":...}}
"""
import json, sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plan = json.load(open("final_latent_plan.json", encoding="utf-8"))
# 远程去重 latent 路径 -> 图片路径
img_plan = {}
n_remote = n_local = 0
for img_id, rec in plan.items():
    if rec["src"] == "remote":
        # latent: dataset/latents/... -> image: dataset/images/...
        img_path = rec["path"].replace("dataset/latents/", "dataset/images/").replace(".npy", ".png")
        img_plan[img_id] = {"src": "remote", "path": img_path}
        n_remote += 1
    else:
        img_plan[img_id] = {"src": "local", "path": None}  # 待定 split，用 manifest
        n_local += 1

# 本地部分补充 split（从 manifest）
manifest = {str(r["img_id"]): r for r in json.load(open("final_manifest_split.json", encoding="utf-8"))}
for img_id, rec in img_plan.items():
    if rec["src"] == "local":
        r = manifest.get(img_id)
        if r:
            rec["path"] = f"final/{r['final_split']}/{img_id}.png"

with open("final_img_plan.json", "w", encoding="utf-8") as f:
    json.dump(img_plan, f, ensure_ascii=False)

print("remote:", n_remote, "local:", n_local, "total:", len(img_plan))
# 校验本地路径存在
missing_local = 0
for img_id, rec in img_plan.items():
    if rec["src"] == "local" and not os.path.exists(rec["path"]):
        missing_local += 1
print("local images missing on disk:", missing_local)

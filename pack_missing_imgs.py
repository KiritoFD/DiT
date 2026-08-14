# -*- coding: utf-8 -*-
"""打包本地 12414 张官方图（local 部分）为 tar.gz，保留 final/{split}/{img_id}.png 路径。"""
import json, sys, os, tarfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plan = json.load(open("final_img_plan.json", encoding="utf-8"))
local = [(img_id, rec["path"]) for img_id, rec in plan.items() if rec["src"] == "local"]
print("local images:", len(local))

out = "final_local_imgs.tar.gz"
with tarfile.open(out, "w:gz") as tar:
    for img_id, path in local:
        if os.path.exists(path):
            tar.add(path, arcname=f"final/{img_id}.png")  # 平铺到 final/{img_id}.png
        else:
            print("MISSING", img_id, path)
sz = os.path.getsize(out)
print(f"packed {out} size {sz/1e6:.1f} MB")

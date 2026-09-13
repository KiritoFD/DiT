# -*- coding: utf-8 -*-
"""把远程复用部分图组织到 final_images/{img_id}.png。
remote: 从 dataset/images/... 复制。local 已在 final_images/。
"""
import json, sys, os, shutil
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plan = json.load(open("final_img_plan.json", encoding="utf-8"))
os.makedirs("final_images", exist_ok=True)
n_ok = n_fail = 0
fails = []
for i, (img_id, rec) in enumerate(plan.items()):
    if rec["src"] != "remote":
        continue
    dst = f"final_images/{img_id}.png"
    if os.path.exists(dst):
        n_ok += 1
        continue
    src = rec["path"]
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            fails.append((img_id, str(e)))
    else:
        n_fail += 1
        fails.append((img_id, "src missing"))
    if (i+1) % 50000 == 0:
        print(f"  progress {i+1} ok={n_ok} fail={n_fail}", flush=True)

print(f"Done remote ok={n_ok} fail={n_fail}")
if fails:
    print("sample fails:", fails[:5])
print("final_images total:", len(os.listdir("final_images")))

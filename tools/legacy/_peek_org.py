import json, sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open("/root/Workspace/xy/DiT/mccd_mapping.json"))
# image_path -> mccd filename 的映射在哪?
# 看 org_final_images.py
print("=== org_final_images.py (head) ===")
with open("/root/Workspace/xy/DiT/org_final_images.py") as f:
    for i, line in enumerate(f):
        if i >= 60: break
        print(line.rstrip())

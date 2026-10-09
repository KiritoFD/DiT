import os
import glob

print("=== 1. 查看 /home/ds/Workspace/moyi 下的所有文件 ===")
for item in sorted(os.listdir("/home/ds/Workspace/moyi")):
    p = os.path.join("/home/ds/Workspace/moyi", item)
    if os.path.isfile(p):
        print(f"  [FILE] {item}")
    else:
        print(f"  [DIR] {item}")

print("\n=== 2. 查看 moyi_top10_rf/eval_ours200fix 是如何生成的 ===")
# 搜索包含 eval_ours200fix 的脚本
for root, dirs, files in os.walk("/home/ds/Workspace/moyi"):
    if ".git" in root or "checkpoints" in root:
        continue
    for f in files:
        if f.endswith(".py") or f.endswith(".sh"):
            fp = os.path.join(root, f)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fobj:
                    if "eval_ours200fix" in fobj.read():
                        print("  found eval_ours200fix in:", fp)
            except:
                pass

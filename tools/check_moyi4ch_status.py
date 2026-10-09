import os
import glob
import json

print("=== 1. 检查 48 机器上 moyi_4ch 的完整情况 ===")
p_moyi4 = "/home/ds/Workspace/moyi/results/moyi_top10_rf_4ch"
if os.path.exists(p_moyi4):
    for root, dirs, files in os.walk(p_moyi4):
        pngs = [f for f in files if f.endswith(".png")]
        ckpts = [f for f in files if f.endswith(".pt")]
        if pngs or ckpts:
            print(f"  {root}: {len(pngs)} pngs, {len(ckpts)} ckpts")

print("\n=== 2. 查看 moyi_4ch 的 checkpoints ===")
ckpt_dir = os.path.join(p_moyi4, "checkpoints")
if os.path.exists(ckpt_dir):
    print("  ckpts:", sorted(os.listdir(ckpt_dir)))

print("\n=== 3. 检查 moyi 中是否有用于评测 4ch 在 eval200fix 上的脚本 ===")
for f in glob.glob("/home/ds/Workspace/moyi/*.py") + glob.glob("/home/ds/Workspace/moyi/scripts/*.py"):
    with open(f, "r", encoding="utf-8", errors="ignore") as pf:
        txt = pf.read()
        if "eval200" in txt:
            print("  moyi eval script:", f)

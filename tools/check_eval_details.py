import os
import json

print("=== 1. 查看 eval_auto.json ===")
p_auto = "/home/ds/Workspace/moyi/results/moyi_top10_rf/eval_ours200fix/eval_auto.json"
if os.path.exists(p_auto):
    with open(p_auto, "r", encoding="utf-8") as f:
        data = json.load(f)
        print("eval_auto.json keys:", list(data.keys()) if isinstance(data, dict) else len(data))
        print("content sample:", str(data)[:300])

print("\n=== 2. 查看 metrics_40k.json ===")
p_m40 = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug_v66route/20261006-202826-v66_cap_tier3_b_aug_route2456/evaluations/eval_60000/metrics_40k.json"
if os.path.exists(p_m40):
    with open(p_m40, "r", encoding="utf-8") as f:
        print(f.read())

print("\n=== 3. 检查 moyi_top10_rf_4ch 是否也有 eval200 或其它 eval 目录 ===")
p_4ch = "/home/ds/Workspace/moyi/results/moyi_top10_rf_4ch"
if os.path.exists(p_4ch):
    for root, dirs, files in os.walk(p_4ch):
        for d in dirs:
            if "eval" in d:
                full_d = os.path.join(root, d)
                pngs = [f for f in os.listdir(full_d) if f.endswith(".png")]
                print(f"  {full_d}: {len(pngs)} pngs")

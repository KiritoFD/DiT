import os
import glob
import json

print("=== 1. 检查 dit_b_aug_v66route 的评测集 ===")
p_dit_b = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug_v66route/20261006-202826-v66_cap_tier3_b_aug_route2456"
# 查看配置文件和评估目录
cfg_p = os.path.join(p_dit_b, "config.json")
if not os.path.exists(cfg_p):
    cfg_p = os.path.join(p_dit_b, "resolved_config.json")
if os.path.exists(cfg_p):
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("dit_b config eval_csv:", cfg.get("eval_csv"))
    print("dit_b config eval_csvs:", cfg.get("eval_csvs"))
    print("dit_b config in_mem_eval_sets:", cfg.get("in_mem_eval_sets"))

# 检查 evaluations 下的所有 png
eval_dir = os.path.join(p_dit_b, "evaluations")
if os.path.exists(eval_dir):
    for root, dirs, files in os.walk(eval_dir):
        pngs = [f for f in files if f.endswith(".png")]
        if pngs:
            print(f"  {root}: {len(pngs)} pngs -> 示例: {pngs[:5]}")

print("\n=== 2. 检查 moyi 4ch 的所有输出目录 ===")
p_moyi_4 = "/home/ds/Workspace/moyi/results/moyi_top10_rf_4ch"
if os.path.exists(p_moyi_4):
    for root, dirs, files in os.walk(p_moyi_4):
        pngs = [f for f in files if f.endswith(".png")]
        if pngs:
            print(f"  {root}: {len(pngs)} pngs -> 示例: {pngs[:5]}")

print("\n=== 3. 检查 eval_full_metrics 里的 10 张图到底是什么字 ===")
# 检查 /home/ds/Workspace/moyi/results/eval_full_metrics 对应的 csv
for f in glob.glob("/home/ds/Workspace/moyi/**/eval*.csv", recursive=True):
    print("  found eval csv in moyi:", f)

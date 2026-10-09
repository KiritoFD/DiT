import os
import glob
from PIL import Image

base = "/root/Workspace/xy/DiT"

# 阶段对应的原始评估输出目录
stage_paths = {
    "01_v10b": os.path.join(base, "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/strict50"),
    "02_v13": os.path.join(base, "assets/results/v13_wd01/20260918-210256-v13-base-50k/eval_samples_ctrl/step0125000/strict50"),
    "03_v21": os.path.join(base, "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/eval_samples_ctrl/step0155000/strict"),
    "04_v23": os.path.join(base, "_archive/20261003_twostage/assets_results/v23_splitnorm/eval_samples_ctrl/step0085000/strict"),
    "05_v66": os.path.join(base, "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix"),
    "06_v68": os.path.join(base, "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix"),
    "07_v70": os.path.join(base, "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl/step0010000/eval200fix")
}

# 验证目录存在性
for sid, sp in stage_paths.items():
    print(f"[{sid}] 目录存在: {os.path.exists(sp)} -> {sp}")

# 我们挑出核心具有代表性的几个公共字:
# 1. '冠': 文徵明·行书 (v13..v70 完美三元组同款: #93 vs #32)
# 2. '豈': 全7阶段同字 (v10 #6, v13..v23 #207, v66..v70 #4)
# 3. '出': 楷书大家 柳公权 vs 褚遂良 (v13..v23 #0, v66..v70 #130)
# 4. '典': 赵孟頫书体对比 楷书 vs 行书 (v13..v23 #41, v66..v70 #162)
# 5. '悟': 名家行书 颜真卿 vs 褚遂良 (v13..v23 #204, v66..v70 #140)
# 6. '照': 名家风韵 赵孟頫(隶) vs 苏轼(楷) (v13..v23 #11, v66..v70 #114)
# 7. '鼓': 名家书体 虞世南(楷) vs 柳公权(行) (v13..v23 #209, v66..v70 #54)
# 8. '呼': 王羲之·行书 vs 褚遂良·行书 (v13..v23 #138, v66..v70 #144)
# 9. '兩': 苏轼(行) vs 王羲之(楷) (v13..v23 #54, v66..v70 #84)
# 10. '其': 董其昌(楷) vs 颜真卿(楷) (v13..v23 #236, v66..v70 #172)

test_chars = [
    {"char": "冠", "v10": None, "v13": 93, "e200": 32, "desc": "文徵明·行 (完美同书家同书体)"},
    {"char": "豈", "v10": 6, "v13": 207, "e200": 4, "desc": "全7阶段同字 (张旭/李邕/何绍基)"},
    {"char": "出", "v10": None, "v13": 0, "e200": 130, "desc": "柳公权楷 vs 褚遂良楷"},
    {"char": "典", "v10": None, "v13": 41, "e200": 162, "desc": "赵孟頫楷 vs 赵孟頫行"},
    {"char": "悟", "v10": None, "v13": 204, "e200": 140, "desc": "颜真卿行 vs 褚遂良行"},
    {"char": "照", "v10": None, "v13": 11, "e200": 114, "desc": "赵孟頫隶 vs 苏轼楷"},
    {"char": "鼓", "v10": None, "v13": 209, "e200": 54, "desc": "虞世南楷 vs 柳公权行"},
    {"char": "呼", "v10": None, "v13": 138, "e200": 144, "desc": "王羲之行 vs 褚遂良行"},
    {"char": "兩", "v10": None, "v13": 54, "e200": 84, "desc": "苏轼行 vs 王羲之楷"},
    {"char": "其", "v10": None, "v13": 236, "e200": 172, "desc": "董其昌楷 vs 颜真卿楷"}
]

print("\n=== 检查各阶段实际 PNG 文件是否存在 ===")
for item in test_chars:
    ch = item["char"]
    print(f"\n--- 字符【{ch}】({item['desc']}) ---")
    for sid, sp in stage_paths.items():
        if "v10" in sid:
            idx = item["v10"]
        elif any(k in sid for k in ["v13", "v21", "v23"]):
            idx = item["v13"]
        else:
            idx = item["e200"]

        if idx is None:
            print(f"  [{sid}]: 无此样本")
            continue

        g_path = os.path.join(sp, f"g{idx}.png")
        gt_path = os.path.join(sp, f"gt{idx}.png")
        print(f"  [{sid}] idx={idx:03d} -> g exists: {os.path.exists(g_path)}, gt exists: {os.path.exists(gt_path)}")

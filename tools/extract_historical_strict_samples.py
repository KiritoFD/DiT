import os
import shutil
import json
from PIL import Image

base = "/root/Workspace/xy/DiT"
out_dir = os.path.join(base, "exp_milestones/historical_strict_common")
os.makedirs(out_dir, exist_ok=True)

# 阶段原始 strict 目录与步数配置
stages_config = {
    "01_v10b": {
        "name": "v10b-stdskel-cos-e",
        "step": "390k",
        "dir": os.path.join(base, "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/strict50")
    },
    "02_v13": {
        "name": "v13-base-50k-wd01",
        "step": "125k",
        "dir": os.path.join(base, "assets/results/v13_wd01/20260918-210256-v13-base-50k/eval_samples_ctrl/step0125000/strict50")
    },
    "03_v21": {
        "name": "v21-skelnet-200k",
        "step": "155k",
        "dir": os.path.join(base, "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/eval_samples_ctrl/step0155000/strict")
    },
    "04_v23": {
        "name": "v23-splitnorm",
        "step": "85k",
        "dir": os.path.join(base, "_archive/20261003_twostage/assets_results/v23_splitnorm/eval_samples_ctrl/step0085000/strict")
    },
    "05_v66": {
        "name": "v66-tables-condroute2456",
        "step": "150k",
        "dir": os.path.join(base, "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix")
    },
    "06_v68": {
        "name": "v68-aug-sp-c2ot",
        "step": "200k",
        "dir": os.path.join(base, "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix")
    },
    "07_v70": {
        "name": "v70-aug-sp-stdskel-c2ot",
        "step": "10k",
        "dir": os.path.join(base, "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl/step0010000/eval200fix")
    }
}

# 提取公共字列表及其在各严格评测集中的索引
common_targets = [
    {
        "char": "冠",
        "highlight": "完美同款 (文徵明·行书)",
        "v10": None,
        "v13": (93, "文徵明", "行"),
        "e200": (32, "文徵明", "行")
    },
    {
        "char": "豈",
        "highlight": "7阶段全覆盖 (张旭/李邕/何绍基)",
        "v10": (6, "张旭", "楷"),
        "v13": (207, "李邕", "行"),
        "e200": (4, "何绍基", "楷")
    },
    {
        "char": "出",
        "highlight": "唐楷大家对比 (柳公权 vs 褚遂良)",
        "v10": None,
        "v13": (0, "柳公权", "楷"),
        "e200": (130, "褚遂良", "楷")
    },
    {
        "char": "典",
        "highlight": "赵孟頫书体演进 (楷书 vs 行书)",
        "v10": None,
        "v13": (41, "赵孟頫", "楷"),
        "e200": (162, "赵孟頫", "行")
    },
    {
        "char": "悟",
        "highlight": "行书大家对比 (颜真卿 vs 褚遂良)",
        "v10": None,
        "v13": (204, "颜真卿", "行"),
        "e200": (140, "褚遂良", "行")
    },
    {
        "char": "照",
        "highlight": "墨韵对比 (赵孟頫隶 vs 苏轼楷)",
        "v10": None,
        "v13": (11, "赵孟頫", "隶"),
        "e200": (114, "苏轼", "楷")
    },
    {
        "char": "鼓",
        "highlight": "名家风骨 (虞世南楷 vs 柳公权行)",
        "v10": None,
        "v13": (209, "虞世南", "楷"),
        "e200": (54, "柳公权", "行")
    },
    {
        "char": "呼",
        "highlight": "晋唐行书 (王羲之行 vs 褚遂良行)",
        "v10": None,
        "v13": (138, "王羲之", "行"),
        "e200": (144, "褚遂良", "行")
    },
    {
        "char": "兩",
        "highlight": "书体结字 (苏轼行 vs 王羲之楷)",
        "v10": None,
        "v13": (54, "苏轼", "行"),
        "e200": (84, "王羲之", "楷")
    },
    {
        "char": "其",
        "highlight": "正书对比 (董其昌楷 vs 颜真卿楷)",
        "v10": None,
        "v13": (236, "董其昌", "楷"),
        "e200": (172, "颜真卿", "楷")
    }
]

manifest = []

print("=== 开始从原始 STRICT 结果目录提取历史评测图像 ===")
for t_idx, target in enumerate(common_targets):
    ch = target["char"]
    char_dir = os.path.join(out_dir, f"{t_idx:02d}_{ch}")
    os.makedirs(char_dir, exist_ok=True)
    
    char_meta = {
        "idx": t_idx,
        "char": ch,
        "highlight": target["highlight"],
        "stages": {}
    }
    print(f"\n[{t_idx+1}/{len(common_targets)}] 处理字符【{ch}】({target['highlight']}) -> {char_dir}")

    for sid, sinfo in stages_config.items():
        sdir = sinfo["dir"]
        if "v10" in sid:
            spec = target["v10"]
        elif any(k in sid for k in ["v13", "v21", "v23"]):
            spec = target["v13"]
        else:
            spec = target["e200"]

        if spec is None:
            continue

        sample_idx, callig, script = spec
        src_g = os.path.join(sdir, f"g{sample_idx}.png")
        src_gt = os.path.join(sdir, f"gt{sample_idx}.png")

        dst_g = os.path.join(char_dir, f"{sid}_g_{callig}_{script}.png")
        dst_gt = os.path.join(char_dir, f"{sid}_gt_{callig}_{script}.png")

        if os.path.exists(src_g):
            shutil.copy2(src_g, dst_g)
        if os.path.exists(src_gt):
            shutil.copy2(src_gt, dst_gt)

        char_meta["stages"][sid] = {
            "stage_name": sinfo["name"],
            "step": sinfo["step"],
            "sample_idx": sample_idx,
            "calligrapher": callig,
            "script": script,
            "has_g": os.path.exists(src_g),
            "has_gt": os.path.exists(src_gt),
            "g_file": os.path.basename(dst_g),
            "gt_file": os.path.basename(dst_gt)
        }
        print(f"  • {sid}: {callig}·{script} (idx={sample_idx}) -> g:{os.path.exists(src_g)}, gt:{os.path.exists(src_gt)}")

    with open(os.path.join(char_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(char_meta, f, ensure_ascii=False, indent=2)
    manifest.append(char_meta)

with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("\n✓ 全部公共字真实 strict eval 产物提取完毕，保存在: exp_milestones/historical_strict_common/")

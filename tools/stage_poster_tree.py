#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import json
import tarfile

STAGE_DIR = "/root/Workspace/xy/DiT/assets/poster_tree"
os.makedirs(STAGE_DIR, exist_ok=True)

# 1. Column metadata
COLUMNS = [
    {"idx": 0, "char": "畫", "callig": "苏轼", "script": "行"},
    {"idx": 1, "char": "轉", "callig": "颜真卿", "script": "楷"},
    {"idx": 2, "char": "躬", "callig": "何绍基", "script": "隶"},
    {"idx": 3, "char": "行", "callig": "苏轼", "script": "行"},
    {"idx": 4, "char": "睡", "callig": "郑板桥", "script": "行"},
    {"idx": 5, "char": "哀", "callig": "欧阳通", "script": "楷"},
    {"idx": 6, "char": "殘", "callig": "欧阳询", "script": "行"},
    {"idx": 7, "char": "僉", "callig": "李邕", "script": "楷"},
    {"idx": 8, "char": "富", "callig": "赵佶", "script": "楷"},
    {"idx": 9, "char": "道", "callig": "何绍基", "script": "隶"}
]
with open(os.path.join(STAGE_DIR, "meta_columns.json"), "w", encoding="utf-8") as f:
    json.dump(COLUMNS, f, ensure_ascii=False, indent=2)

# 2. Stage definitions
STAGES = [
    {
        "folder": "00_input_std",
        "title": "【输入条件】标准骨架",
        "sub": "无偏标准印刷体骨架 (std skel)",
        "badge": "纯几何拓扑引导",
        "badge_type": "input",
        "src_dir": "/root/Workspace/xy/DiT/assets/results/std_callig_aug/eval_samples_ctrl/seen_input_g",
        "pattern": "g{idx}.png"
    },
    {
        "folder": "01_v10b_390k",
        "title": "一阶段: v10b 初代骨架长跑",
        "sub": "去字表纯骨架 / 41位名家全集 / 余弦退火",
        "badge": "Step 390k | Seen 0.7606 | Strict 0.5680",
        "badge_type": "default",
        "src_dir": "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/g",
        "pattern": "g{idx}.png"
    },
    {
        "folder": "02_v13_155k",
        "title": "二阶段: v13 50k高保真清洗",
        "sub": "黄金50k底库 / 45位大师正交化 / 4ch流匹配",
        "badge": "Step 155k | Seen 0.7580 | Strict 0.5547",
        "badge_type": "default",
        "src_dir": "/root/Workspace/xy/DiT/assets/results/v13_base_50k/20260917-211905-v13-base-50k/eval_samples_ctrl/step0155000/g",
        "pattern": "g{idx}.png"
    },
    {
        "folder": "03_v21_skelnet_75k",
        "title": "三阶段A: v21 初代SkelNet",
        "sub": "引入显式双尺度几何网格形变 / 联合归一化",
        "badge": "Step 75k | 形变网格引入 | 目标特异度 +0.0126",
        "badge_type": "warning",
        "src_dir": "/root/Workspace/xy/DiT/archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/eval_samples_ctrl/step0075000/g",
        "pattern": "g{idx}.png"
    },
    {
        "folder": "04_v23_splitnorm_75k",
        "title": "三阶段B: v23 独立归一化",
        "sub": "修复形变通道均值漂移 / 分通道独立LN",
        "badge": "Step 75k | 接口修复 | 目标特异度 +0.0164",
        "badge_type": "warning",
        "src_dir": "/root/Workspace/xy/DiT/archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/eval_samples_ctrl/step0075000/g",
        "pattern": "g{idx}.png"
    },
    {
        "folder": "05_v66_condroute_150k",
        "title": "四阶段: v66 风格层级路由",
        "sub": "多尺度AdaLN逐层注入(Block 2,4,5,6) / 路由解耦",
        "badge": "Step 150k | 层次风格调制 | 笔画质感跃升",
        "badge_type": "success",
        "src_dir": "/root/Workspace/xy/DiT/assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/g",
        "pattern": "g{idx}.png"
    },
    {
        "folder": "06_v68_sp_aug_200k",
        "title": "五阶段: v68 3x对称增广+C2OT",
        "sub": "77.8k全量增广底库 / C2OT流匹配 / REPA对齐",
        "badge": "Step 200k | 旗舰综合基准 | 苍劲墨韵与神态形变",
        "badge_type": "accent",
        "src_dir": "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/g",
        "pattern": "g{idx}.png"
    },
    {
        "folder": "99_ground_truth",
        "title": "【真值基准】古代名家真迹",
        "sub": "历代传世碑帖与墨迹原拓真实真值",
        "badge": "Ground Truth 历史真迹基准",
        "badge_type": "gt",
        "src_dir": "/root/Workspace/xy/DiT/assets/results/v13_base_50k/20260917-211905-v13-base-50k/eval_samples_ctrl/step0155000/g",
        "pattern": "gt{idx}.png"
    }
]

print("=== 开始整理并打包阶段样本树 ===")
for st in STAGES:
    s_folder = os.path.join(STAGE_DIR, st["folder"])
    os.makedirs(s_folder, exist_ok=True)
    meta = {
        "title": st["title"],
        "sub": st["sub"],
        "badge": st["badge"],
        "badge_type": st["badge_type"]
    }
    with open(os.path.join(s_folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Copy 10 sample images
    src_dir = st["src_dir"]
    copied = 0
    for col in COLUMNS:
        idx = col["idx"]
        src_name = st["pattern"].format(idx=idx)
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(s_folder, f"{idx:02d}.png")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
    print(f"[{st['folder']}] 完成 {copied}/10 张样本整理")

# 3. Create tarball
tar_path = "/root/Workspace/xy/DiT/assets/poster_tree.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(STAGE_DIR, arcname="poster_tree")
print(f"🎉 树状目录已打包至: {tar_path} ({os.path.getsize(tar_path) / 1024:.1f} KB)")

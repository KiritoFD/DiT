#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/setup_milestones_dir.py
在 4090 上创建并汇总各阶段代表性 checkpoint 与配置文件。
"""
import os
import sys
import shutil
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = "/root/Workspace/xy/DiT"
TARGET_DIR = os.path.join(BASE_DIR, "exp_milestones")
os.makedirs(TARGET_DIR, exist_ok=True)

# 汇总各阶段代表性 Checkpoint 与 Config
MILESTONES = [
    {
        "stage_id": "01_v10b",
        "name": "v10b-stdskel-cos-e (390k)",
        "ckpt": os.path.join(BASE_DIR, "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/checkpoints/0390000.pt"),
        "config": os.path.join(BASE_DIR, "src/train/configs/v10b_stdskel_fame3_c41x_cos_e.json"),
        "resolved_cfg": os.path.join(BASE_DIR, "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/resolved_config.json"),
        "step": 390000,
        "arch": "DiT-2Cond-Sp/2",
        "cond_mode": "skel_xattn12_nochar",
        "callig_map": "assets/callig_id_map_41.json"
    },
    {
        "stage_id": "02_v13",
        "name": "v13-base-50k-wd01 (125k)",
        "ckpt": os.path.join(BASE_DIR, "assets/results/v13_wd01/20260918-210256-v13-base-50k/checkpoints/0125000.pt"),
        "config": os.path.join(BASE_DIR, "src/train/configs/v13_base_50k.json"),
        "resolved_cfg": os.path.join(BASE_DIR, "assets/results/v13_wd01/20260918-210256-v13-base-50k/resolved_config.json"),
        "step": 125000,
        "arch": "DiT-2Cond-S/2",
        "cond_mode": "skel_adaln4_nochar",
        "callig_map": "assets/callig_id_map.json"
    },
    {
        "stage_id": "03_v21",
        "name": "v21-skelnet-200k (155k)",
        "ckpt": os.path.join(BASE_DIR, "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0155000.pt"),
        "config": os.path.join(BASE_DIR, "src/train/configs/v21_skelnet_200k.json"),
        "resolved_cfg": os.path.join(BASE_DIR, "assets/results/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/resolved_config.json"),
        "step": 155000,
        "arch": "DiT-2Cond-S/2 + SkelNet",
        "cond_mode": "skelnet_joint_adaln4",
        "callig_map": "assets/callig_id_map.json"
    },
    {
        "stage_id": "04_v23",
        "name": "v23-splitnorm (85k)",
        "ckpt": os.path.join(BASE_DIR, "archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/20260927-124752-v23-splitnorm/checkpoints/0085000.pt"),
        "config": os.path.join(BASE_DIR, "src/train/configs/v23_splitnorm.json"),
        "resolved_cfg": os.path.join(BASE_DIR, "archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/20260927-124752-v23-splitnorm/resolved_config.json"),
        "step": 85000,
        "arch": "DiT-2Cond-S/2 + SkelNet (SplitLN)",
        "cond_mode": "skelnet_splitln_adaln4",
        "callig_map": "assets/callig_id_map.json"
    },
    {
        "stage_id": "05_v66",
        "name": "v66-tables-condroute2456 (150k)",
        "ckpt": os.path.join(BASE_DIR, "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/checkpoints/0150000.pt"),
        "config": os.path.join(BASE_DIR, "src/train/configs/v66_tables_condroute2456.json"),
        "resolved_cfg": os.path.join(BASE_DIR, "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/resolved_config.json"),
        "step": 150000,
        "arch": "DiT-2Cond-S/2",
        "cond_mode": "tables_condroute2456",
        "callig_map": "assets/callig_script_id_map.json"
    },
    {
        "stage_id": "06_v68",
        "name": "v68-aug-sp-c2ot (200k)",
        "ckpt": os.path.join(BASE_DIR, "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/checkpoints/0200000.pt"),
        "config": os.path.join(BASE_DIR, "src/train/configs/v68_aug_sp_c2ot.json"),
        "resolved_cfg": os.path.join(BASE_DIR, "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/resolved_config.json"),
        "step": 200000,
        "arch": "DiT-2Cond-Sp/2",
        "cond_mode": "tables_condroute2456_sp_c2ot",
        "callig_map": "assets/callig_script_id_map.json"
    },
    {
        "stage_id": "07_v70",
        "name": "v70-aug-sp-stdskel-c2ot (5k)",
        "ckpt": os.path.join(BASE_DIR, "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/checkpoints/0005000.pt"),
        "config": os.path.join(BASE_DIR, "src/train/configs/v70_aug_sp_stdskel_c2ot.json"),
        "resolved_cfg": os.path.join(BASE_DIR, "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/resolved_config.json"),
        "step": 5000,
        "arch": "DiT-2Cond-Sp/2",
        "cond_mode": "stdskel_condroute2456_sp_c2ot",
        "callig_map": "assets/callig_script_id_map.json"
    }
]

print("=== 正在建立各阶段代表性 Checkpoint 汇总目录 ===")
manifest_all = []

for m in MILESTONES:
    stage_path = os.path.join(TARGET_DIR, m["stage_id"])
    os.makedirs(stage_path, exist_ok=True)
    
    # 1. 建立 Checkpoint 软链接 (节省磁盘空间)
    ckpt_src = m["ckpt"]
    ckpt_dst = os.path.join(stage_path, "checkpoint.pt")
    if os.path.exists(ckpt_src):
        if os.path.islink(ckpt_dst) or os.path.exists(ckpt_dst):
            os.remove(ckpt_dst)
        os.symlink(ckpt_src, ckpt_dst)
        sz_mb = os.path.getsize(ckpt_src) / (1024 * 1024)
        print(f"[{m['stage_id']}] 关联 Checkpoint 成功 -> {ckpt_src} ({sz_mb:.1f} MB)")
    else:
        print(f"[{m['stage_id']}] 警告: Checkpoint 未找到 -> {ckpt_src}")
        
    # 2. 复制配置文件
    for cfg_key, cfg_src in [("config.json", m["config"]), ("resolved_config.json", m["resolved_cfg"])]:
        cfg_dst = os.path.join(stage_path, cfg_key)
        if os.path.exists(cfg_src):
            shutil.copy2(cfg_src, cfg_dst)
            
    # 3. 写入单阶段元信息
    with open(os.path.join(stage_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
        
    manifest_all.append(m)

# 写入总索引
with open(os.path.join(TARGET_DIR, "milestones_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest_all, f, ensure_ascii=False, indent=2)

print(f"\n🎉 汇总完成！Manifest 已保存至: {os.path.join(TARGET_DIR, 'milestones_manifest.json')}")

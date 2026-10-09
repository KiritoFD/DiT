import os
import json
import csv

base = "/root/Workspace/xy/DiT"
out_dir = os.path.join(base, "exp_milestones/eval200_outputs")
eval_csv = os.path.join(base, "exp-std/csv/eval200_fixed.csv")

rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))

# 1. 生成 meta_columns.json
columns_meta = []
for i in range(10):
    r = rows[i]
    columns_meta.append({
        "idx": i,
        "char": r["character"],
        "callig": r["calligrapher"],
        "script": r["script"]
    })

with open(os.path.join(out_dir, "meta_columns.json"), "w", encoding="utf-8") as f:
    json.dump(columns_meta, f, ensure_ascii=False, indent=2)

# 2. 为各阶段写入 meta.json
stage_metas = {
    "00_input_std": {
        "title": "【输入条件】标准骨架 (纯拓扑引导)",
        "sub": "256x256 印刷体宋体中轴骨架潜变量",
        "badge": "输入先验 | 零名家风格信息",
        "badge_type": "input"
    },
    "01_v10b": {
        "title": "一阶段: v10b 初代骨架长跑",
        "sub": "390k步 / 41书家 / 12层Cross-Attention / 无字表",
        "badge": "Step 390k | 强记忆过拟合 / 粗糙笔触",
        "badge_type": "default"
    },
    "02_v13": {
        "title": "二阶段: v13 50k高保真清洗",
        "sub": "125k步 / 45书家 / 4层AdaLN注入 / 流匹配",
        "badge": "Step 125k | 纯净底模 / 风格响应平缓",
        "badge_type": "default"
    },
    "03_v21": {
        "title": "三阶段A: v21 初代SkelNet形变",
        "sub": "155k步 / 联合形变网格 / Joint LayerNorm",
        "badge": "Step 155k | 显式间架调制引入",
        "badge_type": "warning"
    },
    "04_v23": {
        "title": "三阶段B: v23 独立归一化修复",
        "sub": "85k步 / Split LayerNorm / 修复形变抑制",
        "badge": "Step 85k | 书家振幅比提升4.24x",
        "badge_type": "warning"
    },
    "05_v66": {
        "title": "四阶段: v66 风格层级路由",
        "sub": "150k步 / 多尺度AdaLN注入blocks 2,4,5,6",
        "badge": "Step 150k | 字符+名家层次解耦调制",
        "badge_type": "success"
    },
    "06_v68": {
        "title": "五阶段: v68 3x对称增广+C2OT",
        "sub": "200k步 / 旗舰基准 / C2OT流匹配动力学",
        "badge": "Step 200k | 苍劲墨韵与神态形变旗舰",
        "badge_type": "accent"
    },
    "07_v70": {
        "title": "六阶段: v70 纯骨架解耦验证",
        "sub": "5k步 / 彻底去除字表 / std skel潜变量注入",
        "badge": "Step 5k (Live) | 纯骨架通用解耦演进中",
        "badge_type": "accent"
    },
    "99_ground_truth": {
        "title": "【真值基准】古代名家碑帖真迹",
        "sub": "Top10 书法名家真实碑帖原石拓片切片",
        "badge": "Ground Truth | 历史真迹黄金基准",
        "badge_type": "gt"
    }
}

for sname, smeta in stage_metas.items():
    sdir = os.path.join(out_dir, sname)
    if os.path.exists(sdir):
        with open(os.path.join(sdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(smeta, f, ensure_ascii=False, indent=2)

print("✓ meta_columns.json 与各阶段 meta.json 已全部写入！")

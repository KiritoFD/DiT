#!/usr/bin/env python
"""为多个书家各生成一份 few-shot 配置 + 一份串行执行脚本。

每个书家独立一份 config（data_csv / eval csv 不同），
但都指向同一个新增书家行 [45:46) —— 因为一次只跑一个书家，
行号复用没问题。
"""
import json
import os

os.chdir("/root/Workspace/xy/DiT")
BASE = "src/train/configs/v13_fewshot_k10.json"
d0 = json.load(open(BASE, encoding="utf-8"))

CALS = [("怀素", "草"), ("伊秉绶", "隶"), ("徐渭", "草"), ("沈周", "行")]

lines = [
    "#!/bin/bash",
    "# FEW-SHOT 新书家实验（4 个书家，覆盖 草/隶/草/行）—— 让结果更可信",
    "#",
    "# 每个书家: 先跑基线(不训, 新行=已有45个的均值) -> 再训 K=10 -> 对比 ssim",
    "# 冻结主干，只训书家表新增行 [45:46)（128 个参数）",
    "#",
    f"# 生成时间: {os.popen('date').read().strip()}",
    "set -u",
    "cd /root/Workspace/xy/DiT || exit 1",
    "export PYTHONPATH=/root/Workspace/xy/DiT",
    "export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor",
    "PY=/opt/conda/envs/cu121/bin/python",
    "LOGD=/root/Workspace/xy/DiT/logs/v13_series/v13_fewshot",
    "mkdir -p \"$LOGD\"",
    "TS=$(date +%Y%m%d-%H%M%S)",
    "CK=$(ls -t /root/Workspace/xy/DiT/assets/results/v13_base_50k/*/checkpoints/0155000.pt | head -1)",
    "echo \"[fewshot] base ckpt = $CK\"",
    "",
]

port = 29660
for cal, script in CALS:
    tag = f"fs_{cal}"
    d = dict(d0)
    d["data_csv"] = f"assets/{tag}_train.csv"
    d["in_mem_eval_sets"] = f"fewshot:assets/{tag}_eval.csv:100"
    d["num_calligraphers"] = 46
    d["global_batch_size"] = 10
    d["max_steps"] = 500
    out = f"src/train/configs/v13_{tag}.json"
    json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  {out}")

    lines += [
        f"# ── {cal}（{script}书）────────────────────────────────────────",
        f"echo \"[fewshot] ===== {cal} BASELINE (不训练) $(date) =====\"",
        f"LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT={port} \\",
        f"$PY -u src/train/train.py --config {out} \\",
        f"    --resume-full \"$CK\" --eval-only \\",
        f"    --results-dir /tmp/_fs_{cal}_base 2>&1 | tee \"$LOGD/{cal}_base_$TS.log\" \\",
        f"    | grep -E \"eval-only|set=fewshot|Traceback|Error\" | tail -5",
        "",
        f"echo \"[fewshot] ===== {cal} TRAIN K=10 $(date) =====\"",
        f"LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT={port+1} \\",
        f"$PY -u src/train/train.py --config {out} \\",
        f"    --resume-full \"$CK\" --train-only-new-callig --init-new-callig mean \\",
        f"    --results-dir assets/results/v13_{tag} 2>&1 | tee \"$LOGD/{cal}_train_$TS.log\" \\",
        f"    | grep -E \"train-only-new-callig|callig-emb|set=fewshot|Traceback|Error\" | tail -8",
        "",
    ]
    port += 2

lines += [
    "echo \"[fewshot] ===== ALL DONE $(date) =====\"",
    "echo \"[fewshot] 汇总（每个书家: baseline ssim vs 训练后 ssim）:\"",
    "grep -h \"set=fewshot\" \"$LOGD\"/*_base_$TS.log 2>/dev/null | sed \"s/^/  BASE  /\"",
    "grep -h \"set=fewshot\" \"$LOGD\"/*_train_$TS.log 2>/dev/null | sed \"s/^/  TRAIN /\"",
]

sh = "_review/launch_fewshot_multi.sh"
open(sh, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(f"\n  {sh}")
print("  流程: 每个书家先 baseline(--eval-only 不训) 再 K=10 训练")

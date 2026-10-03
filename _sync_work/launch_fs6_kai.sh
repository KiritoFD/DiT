#!/bin/bash
# fs6 楷书接力：等主波(5 行/隶主题)跑完后自动开楷书波（同协议）。
# 协议: batch 384, lr 3e-4 constant, +50k 步, 每 2500 步 eval，判读以 Diff 走平为准。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT

echo "[kai] 等主波结束 ($(date '+%H:%M'))"
while pgrep -f "train.py --config src/train/configs/v15_fs6_" > /dev/null; do
  sleep 120
done
echo "[kai] 主波已结束，patch 楷书配置 ($(date '+%H:%M'))"
/opt/conda/envs/cu121/bin/python - <<'PY'
import json, os
PATCH = {"lr": 3e-4, "lr_schedule": "constant", "warmup_steps": 100,
         "max_steps": 200000, "ckpt_every": 5000, "epoch_steps": 5000,
         "gpu_eval_every": 2500, "global_batch_size": 384}
for t in ["张即之-楷", "郑道昭-楷", "张裕钊-楷", "吴彩鸾-楷"]:
    p = f"src/train/configs/v15_fs6_{t}.json"
    if not os.path.exists(p):
        print("[skip]", p)
        continue
    d = json.load(open(p, encoding="utf-8"))
    d.update(PATCH)
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("[patch]", p)
PY

R=_sync_work/run_fs6.sh
wave() { for t in "$@"; do
           [ -f "src/train/configs/v15_fs6_${t}.json" ] || { echo "[skip] $t 无配置"; continue; }
           bash $R train "$t" row_pt 0.0003 50000 long & sleep 15; done; wait; }
wave 张即之-楷 郑道昭-楷 张裕钊-楷
wave 吴彩鸾-楷

/opt/conda/envs/cu121/bin/python _sync_work/fs6_report.py > logs/_fs6_long_report.txt 2>&1
echo "[kai] ALL DONE $(date '+%m-%d %H:%M')"

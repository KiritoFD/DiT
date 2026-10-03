#!/bin/bash
# -*- coding: utf-8 -*-
# 监控 s28 训练，收敛后自动拉起 s29 ControlNet（1px GT skel）。
# 收敛判定：s28 train.py 进程结束（early-stop 或 max_steps 触发），
# 然后选 best eval ckpt（按 eval_auto ssim 最大），填充 s29 main_ckpt，启动 s29。
# 运行方式：tmux new-session -d -s s28_to_s29 'bash _sync_work/_monitor_s28_to_s29.sh 2>&1 | tee _sync_work/s28_to_s29_watch.log'

ROOT=/root/Workspace/xy/DiT
S28_RESULTS=$ROOT/assets/results/s28_std_dino_pretrain
S29_CFG=$ROOT/src/train/configs/s29_ctrl_gt_skel_1px.json
S29_SCRIPT=$ROOT/_sync_work/_run_s29_ctrl.sh

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# ---- 1. 等 s28 进程结束（最多 30 小时）----
log "watching s28 train.py ..."
WAITED=0
while pgrep -f 'train.py --config.*s28_std_dino' > /dev/null; do
    sleep 300   # 5 min
    WAITED=$((WAITED+300))
    # 每 30 分钟报一次进度
    if [ $((WAITED % 1800)) -eq 0 ]; then
        STEP=$(grep -a -o 'step=[0-9]*' $S28_RESULTS/*/log.txt 2>/dev/null | tail -1)
        log "s28 still running, waited $((WAITED/60)) min, latest: $STEP"
    fi
done
log "s28 train.py exited."

# ---- 2. 选 s28 best ckpt ----
LATEST=$(ls -dt $S28_RESULTS/2026*/ | head -1)
log "s28 latest run: $LATEST"
BEST=""
BEST_SSIM=-1
for f in $(ls "$LATEST"checkpoints/eval_auto_*.json 2>/dev/null | sort); do
    read -r STEP SSIM <<< $(/opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(d['step'], d['ssim'])" 2>/dev/null)
    if [ -n "$SSIM" ] && /opt/conda/bin/python -c "exit(0 if $SSIM > $BEST_SSIM else 1)" 2>/dev/null; then
        BEST_SSIM=$SSIM
        BEST="$LATEST"checkpoints/$(printf '%07d' $STEP).pt
    fi
done
if [ -z "$BEST" ] || [ ! -f "$BEST" ]; then
    # 没有 eval_auto -> 用最新 ckpt
    BEST=$(ls "$LATEST"checkpoints/*.pt 2>/dev/null | grep -v '.done' | sort | tail -1)
fi
log "best ckpt: $BEST (ssim=$BEST_SSIM)"
if [ -z "$BEST" ] || [ ! -f "$BEST" ]; then
    log "ERROR: no valid s28 ckpt found. Abort."
    exit 1
fi

# ---- 3. 填充 s29 main_ckpt ----
/opt/conda/bin/python - "$BEST" <<'PY'
import json, sys
cfg = sys.argv[1]
p = "/root/Workspace/xy/DiT/src/train/configs/s29_ctrl_gt_skel_1px.json"
d = json.load(open(p, encoding="utf-8"))
d["main_ckpt"] = cfg
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated s29 main_ckpt ->", cfg)
PY

# ---- 4. 启动 s29 ControlNet 训练 ----
log "launching s29 ControlNet ..."
mkdir -p $ROOT/assets/results/s29_ctrl_gt_skel_1px
tmux kill-session -t s29_train 2>/dev/null
tmux new-session -d -s s29_train "bash $S29_SCRIPT 2>&1 | tee $ROOT/assets/results/s29_ctrl_gt_skel_1px/train_run.log"
sleep 3
log "s29 launched. tmux:"
tmux ls | grep s29

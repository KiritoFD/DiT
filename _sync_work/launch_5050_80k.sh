#!/usr/bin/env bash
# 起一个固定 50% std / 50% GT、80k 步、从零的训练 (关早停, 跑满)。
# 走 run_stdmix_cascade.sh 只是为了复用它的两道闸门(DRY 干跑算术 + 真 CLI 预检),
# 这里只有 1 个阶段。
cd /root/Workspace/xy/DiT || exit 1
export CFG=src/train/configs/v46_5050_80k.json
export STAGES="0.5"
export STAGE_STEPS=80000
export FIRST_LR=5e-5
export BATCH=384
export RUNS=exp-std/runs_5050
export LOGD=exp-std/logs5050
export INIT=
bash _sync_work/run_stdmix_cascade.sh 2>&1 | tee exp-std/logs5050/run.log

#!/usr/bin/env bash
# 纯 std 条件 + 历史最好配方(v10b: Sp/2 + xattn×12 + 预训练冻结表 + no_char) + 修正版评测
# 单阶段, 从零, 80k 步, 关早停。仅走 run_stdmix_cascade.sh 复用两道闸门。
cd /root/Workspace/xy/DiT || exit 1
export CFG=src/train/configs/v47_purestd_xattn12_top10.json
export STAGES="1.0"          # p_std=1.0 / p_gt=0 -> 纯 std (不再混 GT)
export STAGE_STEPS=80000
export FIRST_LR=5e-5
export BATCH=128             # xattn×12 激活大 (+9G), 用 v10b 原值
export RUNS=exp-std/runs_purestd
export LOGD=exp-std/logs_purestd
export INIT=
bash _sync_work/run_stdmix_cascade.sh 2>&1 | tee exp-std/logs_purestd/run.log

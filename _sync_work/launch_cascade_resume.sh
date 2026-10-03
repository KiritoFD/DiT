#!/usr/bin/env bash
# 从 stage 1 的产出 ckpt 继续跑剩余阶段 (stage 1 已在 17500 步按段内早停收工)。
# stage 1 的结果不受 runner 两个 bug 影响(它没走 resume 分支), 所以不重跑, 直接续。
cd /root/Workspace/xy/DiT || exit 1
export INIT=exp-std/runs/20261003-033702-v46-std-adaln4-top10-p0.2/checkpoints/0017500.pt
export STAGES="0.4 0.6 0.8 1.0"
export FIRST_LR=2e-5          # 续跑首段也是"接续微调", 不用从零的 5e-5
export STAGE_LR=2e-5
export STAGE_STEPS=30000
export BATCH=384
bash _sync_work/run_stdmix_cascade.sh 2>&1 | tee exp-std/logs/cascade2.log

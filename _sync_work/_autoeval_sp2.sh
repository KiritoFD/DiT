#!/bin/bash
# sp2 自动评测: 轮询批量评测器 (幂等, 只评新 ckpt), 训练结束后收尾一次退出
# 注意: 与训练共 GPU (sp2 20.6G/24.5G), 评测进程 ~2.5G, OOM 则下轮重试 (幂等)
cd /root/Workspace/xy/DiT || exit 1
RD=assets/results/v10b_stdskel_fame3_sp2
SETS="seen:assets/eval_seen_v10.csv:10 strict:assets/eval_fame3_strict_clean_v9.csv:237"
while tmux has-session -t v10bstdskel3sp2 2>/dev/null; do
    /opt/conda/bin/python tools/eval/eval_stdskel_batch.py --results-dir "$RD" \
        --sets $SETS --dit-batch 16 --vae-batch 6 >> /tmp/autoeval_sp2.log 2>&1
    sleep 300
done
/opt/conda/bin/python tools/eval/eval_stdskel_batch.py --results-dir "$RD" \
    --sets $SETS --dit-batch 16 --vae-batch 6 >> /tmp/autoeval_sp2.log 2>&1
echo AUTOEVAL_SP2_DONE

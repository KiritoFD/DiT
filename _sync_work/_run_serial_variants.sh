#!/bin/bash
# 串行跑三个 fame3 变体: deep (g编码器加深) -> inject (逐层ZeroAdaLN) -> di (组合)
# 每个跑到 SSIM 早停/80k 为止; 17500 ckpt 时自动打一发 canonical follow-IoU3 (n=8),
# 训练结束后对最后 ckpt 再打一发。跑完自动进下一个。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/bin/python
L=/root/Workspace/xy/DiT/_sync_work/_launch_v10b_stdskel_fame3_variant.sh
R=/root/Workspace/xy/DiT/assets/results

for SPEC in \
    "v10bstdskel3deep v10b_stdskel_fame3_deep v10b_stdskel_fame3_deep" \
    "v10bstdskel3inject v10b_stdskel_fame3_inject v10b_stdskel_fame3_inject" \
    "v10bstdskel3di v10b_stdskel_fame3_di v10b_stdskel_fame3_di"; do
    set -- $SPEC
    NAME=$1; CFG=$2; RES=$3
    echo "=== [$NAME] start $(date) ==="
    bash "$L" "$NAME" "$CFG" "$RES" || { echo "[$NAME] LAUNCH FAILED"; continue; }

    FLAG=/tmp/.iou3_${NAME}_17500
    LOG=/tmp/${NAME}_train.log
    rm -f "$FLAG"
    while tmux has-session -t "$NAME" 2>/dev/null; do
        STEP=$(grep -o 'step=[0-9]*' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)
        if [ -n "${STEP:-}" ] && [ "$STEP" -ge 17500 ] && [ ! -f "$FLAG" ]; then
            CK=$(ls -t $R/$RES/*/checkpoints/0017500.pt 2>/dev/null | head -1)
            if [ -n "$CK" ]; then
                echo "[$NAME] follow-IoU3 @17500 $(date +%H:%M)"
                if $PY tools/eval/skel_follow_gpu.py --model v10b --ckpt "$CK" --n 8 \
                    --tag "${NAME}_17500" >> /tmp/serial_runner.log 2>&1; then
                    touch "$FLAG"
                fi
            fi
        fi
        sleep 180
    done

    LASTCK=$(ls -t $R/$RES/*/checkpoints/[0-9]*.pt 2>/dev/null | head -1)
    if [ -n "$LASTCK" ]; then
        echo "[$NAME] follow-IoU3 @final($(basename $LASTCK)) $(date +%H:%M)"
        $PY tools/eval/skel_follow_gpu.py --model v10b --ckpt "$LASTCK" --n 8 \
            --tag "${NAME}_final" >> /tmp/serial_runner.log 2>&1 || true
    fi
    echo "=== [$NAME] done $(date) ==="
done
echo ALL_EXPERIMENTS_DONE

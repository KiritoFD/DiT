#!/bin/bash
# eval_ctl.sh — CPU eval daemon 的 tmux 托管控制器 (干净轮子).
#
# 用法:
#   eval_ctl.sh start   <results_root> [--threads 32] [--mode pretrain_g]
#                       [--eval-sets "seen=..,strict=.."] [--strict-every 10000] [--force]
#   eval_ctl.sh stop    <results_root>
#   eval_ctl.sh restart <results_root> [同上]
#   eval_ctl.sh status  <results_root>
#
# 行为:
#   - daemon 在 tmux session eval_<name> 中挂起 (ssh 断开不影响)
#   - 日志: logs/<name>/eval_daemon_<ts>.log + 稳定软链 eval_daemon.log
#   - start 前检查活 worker: 有则拒绝 (除非 --force), 防误杀进行中的评测
#   - start 时清理陈旧 lock/.part (只在确认无活 worker 后)
#   - 单实例: 同一 results_root 旧 daemon 会被先杀
set -u
D=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python

CMD=${1:-}; ROOT=${2:-}
if [ -z "$CMD" ] || [ -z "$ROOT" ]; then
    sed -n '2,16p' "$0"; exit 1
fi
shift 2
ROOT=$(realpath "$ROOT" 2>/dev/null || realpath -m "$ROOT")
NAME=$(basename "$ROOT")
SESSION="eval_$NAME"
LOGD="$D/logs/$NAME"
mkdir -p "$LOGD"

THREADS=32; MODE=pretrain_g; EVAL_SETS=""; STRICT_EVERY=10000; FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --threads) THREADS=$2; shift 2;;
        --mode) MODE=$2; shift 2;;
        --eval-sets) EVAL_SETS=$2; shift 2;;
        --strict-every) STRICT_EVERY=$2; shift 2;;
        --force) FORCE=1; shift;;
        *) echo "[eval_ctl] 未知参数: $1"; exit 1;;
    esac
done

daemon_pids() { ps ax -o pid=,args= | grep 'cpu_eval_daemon' | grep -- "--watch-root $ROOT" | grep -v grep | awk '{print $1}'; }
other_daemons() { ps ax -o pid=,args= | grep 'cpu_eval_daemon' | grep -v -- "--watch-root $ROOT" | grep -v grep; }
# worker 的 --ckpt 可能是相对路径 (5script/results/<name>/...), 用 results/<name> 匹配
worker_pids() { ps ax -o pid=,args= | grep 'cpu_eval_worker' | grep -F "results/$NAME/" | grep -v grep | awk '{print $1}'; }

case "$CMD" in
start|restart)
    WP=$(worker_pids)
    if [ -n "$WP" ] && [ "$FORCE" -eq 0 ]; then
        echo "[eval_ctl] 本实验有评测进行中 (worker PIDs: $(echo $WP | tr '\n' ' '))"
        echo "[eval_ctl] 如确认要打断, 用 --force; 否则等它跑完再 start"
        exit 2
    fi
    tmux kill-session -t "$SESSION" 2>/dev/null
    for p in $(daemon_pids); do kill "$p" 2>/dev/null; done
    if [ "$FORCE" -eq 1 ]; then
        for p in $WP; do kill "$p" 2>/dev/null; done
    fi
    sleep 2
    find "$ROOT" -name "*.cpu_eval.lock" -delete 2>/dev/null
    find "$ROOT" -name ".part_p*.json" -delete 2>/dev/null
    OD=$(other_daemons)
    [ -n "$OD" ] && echo "[eval_ctl] 提示: 另有 daemon 在跑 (不同 watch-root), 共享 CPU:" && echo "$OD" | awk '{print "    pid " $1 " " $NF}'
    TS=$(date +%Y%m%d-%H%M%S)
    LOG="$LOGD/eval_daemon_$TS.log"
    EXTRA=""
    [ -n "$EVAL_SETS" ] && EXTRA="--eval-sets $EVAL_SETS"
    tmux new-session -d -s "$SESSION" \
        "export PYTHONPATH=$D; $PY -u $D/src/eval/cpu_eval_daemon.py --mode $MODE \
--watch-root $ROOT --threads $THREADS --strict-every $STRICT_EVERY $EXTRA > $LOG 2>&1"
    ln -sf "$LOG" "$LOGD/eval_daemon.log"
    sleep 4
    echo "[eval_ctl] $CMD OK  session=$SESSION"
    echo "[eval_ctl] log=$LOG"
    tail -3 "$LOG" 2>/dev/null
    ;;
stop)
    tmux kill-session -t "$SESSION" 2>/dev/null
    for p in $(daemon_pids); do kill "$p" 2>/dev/null; done
    sleep 1
    for p in $(worker_pids); do kill "$p" 2>/dev/null; done
    echo "[eval_ctl] stopped session=$SESSION (含遗留 worker)"
    ;;
status)
    echo "== session : $(tmux has-session -t "$SESSION" 2>/dev/null && echo "UP ($SESSION)" || echo "DOWN")"
    echo "== daemon  : $(daemon_pids | tr '\n' ' ')"
    WP=$(worker_pids)
    echo "== workers : $(echo $WP | tr '\n' ' ')"
    OD=$(other_daemons)
    [ -n "$OD" ] && echo "== 其他 daemon: $(echo "$OD" | awk '{print $1}' | tr '\n' ' ')"
    echo "== 当前 step / 最近日志:"
    grep -E "step [0-9]+:" "$LOGD/eval_daemon.log" 2>/dev/null | tail -4 | sed 's/^/   /'
    echo "== worker 进度 (tails):"
    for f in $(ls -t $LOGD/cpu_eval_w_*_p0.log $LOGD/cpu_eval_w_*_p1.log 2>/dev/null | head -2); do
        echo "   -- $(basename $f): $(tail -1 $f 2>/dev/null)"
    done
    echo "== 已完成 eval: $(ls "$ROOT"/*/checkpoints/eval_auto_*.json 2>/dev/null | wc -l) 个"
    echo "== posters: $(ls "$ROOT"/*/posters/*.png 2>/dev/null | wc -l) 张"
    ;;
*)
    sed -n '2,16p' "$0"; exit 1;;
esac

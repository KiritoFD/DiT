#!/bin/bash
# fs6 —— v15 few-shot 跑批（主题数据由 _sync_work/build_fs_topic.py 生成）
#
# 用法:
#   bash _sync_work/run_fs6.sh base  <topic>              # 不训练 baseline: mean / dino 两个 init
#   bash _sync_work/run_fs6.sh train <topic> <init> <lr> <steps> [tag]
#   bash _sync_work/run_fs6.sh ref                        # v15a 主干在 seen/strict 上的参照
#
# topic 形如 沈周-行。init ∈ mean_scaled | row_pt。
#
# 为什么 baseline 也要跑两种 init:
#   v15 表里每一行的语义是"该 (书家×书体) 样本 DINO 特征的 K 个 K-Means 质心"
#   (tools/build_multistyle_k4.py)。主干学的是读质心。所以"能不能装下全新书家"
#   必须先试**零梯度直接把新行写成新书家自己的质心**(row_pt)，
#   而不是只试"所有书家的均值"(mean_scaled) 再去梯度拟合。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v15_series/v15_fs6
CK=$(ls -t assets/results/v15a_multistyle_k4/*/checkpoints/0150000.pt | head -1)
TS=$(date +%m%d-%H%M%S)
port=${PORT:-29810}
mkdir -p "$LOGD"
[ -n "$CK" ] || { echo "[fs6] 找不到 v15a 150k ckpt"; exit 1; }

run() {   # run <logname> <grep-pattern> <args...>
  local name=$1 pat=$2; shift 2
  local log="$LOGD/${name}_${TS}.log"
  echo "[fs6] >>> $name  ($(date +%H:%M:%S))"
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$port \
    $PY -u src/train/train.py "$@" > "$log" 2>&1
  local rc=$?
  grep -E "$pat" "$log" | tail -8
  echo "[fs6] <<< $name rc=$rc  log=$log"
  port=$((port + 1))
  return $rc
}

cmd=${1:-}
case "$cmd" in
  base)
    t=$2
    for init in mean_scaled row_pt; do
      run "${t}_base_${init}" "set=fewshot|Traceback|Error|✗|resume-full\] 风格表扩行|callig-emb\] 新增" \
        --config "src/train/configs/v15_fs6_${t}.json" \
        --resume-full "$CK" --eval-only --init-new-callig "$init" \
        --results-dir "/tmp/_fs6_${t}_base_${init}"
    done
    ;;
  train)
    t=$2; init=$3; lr=$4; steps=$5; tag=${6:-}
    # ★ 结果目录必须每次唯一: in_mem_eval 按 (step,set) 去重读 results_dir 下的
    #   eval_stdskel_summary.csv，复用旧目录会让评测"0s done"整轮静默跳过。
    out="assets/results/v15_fs6_${t}_${init}_lr${lr}${tag:+$tag}_$(date +%m%d-%H%M%S)"
    run "${t}_train_${init}_lr${lr}_s${steps}${tag:+_$tag}" \
        "set=fewshot|train-only-new-callig|风格表扩行|Diff|Traceback|Error|✗" \
        --config "src/train/configs/v15_fs6_${t}.json" \
        --resume-full "$CK" --train-only-new-callig --init-new-callig "$init" \
        --lr "$lr" --max-steps $((150000 + steps)) \
        --results-dir "$out"
    ;;
  ref)
    run "v15a_ref_seen_strict" "set=seen|set=strict|Traceback|Error" \
      --config src/train/configs/v15a_multistyle_k4_pool.json \
      --resume-full "$CK" --eval-only --use-ema false \
      --results-dir /tmp/_fs6_ref
    ;;
  *)
    grep -E "^#" "$0" | head -16
    ;;
esac

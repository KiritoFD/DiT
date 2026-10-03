#!/bin/bash
# 联合训练配置基准: 各跑 40 步, 取 step30->40 的干净窗口算 s/step 与 peak 显存。
# 变量只有一个: gen 是否 torch.compile。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--gen-ckpt assets/results/v33_stage1_xs/20261001-113156-v33-stage1-xs/checkpoints/0030000.pt \
 --bak-ckpt assets/results/v32_stage2_img/20261001-062933-v32-stage2-img/checkpoints/0080000.pt \
 --train-bak 0 --lam-skel 0 --batch 32 --gen-steps 8 --lr 1e-5 \
 --max-steps 40 --log-every 10 --eval-every 0 --ckpt-every 0"

bench () {
  local tag="$1"; shift
  echo "===== $tag =====  $*"
  $PY -u tools/train_joint_stage1_stage2.py $COMMON "$@" --out-dir /tmp/bench_$tag \
      > /tmp/bench_$tag.log 2>&1
  grep -E '^ +step ' /tmp/bench_$tag.log | tail -3
  grep -E 'Traceback|Error|FATAL' /tmp/bench_$tag.log | tail -3
}

bench nocompile
bench compile --compile-mode reduce-overhead
echo "===== BENCH DONE ====="

#!/bin/bash
# Inference-path probes: run each baseline's sampling script with RANDOM-INIT
# weights on the first 4 eval items, then pair against GT with eval_metrics
# (metric values are meaningless; the point is the full infer->eval pipeline).
# Run: bash adapter/run_infer_probe.sh
set -e
PY=/opt/conda/envs/baseline/bin/python
BASE=/root/Workspace/xy/DiT/baseline
LOG=$BASE/smoke_logs
mkdir -p $LOG

probe_fontdiffuser() {
  echo "==== FontDiffuser sampling probe ===="
  cd $BASE/FontDiffuser
  $PY ../adapter/fontdiffuser/sample_top10.py \
    --ckpt_dir ignored --random_init --limit 4 --batch_size 4 \
    --eval_tag strict84 --save_image_dir ../results/probe/fontdiffuser \
    2>&1 | tee $LOG/probe_fd.log | tail -2
}

probe_vqfont() {
  echo "==== VQ-Font sampling probe ===="
  cd $BASE/VQ-Font
  $PY ../adapter/vqfont/sample_top10.py ../adapter/vqfont/cfg_top10.yaml \
    ../adapter/vqfont/cfg_smoke.yaml \
    --weight ignored --random_init --limit 4 --batch_size 4 \
    --eval_tag strict84 --save_dir ../results/probe/vqfont \
    2>&1 | tee $LOG/probe_vq.log | tail -2
}

probe_dgfont() {
  echo "==== DG-Font sampling probe ===="
  cd $BASE/DG-Font
  $PY ../adapter/dgfont/sample_top10.py \
    --random_init --limit 4 --batch_size 4 \
    --eval_tag strict84 --save_dir ../results/probe/dgfont \
    2>&1 | tee $LOG/probe_dg.log | tail -2
}

probe_eval() {
  echo "==== eval_metrics sanity on probe outputs ===="
  cd $BASE
  for m in fontdiffuser vqfont dgfont; do
    echo "--- $m ---"
    $PY adapter/common/eval_metrics.py --gt data/eval/strict84/gt \
      --pred results/probe/$m --device cuda:0 2>&1 | tail -1
  done
}

case ${1:-all} in
  fontdiffuser) probe_fontdiffuser ;;
  vqfont)       probe_vqfont ;;
  dgfont)       probe_dgfont ;;
  eval)         probe_eval ;;
  all)          probe_fontdiffuser; probe_vqfont; probe_dgfont; probe_eval ;;
  *) echo "unknown $1"; exit 1 ;;
esac
echo PROBE_DONE

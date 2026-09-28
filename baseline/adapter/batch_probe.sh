#!/bin/bash
# Batch-size probe: run each baseline's REAL training entry for a few steps at
# increasing batch sizes on an IDLE GPU, record peak VRAM (nvidia-smi poller)
# and wall-clock steps/s. Prints a table + recommended batch at the end.
# Run on an otherwise-idle GPU:  bash adapter/batch_probe.sh [fd|vq|dg|all]
PY=/opt/conda/envs/baseline/bin/python
BASE=/root/Workspace/xy/DiT/baseline
OUT=$BASE/batch_probe
mkdir -p $OUT

watch() {  # watch <tag>: poll used VRAM until PROBE_DONE_<tag> file exists
  local tag=$1
  local max=0
  : > $OUT/mem_$tag.log
  while [ ! -f $OUT/done_$tag ]; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$m" ] && [ "$m" -gt "$max" ] && max=$m && echo "$max" > $OUT/mem_$tag.log
    sleep 2
  done
}

run_cfg() {  # run_cfg <tag> <cmd...>
  local tag=$1; shift
  rm -f $OUT/done_$tag
  watch $tag &
  WATCH_PID=$!
  local t0=$(date +%s)
  "$@" > $OUT/run_$tag.log 2>&1
  local rc=$?
  local t1=$(date +%s)
  touch $OUT/done_$tag; wait $WATCH_PID 2>/dev/null
  echo "$tag rc=$rc wall=$((t1-t0))s peakMem=$(cat $OUT/mem_$tag.log 2>/dev/null)MiB" >> $OUT/summary.txt
  echo "$tag rc=$rc wall=$((t1-t0))s peakMem=$(cat $OUT/mem_$tag.log 2>/dev/null)MiB"
}

probe_fd() {
  echo "== FontDiffuser @256 (steps=6) =="
  for B in 4 8 12 16 20; do
    ( cd $BASE/FontDiffuser && run_cfg fd_B$B \
      env CUDA_VISIBLE_DEVICES=0 $PY train.py --seed=123 --experience_name=probe \
      --data_root=../data/fontdiffuser --output_dir=outputs/probe_$B --report_to=tensorboard \
      --resolution=256 --style_image_size=256 --content_image_size=256 \
      --content_encoder_downsample_size=3 --channel_attn=True \
      --content_start_channel=64 --style_start_channel=64 \
      --train_batch_size=$B --max_train_steps=6 --gradient_accumulation_steps=1 \
      --log_interval=1 --learning_rate=1e-4 --lr_scheduler=linear --lr_warmup_steps=0 \
      --drop_prob=0.1 --mixed_precision=no )
    grep -oE '[0-9.]+(it/s|s/it)' $OUT/run_fd_B$B.log | tail -1 | sed "s/^/fd_B$B speed /" >> $OUT/summary.txt
  done
}

probe_vq() {
  echo "== VQ-Font @256 (iter=6, batch>=6) =="
  for B in 6 8 12 16 20; do
    cat > $OUT/cfg_vqprobe_$B.yaml <<EOF
batch_size: $B
iter: 6
print_freq: 2
n_workers: 8
val_freq: 100000
save_freq: 100000
work_dir: $BASE/VQ-Font/work_probe_B$B
EOF
    ( cd $BASE/VQ-Font && run_cfg vq_B$B \
      env CUDA_VISIBLE_DEVICES=0 $PY ../adapter/vqfont/train_top10.py probe_$B \
      ../adapter/vqfont/cfg_top10.yaml ../adapter/vqfont/cfg_smoke.yaml \
      $OUT/cfg_vqprobe_$B.yaml )
    grep -E "Step +[0-9]+:" $OUT/run_vq_B$B.log | tail -2 | sed "s/^/vq_B$B /" >> $OUT/summary.txt
  done
}

probe_dg() {
  echo "== DG-Font @256 (25 epochs x 1 iter) =="
  for B in 4 8 16 24; do
    ( cd $BASE/DG-Font && run_cfg dg_B$B \
      env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY main.py --gpu 0 --img_size 256 --data_path ../data/dgfont/train \
      --output_k 24 --val_num 8 --val_batch 8 --sty_dim 128 --batch_size $B \
      --workers 8 --epochs 25 --iters 1 --log_step 1 --model_name GAN_probe_$B )
    grep -oE '\([0-9.]+s/it\)|[0-9.]+it/s' $OUT/run_dg_B$B.log | tail -1 | sed "s/^/dg_B$B speed /" >> $OUT/summary.txt
  done
}

rm -f $OUT/summary.txt
case ${1:-all} in
  fd) probe_fd ;;
  vq) probe_vq ;;
  dg) probe_dg ;;
  all) probe_fd; probe_vq; probe_dg ;;
esac
echo ==== SUMMARY ====
cat $OUT/summary.txt
echo BATCH_PROBE_DONE

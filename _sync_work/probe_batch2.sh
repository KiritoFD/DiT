#!/bin/bash
# probe_batch2.sh —— 按**真实配置**(compile=True) 测 batch 显存/速度
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

pkill -f 'src.train.train' 2>/dev/null
sleep 6

probe () {
  local CFG=$1 B=$2
  $PY - "$CFG" "$B" <<'EOF'
import json, sys
src, b = sys.argv[1], int(sys.argv[2])
c = json.load(open(src, encoding="utf-8"))
c.update(global_batch_size=b, max_steps=250, ckpt_every=100000, epoch_steps=100000,
         in_mem_eval=False, gpu_eval_every=100000,
         experiment_name="probe", results_dir="assets/results/_probe")
json.dump(c, open("src/train/configs/_probe_tmp.json", "w", encoding="utf-8"),
          ensure_ascii=False)
EOF
  echo "--- $(basename "$CFG" .json)  batch=$B"
  timeout 320 $PY -m src.train.train --config src/train/configs/_probe_tmp.json \
      > /tmp/probe_out.log 2>&1
  grep -E 'Mem:' /tmp/probe_out.log | tail -1
  grep -oE 'OutOfMemory.{0,60}' /tmp/probe_out.log | head -1
  rm -f src/train/configs/_probe_tmp.json
}

probe src/train/configs/v31_stage1_skel.json 256
probe src/train/configs/v32_stage2_img.json  256
echo PROBE2_DONE

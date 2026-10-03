#!/bin/bash
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
pkill -f 'src.train.train' 2>/dev/null
sleep 6

# stage2 从 30k 续训, probe 的 max_steps 必须 > 30001 才会真的跑
$PY - <<'EOF'
import json
c = json.load(open("src/train/configs/v32_stage2_img.json", encoding="utf-8"))
c.update(global_batch_size=384, max_steps=30250, ckpt_every=100000,
         epoch_steps=100000, in_mem_eval=False, gpu_eval_every=100000,
         experiment_name="probe", results_dir="assets/results/_probe")
json.dump(c, open("src/train/configs/_probe_tmp.json", "w", encoding="utf-8"),
          ensure_ascii=False)
EOF
echo "--- v32_stage2_img  batch=384 (续训, 跑 250 步)"
timeout 320 $PY -m src.train.train --config src/train/configs/_probe_tmp.json \
    > /tmp/probe_v32_384.log 2>&1
grep -E 'Mem:' /tmp/probe_v32_384.log | tail -2
grep -oE 'OutOfMemory.{0,50}' /tmp/probe_v32_384.log | head -1
rm -f src/train/configs/_probe_tmp.json
echo PROBE4_DONE

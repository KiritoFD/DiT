#!/bin/bash
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
$PY - <<'EOF'
import json
for p in ["src/train/configs/v31_stage1_skel.json",
          "src/train/configs/v32_stage2_img.json"]:
    c = json.load(open(p, encoding="utf-8"))
    print("JSON OK", p, "| batch =", c["global_batch_size"],
          "| max_steps =", c["max_steps"], "| early_stop =", c["early_stop"])
EOF
[ $? -ne 0 ] && exit 1
bash run_v31_v32_two_stage.sh

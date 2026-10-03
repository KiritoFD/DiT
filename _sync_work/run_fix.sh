#!/bin/bash
# 跑 eval200 条件图修正 + 核对 shard 格式是否与 shards_std_w7 同构
cd /root/Workspace/xy/DiT || exit 1

echo "########## 原 shard 格式 ##########"
cat > /tmp/_chkfmt.py <<'PY'
import numpy as np
for p in ("exp-std/data/shards_std_w7", "exp-std/data/shards_gtskel_w7",
          "exp-std/data/shards_img"):
    import glob, os
    fs = sorted(glob.glob(p + "/*.npz"))
    if not fs:
        print(p, "-> 无 npz"); continue
    d = np.load(fs[0])
    print(f"{p}: {len(fs)} 个 shard; keys={ {k: (d[k].shape, str(d[k].dtype)) for k in d.files} }")
PY
CUDA_VISIBLE_DEVICES= /opt/conda/envs/cu121/bin/python /tmp/_chkfmt.py

echo
echo "########## 跑条件图修正 ##########"
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /opt/conda/envs/cu121/bin/python -u tools/fix_eval200_abs.py 2>&1 | tail -22

echo
echo "########## 产物 ##########"
ls -la exp-std/data/std_fixed_eval200 | head -3
echo -n "修正 png 数 = "; ls exp-std/data/std_fixed_eval200 | wc -l
ls -la exp-std/data/shards_std_w7_fixed_eval200/
echo -n "eval200_fixed.csv 行数 = "; wc -l < exp-std/csv/eval200_fixed.csv
echo "--- 复 那行 ---"; grep -n '复' exp-std/csv/eval200_fixed.csv | head -3

#!/bin/bash
# 922/80 改动 1 冒烟：真实训练路径跑 40 步。
# 关键是让 preload 便宜 -> 用极小 shard 子集 + max_preload 限制。
set -e
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
OUT=/tmp/smoke_sb
rm -rf $OUT && mkdir -p $OUT
$PY - <<'PYEOF'
import json
c = json.load(open("src/train/configs/v17_s2_s2z_baseline.json"))
c["style_ln"] = True
c["style_gain_init"] = -1.0
c["style_y_over_t_init"] = 1.0
c["style_ada_rank"] = 64       # 922/80 改动 2
c["max_steps"] = 40
c["ckpt_every"] = 40          # 必须与 epoch_steps 一致
c["epoch_steps"] = 40
c["log_every"] = 5
c["batch_size"] = 2
c["in_mem_eval_every"] = 0
c["results_dir"] = "/tmp/smoke_sb/out"
c["exp_name"] = "smoke_sb"
c["preload_latents"] = True
c["num_workers"] = 0
json.dump(c, open("/tmp/smoke_sb/cfg.json", "w"), indent=2, ensure_ascii=False)
print("cfg ok: style_ln=%s gain_init=%s max_steps=%s epoch_steps=%s ckpt_every=%s"
      % (c["style_ln"], c["style_gain_init"], c["max_steps"], c["epoch_steps"], c["ckpt_every"]))
PYEOF
CUDA_VISIBLE_DEVICES=0 $PY -m src.train.train --config /tmp/smoke_sb/cfg.json 2>&1 \
  | tee /tmp/smoke_sb/train.log | grep -Ev "FutureWarning|_register_pytree|pkg_resources|RequestsDependency|warnings.warn|^  import pkg" | tail -60

#!/bin/bash
# 摸清 runs_purestd 这轮的启动方式与有效参数, 为"原地续训"做准备
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. runs_purestd 下有什么 ##########"
ls -la exp-std/runs_purestd/ 2>/dev/null
echo "--- 最新 run 的内容 ---"
R=$(ls -dt exp-std/runs_purestd/*/ 2>/dev/null | head -1)
echo "R=$R"; ls -la "$R" | head -20
echo "--- ckpt ---"; ls -la "$R/checkpoints/" 2>/dev/null

echo
echo "########## 2. resolved_config 关键字段 ##########"
/opt/conda/envs/cu121/bin/python - "$R" <<'PY'
import json, sys, os
p = os.path.join(sys.argv[1], "resolved_config.json")
c = json.load(open(p, encoding="utf-8"))
keys = ["exp_name", "model", "max_steps", "global_batch_size", "lr", "lr_schedule",
        "warmup_steps", "ckpt_every", "ckpt_keep", "results_dir", "data_csv",
        "skel_latent_shards_dir", "eval_skel_latent_shards_dir", "eval_csv",
        "in_mem_eval_sets", "gpu_eval_every", "num_calligraphers",
        "skel_as_glyph_cond", "glyph_inject_mode", "glyph_inject_layers",
        "skel_latent_shards_dirs", "skel_latent_shards_weights",
        "w_repa", "repa_layer", "use_ema", "ema_decay", "no_char_cond",
        "callig_emb_pretrained", "freeze_callig_table", "resume_lr", "fresh_scheduler"]
for k in keys:
    if k in c:
        print(f"  {k:32s} = {c[k]}")
miss = [k for k in keys if k not in c]
print("  (未配置:", ", ".join(miss), ")")
PY

echo
echo "########## 3. 启动脚本(含 purestd/p1.0) ##########"
ls -lt _sync_work/*.sh 2>/dev/null | head -12
grep -rln "runs_purestd\|purestd" _sync_work/*.sh _sync_work/*.py 2>/dev/null | head -8

echo
echo "########## 4. 日志 ##########"
ls -lt _sync_work/*.log exp-std/logs/*.log 2>/dev/null | head -8
for f in $(ls -t _sync_work/*purestd*.log exp-std/logs/*purestd*.log 2>/dev/null | head -2); do
  echo "--- $f (头 12 行 = 启动命令) ---"; head -12 "$f"
  echo "--- 尾 6 行 ---"; tail -6 "$f"
done

echo
echo "########## 5. 是否有 best/last 标记 ##########"
find exp-std/runs_purestd -maxdepth 3 -name '*.pt' 2>/dev/null | tail -6

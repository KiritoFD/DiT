#!/usr/bin/env bash
# A/B 全量开跑前的冒烟: 每个配置只跑 40 步 (batch 32), 验证
#   [1] 模型能 build (attn_tau 是否被 config->model 通路接上, PCA .pt 能否灌进 MultiStyleEmbedder)
#   [2] forward/backward 无 shape 错误
#   [3] [style-ortho] 自检是否报告"正交基"(B 路) / 不出现(A 路)
#   [4] 40 步能正常推进 (step 日志递增)
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=${PY:-/opt/conda/envs/cu121/bin/python}
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p exp-std/logs_smoke

smoke() {
  local cfg="$1" name="$2"
  local log="exp-std/logs_smoke/${name}.log"
  echo ""
  echo "================= SMOKE $name ================="
  echo "cfg=$cfg  log=$log"
  timeout 1500 $PY -u src/train/train.py --config "$cfg" \
      --experiment-name "$name" --results-dir exp-std/runs_smoke \
      --global-batch-size 32 --skel-latent-shards-weights 1.0,0.0 \
      --max-steps 40 --lr 5e-5 > "$log" 2>&1
  local rc=$?
  echo "rc=$rc"
  echo "--- 关键行 ---"
  grep -aE 'style-ortho|style-anchor|Traceback|Error|error:|RuntimeError|ValueError|Assertion|shape|attn_tau' \
      "$log" | head -20
  echo "--- 步进确认 (最后 3 行) ---"
  grep -aE 'step=' "$log" | tail -3
  if [ "$rc" -ne 0 ]; then
    echo "!!! 失败, 日志尾部:"
    tail -25 "$log"
  fi
  return $rc
}

RC=0
smoke src/train/configs/v50_A_space_xattn_style_adaLN_top10.json smoke_A || RC=1
smoke src/train/configs/v51_B_joint_kv_k4_top10.json smoke_B || RC=1

echo ""
echo "================= 冒烟总结 ================="
if [ "$RC" -eq 0 ]; then
  echo "✓ 两路都能跑 -> 可以开全量"
else
  echo "✗ 有配置失败 -> 先修, 不要开全量 (省 5 小时)"
fi
exit $RC

#!/usr/bin/env bash
set -e
export PYTHONPATH="/home/ds/Workspace/xy/qwen-calli-20261022/vendor:/home/ds/Workspace/xy/qwen-lora-ref-20261025/vendor:/home/ds/Workspace/DiT:$PYTHONPATH"
cd /home/ds/Workspace/DiT

PYTHON="/home/ds/miniconda3/envs/pytorch/bin/python"
OUT_DIR="/home/ds/Workspace/xy/qwen_synth_1000"
mkdir -p "$OUT_DIR"

echo "=== [1/2] 启动 1,000 张 Qwen P2-A 缺失组合定向合成 ==="
$PYTHON tools/qwen_synth_1000/run_synth_pipeline.py 2>&1 | tee "$OUT_DIR/synth.log"

echo ""
echo "=== [2/2] 启动两级漏斗严格质检与独立风格判定 ==="
$PYTHON tools/qwen_synth_1000/run_funnel_filter.py 2>&1 | tee "$OUT_DIR/funnel.log"

echo ""
echo "=== 全部完成，输出摘要 ==="
cat "$OUT_DIR/funnel_summary.json"

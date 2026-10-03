#!/usr/bin/env bash
# D2 — cfg=1.0 重测三种注入方式（拆掉 CFG 结构性抵消）
#   历史"xattn 无用"的结论来自 cfg=0.7：g2=cat([g,g]) 让只由 g 驱动的通路
#   在 (cond - uncond) 里恒为 0。cfg=1.0 时该抵消消失。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
R=assets/results
OUT=assets/d2_cfg10
mkdir -p $OUT

run_one () {
  local tag="$1" rdir="$2" ckpt="$3" cfg="$4"
  [ -f "$ckpt" ] || { echo "[skip] $tag: $ckpt 不存在"; return; }
  local od="$OUT/$tag"
  mkdir -p "$od"
  echo "=================================================================="
  echo "[D2] $tag  cfg=$cfg"
  echo "=================================================================="
  $PY -m src.eval.batch_eval --results-dir "$rdir" \
      --ckpt-override "$ckpt" \
      --sets "strict:assets/eval_v13_strict.csv:249" \
      --cfg "$cfg" --steps 50 --force 2>&1 | \
      tail -20
  # batch_eval 把 csv 写在 results_dir，搬到我们的输出目录
  for f in "$rdir"/eval_stdskel_summary.csv "$rdir"/eval_stdskel_batch.csv; do
    [ -f "$f" ] && cp "$f" "$od/$(basename $f)" && rm -f "$f"
  done
}

run_one v15a_adaln_cfg10 $R/v15a_multistyle_k4 \
        $R/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt 1.0
run_one v15b_ca_cfg10    $R/v15b_multistyle_k4 \
        $R/v15b_multistyle_k4/*/checkpoints/0125000.pt 1.0
run_one v15c_xattn_cfg10 $R/v15c_fixed \
        $R/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt 1.0
echo "D2 ALL DONE"

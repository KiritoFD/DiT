#!/bin/bash
# 用修复后的口径评历史 ckpt（CPU，seen 20 + strict 249）
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
OUT=/root/Workspace/xy/DiT/assets/hist_eval_fixed.csv
echo "name,ckpt,set,ssim,med,n" > $OUT
run() {
  local name="$1" ck="$2" set="$3" csv="$4" n="$5"
  [ -f "$ck" ] || { echo "  [skip] $name: ckpt 不存在"; return; }
  local D=/tmp/_hist_${name}_${set}; rm -rf $D; mkdir -p $D
  local o=$(nice -n 10 $PY -u tools/eval/eval_stdskel_batch.py \
    --results-dir $D --ckpt-override "$ck" --device cuda \
    --sets "$set:$csv:$n" --dit-batch 64 --vae-batch 32 2>&1 | grep -E "ssim=" | tail -1)
  echo "  $name [$set] $o"
  local s=$(echo "$o" | grep -oE "ssim=[0-9.]+" | cut -d= -f2)
  local m=$(echo "$o" | grep -oE "med=[0-9.]+" | cut -d= -f2)
  [ -n "$s" ] && echo "$name,$ck,$set,$s,$m,$n" >> $OUT
}
SEEN=assets/eval_v13_seen_fixed.csv
STRICT=assets/eval_v13_strict_fixed.csv
for spec in "v13base_155k:assets/results/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt" \
            "v13wd01_125k:assets/results/v13_wd01/20260918-210256-v13-base-50k/checkpoints/0125000.pt" \
            "v15a_150k:assets/results/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt" \
            "v15c_fixed_210k:assets/results/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt"; do
  n="${spec%%:*}"; ck="${spec#*:}"
  run "$n" "$ck" seen "$SEEN" 20
  run "$n" "$ck" strict "$STRICT" 249
done
echo "=== ALL DONE ==="
cat $OUT

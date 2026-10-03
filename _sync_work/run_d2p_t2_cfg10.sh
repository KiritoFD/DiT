#!/usr/bin/env bash
# D2' - cfg=1.0 下用 **T2 风格可分性** 对照三种注入方式
#   动机: D2 已证 SSIM 三者在 cfg=1.0 下无差异(0.5636/0.5644/0.5699)
#         -> SSIM 测的是字形保真, 不是风格. 必须换 T2 这个真风格指标.
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
COLLECT=assets/ink_eval_cfg10
mkdir -p 

gen_one () {
  local tag="$1" rdir="$2" ckpt="$3" cfg="$4" step="$5"
  [ -f "$ckpt" ] || { echo "[skip] $tag: $ckpt"; return; }
  local D=/tmp/_ink10_${tag}; rm -rf $D; mkdir -p $D
  echo "=================================================================="
  echo "[D2'] $tag  cfg=$cfg"
  echo "=================================================================="
  $PY -u tools/eval/eval_stdskel_batch.py --results-dir $D       --ckpt-override "$ckpt" --device cuda       --sets "strict:assets/eval_v13_strict.csv:249"       --cfg $cfg --steps 50 --dit-batch 64 --vae-batch 32 --save-samples 2>&1 |       grep -E 'batch|strict' | tail -5
  # 找他落盘的原图目录
  local src
  src=$(find "$rdir" -path '*eval_samples_ctrl/strict*' -name 'g0.png' -newermt '-6 minutes' 2>/dev/null | head -1)
  if [ -n "$src" ]; then
    local dst=$COLLECT/${tag}__${step}__strict; mkdir -p $dst
    cp "$(dirname $src)"/g*.png "$(dirname $src)"/gt*.png $dst/ 2>/dev/null
    echo "  -> $dst  图=$(ls $dst | wc -l)"
  else
    echo "  !! 没找到落盘图"
  fi
}

R=assets/results
gen_one v15a_adaln_cfg10  $R/v15a_multistyle_k4         $R/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt 1.0 0150000
gen_one v15b_ca_cfg10     $R/v15b_multistyle_k4         $R/v15b_multistyle_k4/*/checkpoints/0125000.pt 1.0 0125000
gen_one v15c_xattn_cfg10  $R/v15c_fixed         $R/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt 1.0 0210000

echo
echo '==================== T2 (cfg=1.0) ===================='
for D in $COLLECT/*__strict; do
  [ -f "$D/g0.png" ] || { echo "[skip] $D"; continue; }
  RUN=$(basename "$D" | sed 's/__strict$//')
  echo "--- $RUN ---"
  $PY tools/probe_style_separability.py       --from-samples "$D" --eval-csv assets/eval_v13_strict.csv --group-by char       --dino-ckpt data/pretrained/pretrained_models/dinov2_vits14_pretrain.safetensors       --out "assets/t2cfg10_${RUN}.json" 2>&1 | tail -22
done
echo 'D2P ALL DONE'

#!/bin/bash
# eval_noisestart.sh — 测「噪声起步 + g 当条件」两臂的下游 (ckpt 已在, 不用训)
#
# 动机: bridge 起步下 v=0==照抄 -> 模型被往照抄按; 噪声起步没有这条捷径,
#   resAlign 反而最高 (E_inj6 0.5042 / B_w3 0.4933 vs H(w7 bridge) 0.3933)。
#   但这两臂**从来没测过下游** —— 现在补上, 与 H 的 strict 0.6427 比。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
D=data/top10_style23

run_one () {  # $1=名 $2=ckpt $3=inject
  local NM=$1 CK=$2 INJ=$3
  if [ ! -f "$CK" ]; then echo "  跳过 $NM (缺 $CK)"; return; fi
  echo "===== $NM  inject=$INJ  ckpt=$(basename "$CK") ====="
  for S in seen20 strict84; do
    $PY -u tools/gen_predskel_dit.py --set $S --resume "$CK" \
        --noise-start --inject-layers "$INJ" --beta 0.634 --tag ns 2>&1 | tail -1
    # 换名, 避免多臂互相覆盖
    mv "$D/predskel_dit_${S}_ns" "/tmp/pred_${NM}_${S}" 2>/dev/null
  done
  $PY -u tools/run_skel_calibration.py --ckpt "$V26" --alphas 0 \
      --pred-seen   "/tmp/pred_${NM}_seen20" \
      --pred-strict "/tmp/pred_${NM}_strict84" \
      --out "assets/results/_calib_${NM}" 2>&1 | grep -E 'set=.*pred ' | tail -2
}

run_one E_inj6 assets/skelnet_dit_E_inj6.pt.best 6
run_one B_w3   assets/skelnet_dit_B_w3.pt.best   2
echo NOISESTART_EVAL_DONE

#!/bin/bash
# after_arms_beta.sh — 等 auxloss 两臂跑完, 自动接 β 对照。
#
# 动机 (2026-09-30):
#   · 训练期评估 (raw 生成器输出 vs 3px GT png): ink比 1.6~1.8 = **过墨**;
#     w7 目标载体本身约 2.33x(7px/3px) -> raw 其实是"略欠"而不是漂白。
#   · 但我在 poster / ink_report 上量到的"漂白 0.43x"是在 β=0.634 **混合后**
#     的 shards 上量的。两个骨架 latent 的凸组合 -> 解码成淡影(均值 latent 的
#     非线性) -> 怀疑**变细是 β 混合造成的, 不是生成器**。
#   · β 扫描此前被 run_skel_calibration 的目录优先级 bug 毁掉(三个 β 全测了
#     同一个旧目录), 之后所有下游测试都固定 β=0.634 -> **β=1 从未测过下游**。
#   本脚本补这个关键对照: β=1(纯生成器) vs β=0.634(混合)。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
D=data/top10_style23

while ! grep -q 'AUXLOSS_ARMS_DONE' /tmp/auxloss.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] 两臂完成, 开始 β 对照"

run_one () {   # $1=名字 $2=ckpt $3=beta $4=tag $5=outdir
  local NM=$1 CK=$2 B=$3 TAG=$4 OUT=$5
  if [ ! -f "$CK" ]; then echo "  跳过 $NM (缺 $CK)"; return; fi
  echo "===== $NM   ckpt=$(basename "$CK")   beta=$B ====="
  $PY -u tools/gen_predskel_dit.py --set seen20   --resume "$CK" --beta "$B" --tag "$TAG" 2>&1 | tail -1
  $PY -u tools/gen_predskel_dit.py --set strict84 --resume "$CK" --beta "$B" --tag "$TAG" 2>&1 | tail -1
  $PY -u tools/run_skel_calibration.py --ckpt "$V26" --alphas 0 \
      --pred-seen   "$D/predskel_dit_seen20_$TAG" \
      --pred-strict "$D/predskel_dit_strict84_$TAG" \
      --out "$OUT" 2>&1 | grep -E 'set=.*pred ' | tail -2
  # ★ 判据: 直接看 poster (输入骨架 / 生成 / GT 三行)
  echo "  poster: $OUT/posters/seen_pred_poster.png $OUT/posters/strict_pred_poster.png"
  # 骨架本身的对比 (std | raw | calib | GT), 用同一批样本
  $PY -u tools/poster_skel_variants.py --n 8 --cell 150 2>&1 | tail -2
}

run_one "H  beta=1.0  (老损失, 纯生成器)" assets/skelnet_dit_H_bridge_nog_w7.pt.best 1.0   Hb1   assets/results/_calib_H_b1
run_one "I  beta=1.0  (新损失, 纯生成器)" assets/skelnet_dit_I_w7_wdm.pt.best       1.0   Ib1   assets/results/_calib_I_b1
run_one "I  beta=0.634(新损失, 混合)"     assets/skelnet_dit_I_w7_wdm.pt.best       0.634 Ib634 assets/results/_calib_I_b634
echo BETA_TEST_DONE

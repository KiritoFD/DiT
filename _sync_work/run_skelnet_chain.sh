#!/bin/bash
# run_skelnet_chain.sh — SkelNet 链式实验 (总预算 ~12h, 单卡串行)
#
# 背景 (2026-09-30 诊断):
#   · clDice 不是好判据: "照抄输入"基线就有 0.2666, 动态范围被压死 -> 改用
#     **resAlign** = cos(生成-g, GT-g), 基线 0 / 上限 ~0.51 (残差组内一致度实测)
#   · 噪声起步(A_base/B_w3): 收敛到"照抄输入", cosGen 0.8405 ≈ cosStd 0.8396
#   · bridge(g起步) **带 g 条件会泄漏**: xt=(1-t)x0+t*g 且 g 又传入 ->
#     可代数解 x0=(xt-t*g)/(1-t), flow 掉到 0.003 但 resAlign 转负(-0.60)
#   -> 本轮三类假设各跑一个 arm, 都用 resAlign 判胜负
#
# 用法: nohup bash _sync_work/run_skelnet_chain.sh > /tmp/chain.out 2>&1 &
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
ARMS=logs/skelnet_dit_arms
mkdir -p "$ARMS"

ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*"; }

COMMON="--steps 40000 --batch 128 --lr 2e-4 --eval-every 1000 --save-every 500 \
--val-n 128 --sample-steps 50 --es-patience 10"

# ══ Step 0: 已有"噪声起步"ckpt 用新判据 resAlign 重测 (拿参考点) ══
say "===== STEP 0: 重测噪声起步 ckpt (resAlign 参考点) ====="
for NAME in A_base B_w3; do
  CK=$(ls -1 assets/skelnet_dit_${NAME}.pt.step* 2>/dev/null | sort | tail -1)
  if [ -z "$CK" ]; then say "  跳过 $NAME (无 ckpt)"; continue; fi
  say "  -> $CK"
  $PY -u tools/train_skelnet_dit.py --eval-only --resume "$CK" \
      --depth 6 --hidden 256 --heads 4 --inject-layers 2 \
      --tgt-shards data/top10_style23/shards_gtskel_w3 \
      --val-n 128 --sample-steps 50 \
      2>&1 | grep -E "eval-only|指标" | tee -a "$ARMS/step0_rescore.log"
done

# ══ Step 1: bridge + 隐藏 g (堵掉代数泄漏) ══
say "===== STEP 1: G_bridge_nog (g起步, 不给网络看g, w3) ====="
$PY -u tools/train_skelnet_dit.py $COMMON \
    --depth 6 --hidden 256 --heads 4 --inject-layers 2 \
    --bridge --bridge-hide-g \
    --tgt-shards data/top10_style23/shards_gtskel_w3 \
    --log "$ARMS/G_bridge_nog.log" --out assets/skelnet_dit_G_bridge_nog.pt
say "STEP1 done"

# ══ Step 2: 噪声起步 + inject 6 (条件通路容量) ══
say "===== STEP 2: E_inj6 (噪声起步, inject6, w3) ====="
$PY -u tools/train_skelnet_dit.py $COMMON \
    --depth 6 --hidden 256 --heads 4 --inject-layers 6 \
    --tgt-shards data/top10_style23/shards_gtskel_w3 \
    --log "$ARMS/E_inj6.log" --out assets/skelnet_dit_E_inj6.pt
say "STEP2 done"

# ══ Step 3: bridge + 隐藏 g + w7 粗载体 ══
say "===== STEP 3: H_bridge_nog_w7 (g起步, 隐藏g, 目标7px) ====="
$PY -u tools/train_skelnet_dit.py $COMMON \
    --depth 6 --hidden 256 --heads 4 --inject-layers 2 \
    --bridge --bridge-hide-g \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --log "$ARMS/H_bridge_nog_w7.log" --out assets/skelnet_dit_H_bridge_nog_w7.pt
say "STEP3 done"

say "===== 链式完成 (step4/5 需下游脚本, 见 tools/gen_predskel_shards.py) ====="

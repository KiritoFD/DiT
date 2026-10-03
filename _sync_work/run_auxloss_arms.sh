#!/bin/bash
# run_auxloss_arms.sh — 新损失 (白化 + 方向 + 模长) 在 w7 bridge 上的验证
#
# 动机 (2026-09-30): 现有 w7 生成器下游 strict ssim 0.6427 (std 0.5351 基线,
#   GT oracle 0.8163), 缺口 ~38% 吃掉了; 但解码墨量只有 GT 的 0.44x -> 漂白。
#   病根 = latent MSE 把幅度拉向均值。两个补丁:
#     ① 逐通道白化 (1/std)  —— SD-VAE 通道量级差大, 结构通道被背景通道淹没
#     ② 方向项 1-cos + 模长项 (|v|-|u|)^2/|u|^2 —— 方向已是强项(resAlign 0.51),
#        单独把幅度往 1 推, 不让相消平均吃掉它
#   新日志会打 |v|/|u|: 1.0 = 幅度正确; 老臂是 ~0.78 (漂白直因)。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
ARMS=logs/skelnet_dit_arms
mkdir -p "$ARMS"

COMMON="--steps 30000 --batch 128 --lr 2e-4 --eval-every 1000 --save-every 500 \
--val-n 128 --sample-steps 50 --es-patience 8 \
--depth 6 --hidden 256 --heads 4 --inject-layers 2 \
--bridge --bridge-hide-g --tgt-shards data/top10_style23/shards_gtskel_w7"

echo "===== 冒烟 60 步 (验证新损失无形状/语法错) ====="
$PY -u tools/train_skelnet_dit.py --steps 60 --batch 8 --eval-every 0 --save-every 0 \
    --loss-whiten --dir-weight 1.0 --mag-weight 1.0 --loss-ramp 0 \
    --depth 6 --hidden 256 --heads 4 --inject-layers 2 --bridge --bridge-hide-g \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --log "$ARMS/smoke_auxloss.log" --out /tmp/skelnet_smoke_aux.pt 2>&1 \
    | grep -vE 'Warning|warn|pytree|pkg_resources|preload|skel-guard' | tail -8
if ! grep -q 'step .*flow' "$ARMS/smoke_auxloss.log"; then
  echo "★★ 冒烟失败, 终止 (不浪费 GPU)"; exit 1
fi
echo "冒烟通过, 开两臂"

echo "===== Arm I: w7 + 白化 + 方向 + 模长 ====="
$PY -u tools/train_skelnet_dit.py $COMMON \
    --loss-whiten --dir-weight 1.0 --mag-weight 1.0 --loss-ramp 1000 \
    --log "$ARMS/I_w7_wdm.log" --out assets/skelnet_dit_I_w7_wdm.pt
echo "ARM_I_DONE"

echo "===== Arm J: w7 + 方向 + 模长 (去白化, 消融) ====="
$PY -u tools/train_skelnet_dit.py $COMMON \
    --dir-weight 1.0 --mag-weight 1.0 --loss-ramp 1000 \
    --log "$ARMS/J_w7_dm.log" --out assets/skelnet_dit_J_w7_dm.pt
echo "ARM_J_DONE"

echo "AUXLOSS_ARMS_DONE"

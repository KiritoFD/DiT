#!/usr/bin/env bash
# run_skelnet_standalone_v2.sh — 用「正确的监督」重训 standalone SkelNet
#
# 继承 v29_skelnet_v2 的全部好设计 + 加上 standalone 独有的正确监督:
#   【v29 继承】
#     residual=0            废黜自由加墨残差(阻断过拟合记忆入口)
#     preserve_amp=1        墨迹保幅校准 —— 根除双线性采样导致的细线淡化与断裂
#     topo_mode=1           稀疏拓扑增删(剪刀/胶水) + L1 稀疏
#     deform_prob=0.5       书法域内随机仿射+低频弹性扰动(替代高斯白噪)
#     w_tv/w_fold/w_tv_out/w_tv_stroke/w_prune_l1/w_lig_l1   形变正则
#   【standalone 独有 = 正确的监督】
#     w_img>0               图像域监督 —— latent MSE 控制不住解码后的连通性
#                           (实测 latent cos 0.947 但墨量只剩 40%、连通分量 1.7->28.4)
#     w_contr>0             监督对比 —— 防"字->平均形变"躺平
#     直接 std->GT 监督, batch 4096, 不经扩散模型
#
# 用法:  bash run_skelnet_standalone_v2.sh            # 正式
#        STEPS=200 bash run_skelnet_standalone_v2.sh   # 探针
cd /root/Workspace/xy/DiT
mkdir -p logs

STEPS=${STEPS:-8000}
BATCH=${BATCH:-4096}
WIDTH=${WIDTH:-128}
IMGB=${IMGB:-64}
OUT=${OUT:-assets/deform_skel_top10_v2.pt}
LOG=logs/skelnet_standalone_v2.log

echo "STEPS=$STEPS BATCH=$BATCH WIDTH=$WIDTH IMG_BATCH=$IMGB OUT=$OUT"

PYTHONPATH=. /opt/conda/envs/cu121/bin/python tools/train_deform_standalone.py \
  --csv assets/train_top10_style23.csv \
  --std-dir data/top10_style23/shards_std \
  --gt-dir  data/top10_style23/shards_aux_skel3 \
  --style-emb assets/callig_script_emb_top10.pt \
  --width $WIDTH --ckpt 1 \
  --residual 0 --res-cap 1.0 \
  --stroke-mod 1 --stroke-cap 1.0 --gate-radius 0.25 \
  --topo-mode 1 \
  --preserve-amp 1 \
  --dt-ch 1 \
  --deform-prob 0.5 --deform-scale 1.0 \
  --batch $BATCH --group 32 --lr 1e-3 --steps $STEPS \
  --lr-min-ratio 0.1 \
  --w-img 0.0 --img-batch $IMGB \
  --w-mass 1.0 \
  --w-contr 0.5 --contr-mode cos --contr-tau 0.07 \
  --w-tv 1e-2 --w-tv-out 1e-2 --w-tv-res 0.0 --w-tv-stroke 1e-2 \
  --w-fold 1e-1 --w-prune-l1 0.03 --w-lig-l1 0.03 \
  --eval-every 1000 --save-every 1000 \
  --diag-decode 1 \
  --out $OUT 2>&1 | tee $LOG

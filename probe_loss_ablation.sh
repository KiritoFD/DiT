#!/usr/bin/env bash
# probe_loss_ablation.sh — 对照实验: L2 vs L1 vs 超强墨量惩罚, 看谁能防住"塌到空白"
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
COMMON="--csv assets/train_top10_style23.csv \
  --std-dir data/top10_style23/shards_std --gt-dir data/top10_style23/shards_aux_skel3 \
  --style-emb assets/callig_script_emb_top10.pt \
  --width 128 --ckpt 1 --residual 0 --res-cap 1.0 \
  --stroke-mod 1 --stroke-cap 1.0 --gate-radius 0.25 --topo-mode 1 --preserve-amp 1 \
  --dt-ch 1 --deform-prob 0.5 --deform-scale 1.0 \
  --batch 2048 --group 32 --lr 1e-3 --steps 300 --lr-min-ratio 0.1 \
  --w-img 0.0 --w-contr 0.5 --contr-mode cos --contr-tau 0.07 \
  --w-tv 1e-2 --w-tv-out 1e-2 --w-tv-res 0.0 --w-tv-stroke 1e-2 \
  --w-fold 1e-1 --w-prune-l1 0.03 --w-lig-l1 0.03 \
  --eval-every 300 --save-every 0 --diag-decode 1"

run() {  # run <tag> <extra args>
  echo ""
  echo "############ $1 ############"
  PYTHONPATH=. $PY tools/train_deform_standalone.py $COMMON $2 \
      --out /tmp/skel_ab_$1.pt 2>&1 \
    | grep -E 'step0|final|图空间|闭合|style-follow' | tail -6
}

run "A_l2_mass1"   "--latent-loss l2 --w-mass 1.0"
run "B_l1_mass1"   "--latent-loss l1 --w-mass 1.0"
run "C_l2_mass20"  "--latent-loss l2 --w-mass 20.0"
echo ""
echo "ABLATION_DONE"

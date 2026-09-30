#!/bin/bash
# run_v30_clean_chain.sh — v30 无泄露三步重训链 (单卡串行, 全自动).
#   步1: w7 生成器重训 (clean84, bridge+hide-g, es by val dice64)
#   步2: v26 主干重训  (clean84, GT-skel 条件+噪声增广, 60k)
#   步3: v30-union 联训 (干净双 ckpt, 全解冻, 端到端 diff loss)
# 全部配置/数据/脚本已入 git (d5302a5), seed 固定, 可复现.
set -u
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
LOG=logs/v30_clean_chain.log
TS=$(date +%Y%m%d-%H%M%S)

echo "=== [chain] $TS 三步重训开始 ===" >> $LOG

# ── 步1: w7 生成器 (clean84) ──
echo "--- [1/3] w7 generator (clean84) $(date '+%F %T') ---" >> $LOG
$PY -u tools/train_skelnet_dit.py \
    --csv assets/train_top10_style23_minusval_clean84.csv \
    --val-csv assets/val_skelnet_clean84.csv \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --cond-shards data/top10_style23/shards_std \
    --bridge --bridge-hide-g \
    --out assets/skelnet_dit_H_bridge_nog_w7_clean84.pt \
    --log logs/v30_w7_clean.log \
    >> $LOG 2>&1
RC1=$?
echo "[1] rc=$RC1 $(date '+%F %T')" >> $LOG
[ $RC1 -ne 0 ] && { tail -20 $LOG; exit 1; }
GEN_CKPT=assets/skelnet_dit_H_bridge_nog_w7_clean84.pt.best
[ -f "$GEN_CKPT" ] || GEN_CKPT=assets/skelnet_dit_H_bridge_nog_w7_clean84.pt

# ── 步2: v26 主干 (clean84) ──
echo "--- [2/3] v26 backbone (clean84) $(date '+%F %T') ---" >> $LOG
$PY -u src/train/train.py --config src/train/configs/v26_gtskel_clean84.json \
    >> $LOG 2>&1
RC2=$?
echo "[2] rc=$RC2 $(date '+%F %T')" >> $LOG
[ $RC2 -ne 0 ] && { tail -20 $LOG; exit 1; }
B_CKPT=$($PY -c "
import json, glob, os
best, bf = -1, ''
for d in glob.glob('assets/results/v26_gtskel_clean84/*/checkpoints'):
    for f in glob.glob(os.path.join(d, 'eval_auto_*.json')):
        try:
            dd = json.load(open(f)); s = dd.get('ssim', -1)
            if s > best:
                st = os.path.basename(f).replace('eval_auto_','').replace('.json','')
                cand = os.path.join(d, f'{int(st):07d}.pt')
                if os.path.exists(cand): best, bf = s, cand
        except Exception: pass
if not bf:
    cks = sorted(glob.glob('assets/results/v26_gtskel_clean84/*/checkpoints/0*.pt'))
    bf = cks[-1] if cks else ''
print(bf)
")

# ── 步3: v30-union 联训 (干净双 ckpt) ──
echo "--- [3/3] v30-union joint (clean ckpts) $(date '+%F %T') ---" >> $LOG
$PY -u tools/train_joint_g2img.py --batch 160 \
    --gen-ckpt "$GEN_CKPT" \
    --bak-ckpt "$B_CKPT" \
    --bak-config assets/results/v26_gtskel_clean84/$(ls -t assets/results/v26_gtskel_clean84/ | head -1)/resolved_config.json \
    --csv assets/train_top10_style23_minusval_clean84.csv \
    --shards-img data/top10_style23/shards_img \
    --shards-std data/top10_style23/shards_std \
    --strict-csv assets/eval_v13_strict84_aligned.csv \
    --results-dir assets/results/v30_union_clean \
    --experiment-name v30-union-clean \
    >> $LOG 2>&1
echo "[3] rc=$? $(date '+%F %T')" >> $LOG
echo "=== [chain] 完成 ===" >> $LOG

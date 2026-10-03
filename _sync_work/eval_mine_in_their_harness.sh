#!/bin/bash
# 决定性对比: 用**他们的 harness** 评**我的 pred 目录**。
# 若这里出 ~0.64 -> 两个 pred 目录真的不同, 我的路径正确;
# 若这里也出 0.5399 -> 错的是我之前的评路径, 我的 0.6427 不可信。
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python

echo "=== 我的 pred 目录 ==="
ls -d data/top10_style23/predskel_dit_* 2>/dev/null

echo "=== [A] 他们的 harness + 我的 w7raw pred (--no-pred) ==="
$PY -u tools/eval_union_ckpt.py --no-pred \
    --pred-shards-out data/top10_style23/predskel_dit_strict84_w7raw \
    --out-dir assets/results/v30_mine_in_their_harness --tag MINE \
    2>&1 | grep -E 'in-mem-eval.*(strict84|seen)|set=.*pred|RuntimeError' | head -8

echo "=== [B] 两套 pred 目录的 latent 统计 ==="
$PY _sync_work/cmp_pred_shards.py 2>&1 | tail -12
echo "MINE_EVAL_DONE"

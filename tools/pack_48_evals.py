import os
import subprocess

print("=== 准备在 48 机器上打包 eval 成果 ===")
remote_script = """
mkdir -p /tmp/server_48_eval_pack
cd /tmp/server_48_eval_pack

# 1. 复制 moyi eval_full_metrics
cp -r /home/ds/Workspace/moyi/results/eval_full_metrics ./moyi_eval_full_metrics

# 2. 复制 moyi_top10_rf eval_ours200fix (包含全部 g0..g186)
cp -r /home/ds/Workspace/moyi/results/moyi_top10_rf/eval_ours200fix ./moyi_12ch_eval200fix

# 3. 复制 tier3_b_aug_v66route eval_60000
cp -r /home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug_v66route/20261006-202826-v66_cap_tier3_b_aug_route2456/evaluations/eval_60000 ./dit_b_aug_v66route_eval

# 4. 打包为 tar.gz
tar -czf /home/ds/server_48_evals.tar.gz .
ls -lh /home/ds/server_48_evals.tar.gz
"""

print("执行远程命令...")

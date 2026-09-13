#!/bin/bash
# 诊断 watchdog 的 QUEUE 解析 + start_training 传参
QUEUE=(
  "C|exp_xl_head_r8|DiT-3Cond-XL/2|1|8|all|data/pretrained/DiT-XL-2-256x256.pt|test_eval_C"
)

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r name cfg model use_lora lora_r lora_target pretrained out <<< "$entry"
  echo "name=[$name]"
  echo "cfg=[$cfg]"
  echo "model=[$model]"
  echo "use_lora=[$use_lora]"
  echo "lora_r=[$lora_r]"
  echo "lora_target=[$lora_target]"
  echo "pretrained=[$pretrained]"
  echo "out=[$out]"
done

echo "--- 测试 tmux 启动传参 ---"
log="testcfg2.log"
tmux kill-session -t testexp 2>/dev/null
rm -f "$log"
# 模拟 watchdog 的 start_training：用变量传参
tmux new-session -d -s testexp "bash /root/Workspace/xy/DiT/_launch_exp.sh $cfg $log"
sleep 8
echo "=== 结果 $log ==="
head -3 "$log"
echo "=== tmux 会话进程 cmdline ==="
tmux list-sessions

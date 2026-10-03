#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== fame_ctrl_watchdog.json ==="
cat assets/fame_ctrl_watchdog.json 2>/dev/null
echo ""
echo "=== ctrl_fame_1pix_v1 成功 run 的 source_manifest ==="
cat assets/results/ctrl_fame_1pix_v1/20260830-205652-fame-ctrl-skel-1px-v1/source_manifest.json 2>/dev/null | head -30
echo ""
echo "=== 该 run 的 checkpoint ==="
ls assets/results/ctrl_fame_1pix_v1/20260830-205652-fame-ctrl-skel-1px-v1/checkpoints/ 2>/dev/null | tail -10
echo ""
echo "=== 该 run 是否有 log / resolved config ==="
ls assets/results/ctrl_fame_1pix_v1/20260830-205652-fame-ctrl-skel-1px-v1/ 2>/dev/null

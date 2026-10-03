#!/usr/bin/env bash
# 盘点实况: 训练为何停 / v48 v49 配置内容 / k4_top10 资产 / 现有 run 的 ckpt
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python

echo "########## [1] 续训日志尾 (为何停) ##########"
tail -30 exp-std/logs_purestd/stage1_p1.0_20261003-182216.log 2>/dev/null
echo
echo "########## [2] 失败标记 ##########"
ls -la exp-std/logs_purestd/CASCADE_FAILED.txt exp-std/logs/CASCADE_FAILED.txt 2>/dev/null || echo "(无失败标记)"
echo
echo "########## [3] v48 / v49 配置 (远端新增) ##########"
for f in src/train/configs/v48_inject_D_k4_styleca.json src/train/configs/v49_branch_D_k4_styleca.json; do
  echo "----- $f -----"
  [ -f "$f" ] && cat "$f" || echo "(不存在)"
  echo
done
echo "########## [4] k4_top10 资产结构 ##########"
$PY - <<'EOF'
import torch, os
p = "assets/multistyle_k4_top10.pt"
print("exists:", os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else "")
d = torch.load(p, map_location="cpu")
print("type:", type(d))
if isinstance(d, dict):
    for k, v in d.items():
        print("  key:", k, getattr(v, "shape", v) if not torch.is_tensor(v) else tuple(v.shape))
else:
    print("shape:", tuple(d.shape))
EOF
echo
echo "########## [5] 现有 run 与最新 ckpt ##########"
echo "--- runs_purestd ---"
ls -t exp-std/runs_purestd/ | head -6
for d in $(ls -dt exp-std/runs_purestd/*/ | head -3); do
  echo "  $d -> ckpt: $(ls $d/checkpoints/*.pt 2>/dev/null | tail -2 | tr '\n' ' ')"
done
echo
echo "########## [6] git 未跟踪/新增 ##########"
git status --porcelain 2>/dev/null | head -30
echo
echo "########## [7] build_k4_top10.sh 内容 ##########"
cat _sync_work/build_k4_top10.sh 2>/dev/null | head -40
echo
echo "########## [8] 其它 inject/branch 配置 ##########"
ls -la src/train/configs/ | grep -iE 'v4[89]|inject|branch|k4' 

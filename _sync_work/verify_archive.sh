#!/usr/bin/env bash
set -u
cd /root/Workspace/xy/DiT || exit 1
ARC=_archive/20261003_twostage

echo "=== [1] 归档目录真实内容 ==="
du -sh --apparent-size "$ARC" 2>/dev/null
echo "  文件数: $(find "$ARC" -type f | wc -l)   目录数: $(find "$ARC" -type d | wc -l)"
echo "  符号链接 (du 大数字的嫌疑):"
find "$ARC" -type l -exec ls -la {} \; 2>/dev/null | head -10 || echo "    无"
echo "  --- 各项实际大小 ---"
du -sh --apparent-size "$ARC"/* 2>/dev/null | sort -rh | head -8

echo
echo "=== [2] 工作区大小 (应与归档前一致 1.3T) ==="
du -sh --apparent-size . 2>/dev/null | tail -1

echo
echo "=== [3] git 现状 (归档后) ==="
echo "  未跟踪: $(git status --porcelain 2>/dev/null | grep -c '^??')"
echo "  修改:   $(git status --porcelain 2>/dev/null | grep -c '^ M')"
echo "  .gitignore 生效检查: 下面这些**不应**再出现在未跟踪里"
for f in src/utils/std_glyph_latent_v2 exp-std/runs_AB _archive; do
  git status --porcelain --ignored=no -- "$f" >/dev/null 2>&1
  n=$(git status --porcelain -- "$f" 2>/dev/null | wc -l)
  printf '    %-34s git 可见条目: %s %s\n' "$f" "$n" "$([ "$n" -eq 0 ] && echo '✓ 已被忽略' || echo '⚠ 仍可见')"
done

echo
echo "=== [4] 训练是否安然无恙 ==="
pgrep -af 'launch_AB_100k|train.py' | head -3 | cut -c1-140
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
A=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
grep -a 'Steps/Sec' "$A" 2>/dev/null | tail -2 | sed 's/\x1b\[[0-9;]*m//g'
date

echo
echo "=== [5] 关键模块仍可 import (防归档误伤) ==="
PYTHONPATH=. HF_HUB_OFFLINE=1 /opt/conda/envs/cu121/bin/python -c "
import importlib
for m in ['src.model.dit','src.model','src.eval.in_mem_eval','src.utils.latent_dataset','src.train.ckpt']:
    try:
        importlib.import_module(m); print('  ✓', m)
    except Exception as e:
        print('  ✗', m, type(e).__name__, str(e)[:80])
" 2>&1 | grep -E '✓|✗'

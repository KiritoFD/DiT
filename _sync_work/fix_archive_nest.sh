#!/usr/bin/env bash
# 修归档命名冲突造成的嵌套 + 输出待提交复核清单
set -u
cd /root/Workspace/xy/DiT || exit 1
ARC=_archive/20261003_twostage

echo "=== [1] 嵌套检查 (mv 到已存在同名目录会变成 目录/目录) ==="
for n in src_eval_legacy src_train_legacy tmp_revert assets_results; do
  if [ -d "$ARC/$n" ]; then
    sub=$(find "$ARC/$n" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -3)
    echo "  $n 的子目录: $(echo "$sub" | tr '\n' ' ')"
  fi
done

echo
echo "=== [2] 若 src_eval_legacy 里有嵌套的 legacy, 提出来 ==="
if [ -d "$ARC/src_eval_legacy/legacy" ]; then
  mv "$ARC/src_eval_legacy/legacy" "$ARC/src_eval_legacy__mine"
  # 修正 MANIFEST, 让 RESTORE.sh 精确还原
  sed -i 's#^src/eval/legacy\t#src/eval/legacy\t#' "$ARC/MANIFEST.tsv" 2>/dev/null || true
  echo "  ✓ 嵌套已提出为 src_eval_legacy__mine (MANIFEST 路径不变, RESTORE 仍按 src/eval/legacy 还原)"
else
  echo "  (无嵌套)"
fi

echo
echo "=== [3] 归档目录最终形态 ==="
echo "  我的条目 (mtime 23:1x):"
ls -la "$ARC" | awk '$6=="Oct" && $7=="3" && $8 ~ /^23:/ {print "    "$9}'
echo "  占位总量: $(du -sh --apparent-size "$ARC" 2>/dev/null | cut -f1) (绝大部分是之前已归档的两阶段 run 产物)"
echo "  MANIFEST 条目: $(wc -l < "$ARC/MANIFEST.tsv")"

echo
echo "=== [4] 源位置确认已清空 ==="
for p in src/train/legacy src/eval/legacy src/train/configs.bak_922 tmp_revert \
         src/train/train_controlnet.py src/train/train_repa.py; do
  printf '    %-38s %s\n' "$p" "$([ -e "$p" ] && echo '✗ 仍在!' || echo '✓ 已归档')"
done

echo
echo "=== [5] 训练仍健康? ==="
A=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
grep -a 'Steps/Sec' "$A" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g'
pgrep -c -f 'src/train/train.py' | sed 's/^/  train.py 进程数: /'

echo
echo "=== [6] 待你点头的提交复核清单 (只列将要入库的源码/配置) ==="
echo "  .gitignore:              $([ -f .gitignore ] && echo '新建 ✓')"
echo "  已跟踪文件被修改(20):    $(git status --porcelain | grep -c '^ M')"
echo "  已跟踪文件被删除(10):    $(git status --porcelain | grep -c '^ D')"
echo "  候选新增源码 (git 会看到且未被忽略的 src/*.py):"
git status --porcelain -- 'src/**/*.py' 2>/dev/null | head -12 | sed 's/^/    /'
echo "  ⚠ 请先跑: git status --porcelain | wc -l 与 git status --porcelain | head -40 人工复核"
echo "  ⚠ 确认无 *.pt/*.npz/大件混入后再 commit"

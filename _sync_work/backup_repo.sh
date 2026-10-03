#!/usr/bin/env bash
# P2-2: 代码/配置/脚本/清单 定期外备 (git 有 1951 个未跟踪文件, 风险最高)
#
# 只备"文本资产 + 清单", 不含大件 (shards / dino_cache / ckpt / 采样图) ——
# 那些可以从清单重建; 真正丢了找不回来的是这些手写代码与实验配置。
# 保留最近 KEEP 份, 旧的自动删。
set -u
ROOT=/root/Workspace/xy/DiT
cd "$ROOT" || exit 1
KEEP=${KEEP:-10}
DEST=${DEST:-/root/_repo_backups}
mkdir -p "$DEST"

TS=$(date +%Y%m%d-%H%M%S)
OUT="$DEST/DiT_code_${TS}.tar.gz"

INCLUDE=(
  src _sync_work tools docs scripts
  exp-std/csv exp-std/logs_AB exp-std/logs_smoke
  src/train/configs
)
EXCLUDE=(
  '--exclude=*.pt' '--exclude=*.pth' '--exclude=*.ckpt'
  '--exclude=*.npz' '--exclude=*.npy' '--exclude=*.png' '--exclude=*.jpg'
  '--exclude=__pycache__' '--exclude=*.pyc' '--exclude=.git'
  '--exclude=*_AB_patch/backup_*'
)

EXIST=()
for d in "${INCLUDE[@]}"; do [ -e "$d" ] && EXIST+=("$d"); done

tar "${EXCLUDE[@]}" -czf "$OUT" "${EXIST[@]}" 2>/dev/null
RC=$?
if [ "$RC" -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[FATAL] 备份失败 (rc=$RC)"
  exit "$RC"
fi

echo "[backup] $OUT"
echo "  大小: $(du -h "$OUT" | cut -f1)"
echo "  内含文件数: $(tar -tzf "$OUT" 2>/dev/null | wc -l)"
echo "  覆盖: ${EXIST[*]}"

# 旧备份清理
n=$(ls -1 "$DEST"/DiT_code_*.tar.gz 2>/dev/null | wc -l)
if [ "$n" -gt "$KEEP" ]; then
  ls -1t "$DEST"/DiT_code_*.tar.gz | tail -n +$((KEEP + 1)) | while read -r f; do
    echo "  [prune] $(basename "$f")"; rm -f "$f"
  done
fi
echo "  当前备份数: $(ls -1 "$DEST"/DiT_code_*.tar.gz 2>/dev/null | wc -l) (保留最近 $KEEP)"
echo
echo "拉到本地 (在你自己机器上执行):"
echo "  scp 4090:$OUT  g:\\GitHub\\DiT\\_ot_scratch\\"

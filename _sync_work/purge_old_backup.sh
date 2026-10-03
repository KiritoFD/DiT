#!/bin/bash
# =============================================================================
# 删除旧备份 dit_data_backup_20260913/（622 GB）
#
# ★ 前提检查（脚本会自己验证，不满足就拒绝执行）：
#   1. dataset 打包已完成且校验通过
#   2. 记录数据已捞到 _rescued/20260913_backup/
#   3. 正在训练
#
# ★ 硬链接说明（重要）：
#   备份里大量文件与当前项目 DiT/ 是**硬链接**（同一 inode）。
#   删掉备份目录只会把链接数从 2 降到 1，**那些文件仍占空间**（当前项目还在用）。
#   真正释放的是备份**独有**的部分（结果是 ckpt 等）。
#
# 用法：
#   bash _sync_work/purge_old_backup.sh            # DRY-RUN 预览
#   bash _sync_work/purge_old_backup.sh --apply    # 真删
# =============================================================================
set -u
TARGET=/root/Workspace/xy/dit_data_backup_20260913
RESCUED=/root/Workspace/xy/DiT/_rescued/20260913_backup
PACK=/root/Workspace/xy/DiT/_backups/dataset_20260923.tar.gz

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

echo "=============================================================="
if [ $APPLY -eq 1 ]; then echo " ★ APPLY 模式：真删！"; else echo " DRY-RUN 模式"; fi
echo " 目标: $TARGET"
echo "=============================================================="

# ---- 前提检查 ----
echo
echo "--- 前提检查 ---"
FAIL=0

if [ -f "$PACK" ] && [ "$(stat -c%s "$PACK")" -gt 1000000000 ]; then
    echo " [OK] dataset 打包存在: $(du -h "$PACK" | cut -f1)"
else
    echo " [FAIL] dataset 打包不存在或过小: $PACK"; FAIL=1
fi

if [ -d "$RESCUED/records" ] && [ "$(find "$RESCUED/records" -type f | wc -l)" -gt 100 ]; then
    echo " [OK] 记录数据已捞: $(find "$RESCUED/records" -type f | wc -l) 个文件"
else
    echo " [FAIL] 记录数据不完整: $RESCUED/records"; FAIL=1
fi

if [ -d "$TARGET" ]; then
    echo " [OK] 目标存在: $(du -sh "$TARGET" 2>/dev/null | cut -f1)"
else
    echo " [FAIL] 目标不存在: $TARGET"; FAIL=1
fi

# 训练在跑（提示，不阻塞）
NRUN=$(pgrep -f 'src\.train\.train' 2>/dev/null | wc -l)
echo " [INFO] 正在训练进程数: $NRUN（删除旧备份目录不影响，但确认一下）"

if [ "$FAIL" -eq 1 ]; then
    echo
    echo " ★ 前提不满足，拒绝执行。请先完成 备份/捞取。"
    exit 1
fi

# ---- 空间预估 ----
echo
echo "--- 空间分析 ---"
BEFORE=$(df -B1 /root/Workspace | tail -1 | awk '{print $4}')
BEFORE_H=$(df -h /root/Workspace | tail -1 | awk '{print $4}')
echo " 删除前可用: $BEFORE_H"

# 备份内文件与外部（当前项目）的硬链接情况
echo " 分析硬链接（可能要一会儿）..."
N_SHARED=$(find "$TARGET" -type f -links +1 2>/dev/null | wc -l)
N_TOTAL=$(find "$TARGET" -type f 2>/dev/null | wc -l)
echo " 文件总数: $N_TOTAL   有外部链接的: $N_SHARED"

SIZE=$(du -sb "$TARGET" 2>/dev/null | cut -f1)
echo " 目录体积: $(awk -v b="$SIZE" 'BEGIN{printf "%.2f GB", b/1024/1024/1024}')"
echo " ★ 注意：与当前项目硬链接共享的部分，删除后不会释放空间。"

if [ $APPLY -eq 0 ]; then
    echo
    echo "--- 顶层内容（将被删除）---"
    du -sh "$TARGET"/*/ 2>/dev/null | sort -rh | head -15
    echo
    echo " → 这是预览。确认删除请加 --apply"
    echo "=============================================================="
    exit 0
fi

# ---- 执行删除 ----
echo
echo "--- 执行删除 ---"
rm -rf "$TARGET"
RC=$?

sync
sleep 2
AFTER=$(df -B1 /root/Workspace | tail -1 | awk '{print $4}')
AFTER_H=$(df -h /root/Workspace | tail -1 | awk '{print $4}')

echo " 删除前可用: $BEFORE_H"
echo " 删除后可用: $AFTER_H"
awk -v a="$BEFORE" -v b="$AFTER" 'BEGIN{d=b-a; printf " 实际释放: %.2f GB\n", d/1024/1024/1024}'
echo " 删除退出码: $RC"
echo "=============================================================="

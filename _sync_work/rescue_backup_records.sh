#!/bin/bash
# =============================================================================
# 从 dit_data_backup_20260913/ 捞取记录数据（删除备份前的抢救）
#
# 捞取内容：
#   1. 所有 csv / json / md / txt / log  —— 记录与配置
#   2. eval 结果图片 —— eval_samples_ctrl/ + eval_strata/ 目录
#   3. 骨架/结构对照图 —— 小尺寸的 struct/skel 对照（可选）
#
# 输出到：_rescued/20260913_backup/
# =============================================================================
set -u
SRC=/root/Workspace/xy/dit_data_backup_20260913
DST=/root/Workspace/xy/DiT/_rescued/20260913_backup

mkdir -p "$DST"

echo "=============================================================="
echo " 从备份捞取记录数据"
echo " 源: $SRC"
echo " 目标: $DST"
echo "=============================================================="

# ---- 1. 记录数据：csv / json / md / txt ----
echo
echo "--- [1/3] 记录数据 (csv/json/md/txt/log) ---"
cd "$SRC" || exit 1
find . -type f \( -name '*.csv' -o -name '*.json' -o -name '*.md' \
     -o -name '*.txt' -o -name '*.log' -o -name '*.yaml' -o -name '*.yml' \) \
     -print0 2>/dev/null | while IFS= read -r -d '' f; do
    rel=${f#./}
    mkdir -p "$DST/records/$(dirname "$rel")"
    cp -p "$f" "$DST/records/$rel"
done
N_REC=$(find "$DST/records" -type f 2>/dev/null | wc -l)
echo "  已捞: $N_REC 个文件"
du -sh "$DST/records" 2>/dev/null

# ---- 2. eval 结果图片 ----
echo
echo "--- [2/3] eval 结果图片 (eval_samples_ctrl / eval_strata) ---"
find . -type d -name 'eval_samples*' -print0 2>/dev/null | while IFS= read -r -d '' d; do
    rel=${d#./}
    mkdir -p "$DST/eval_imgs/$(dirname "$rel")"
    cp -rp "$d" "$DST/eval_imgs/$(dirname "$rel")/"
done
# eval_strata 也一起（小样本集）
if [ -d "5script/eval_strata" ]; then
    mkdir -p "$DST/eval_imgs/5script"
    cp -rp "5script/eval_strata" "$DST/eval_imgs/5script/"
fi
N_IMG=$(find "$DST/eval_imgs" -type f 2>/dev/null | wc -l)
echo "  已捞: $N_IMG 张图"
du -sh "$DST/eval_imgs" 2>/dev/null

# ---- 3. 顶层文档/脚本记录 ----
echo
echo "--- [3/3] 顶层 md/csv/json（项目级文档） ---"
find . -maxdepth 1 -type f \( -name '*.md' -o -name '*.csv' -o -name '*.json' -o -name '*.txt' \) \
     -print0 2>/dev/null | while IFS= read -r -d '' f; do
    mkdir -p "$DST/top"
    cp -p "$f" "$DST/top/"
done
N_TOP=$(find "$DST/top" -type f 2>/dev/null | wc -l)
echo "  已捞: $N_TOP 个文件"

echo
echo "=============================================================="
echo " 汇总"
echo "=============================================================="
du -sh "$DST"/*/ 2>/dev/null
echo "---"
du -sh "$DST"
echo " 总文件数: $(find "$DST" -type f | wc -l)"
echo "=============================================================="

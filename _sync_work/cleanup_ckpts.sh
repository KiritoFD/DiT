#!/bin/bash
# =============================================================================
# 磁盘清理：旧实验 ckpt 瘦身
#
# ★ 默认策略：每个 checkpoints/ 目录只留【最好】+【最后】
#     最好 = 名字含 best / peak / BEST（人工或脚本标注的最佳点）
#     最后 = step 号最大的那个（= 收敛点/最新点）
#   两者常常是同一个文件 -> 那就只留 1 个。
#   想额外多留中间点：--keep N   （额外再留 step 最大的 N 个，含"最后"）
#
# ★ 默认 **DRY-RUN**（只打印，不删）。确认无误后加 --apply 才真删。
#
# 用法：
#   bash _sync_work/cleanup_ckpts.sh                 # 预览（不动）
#   bash _sync_work/cleanup_ckpts.sh --keep 3        # 预览：每目录最多留 3 个
#   bash _sync_work/cleanup_ckpts.sh --apply         # 按默认（最好+最后）真删
#   bash _sync_work/cleanup_ckpts.sh --include-archive --apply
#
# 保留规则（每个 checkpoints/ 目录内）：
#   1. ★ 名字含 best / peak / BEST 的 -> 无条件保留（每个都保）
#   2. step 号最大的那个（"最后"）   -> 无条件保留
#   3. --keep N 时，再额外保 step 最大的 N 个（N>=2 才有意义）
#
# 安全措施：
#   - 只删 <dir>/checkpoints/*.pt 及其 .done 和 eval_<base>/ 同名目录
#   - 不删 .pt 以外的任何文件（poster / csv / json / log 全留）
#   - ★ 活跃 run 保护：扫描全库所有 _active_ckpt_dir.txt，把里面写的
#     每个 checkpoints 目录整条路径加入保护集合（精确前缀匹配）
#   - _archive/ 目录默认整个跳过（显式 --include-archive 才处理）
#   - 删除前把清单写到 cleanup_manifest.txt 供事后审计
# =============================================================================
set -u
ROOT=/root/Workspace/xy/DiT
RESULTS_PREFIX="assets/results/"
cd "$ROOT/assets/results" || exit 1

APPLY=0
KEEP_LAST=${KEEP_LAST:-1}      # 1 = 只留"最后"
INCLUDE_ARCHIVE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --apply)            APPLY=1 ;;
        --keep)             KEEP_LAST="${2:-1}"; shift ;;
        --include-archive)  INCLUDE_ARCHIVE=1 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

MANIFEST=$ROOT/_sync_work/cleanup_manifest.txt
PROTECT_LIST=$ROOT/_sync_work/_protected_ckpt_dirs.txt

# ---- ★ 构建保护集合：扫描全库所有 _active_ckpt_dir.txt ----
: > "$PROTECT_LIST"
while IFS= read -r f; do
    [ -z "$f" ] && continue
    # 文件内容可能有多行，逐行读；去空白
    while IFS= read -r line; do
        line=$(echo "$line" | tr -d '\r' | sed 's:/*$::' | sed 's:^\./::')
        [ -z "$line" ] && continue
        # ★ 归一化：保护文件里写的是 assets/results/xxx/... 形式的**相对项目根**路径，
        #   而本脚本已 cd 到 assets/results/，CPDIR 是 xxx/... ->
        #   必须剥掉 assets/results/ 前缀，否则前缀匹配永不命中（保护静默失效！）
        line=${line#"$RESULTS_PREFIX"}
        [ -z "$line" ] && continue
        echo "$line" >> "$PROTECT_LIST"
    done < "$f"
done < <(find . -maxdepth 2 -name '_active_ckpt_dir.txt' 2>/dev/null)

# ---- ★ 硬保护 2：正在跑的训练进程对应的 run 目录 ----
# 从进程命令行里解析 --config <path>，取出 run 名，把 assets/results/<run>/ 整条加入保护。
# 用途：run 刚起还没写第一个 ckpt 时，24h 活跃检测抓不到，必须靠这个兜住。
RUNNING_NAMES=""
while IFS= read -r cmdline; do
    [ -z "$cmdline" ] && continue
    cfg=$(echo "$cmdline" | grep -oE '\-\-config[= ][^ ]+' | awk '{print $NF}' | head -1)
    [ -z "$cfg" ] && continue
    rn=$(basename "$cfg" | sed 's/\.run\.json$//; s/\.json$//')
    [ -z "$rn" ] && continue
    RUNNING_NAMES="$RUNNING_NAMES $rn"
    if [ -d "$rn" ]; then
        echo "$rn" >> "$PROTECT_LIST"        # 目录本身（其下所有 checkpoints 都被保护）
    else
        # 目录尚未创建：按名字前缀保护（is_protected 支持前缀匹配）
        echo "$rn" >> "$PROTECT_LIST"
    fi
done < <(pgrep -af 'src\.train\.train' 2>/dev/null)

# ★ 活跃度过滤：保护集里很多是历史 run 遗留的陈旧文件（几周前就训完了）。
#   真正的"活跃"= 该目录下 24h 内还有新写入。陈旧条目降级为普通目录
#   （仍走"最好+最后"规则，不会裸删）。
ACTIVE_HOURS=${ACTIVE_HOURS:-24}
sort -u "$PROTECT_LIST" > "$PROTECT_LIST.raw"
: > "$PROTECT_LIST"
NSTALE=0
while IFS= read -r p; do
    [ -z "$p" ] && continue
    # ★ 正在跑的训练 run -> 豁免活跃度检查，无条件保护
    _is_running=0
    for rn in $RUNNING_NAMES; do
        case "$p" in "$rn"|"$rn"/*|"assets/results/$rn"|"assets/results/$rn"/*) _is_running=1 ;; esac
    done
    if [ "$_is_running" -eq 1 ]; then
        echo "$p" >> "$PROTECT_LIST"
        continue
    fi
    if [ -d "$p" ] && [ -n "$(find "$p" -maxdepth 1 -name '*.pt' -mmin -$((ACTIVE_HOURS*60)) 2>/dev/null | head -1)" ]; then
        echo "$p" >> "$PROTECT_LIST"
    else
        NSTALE=$((NSTALE + 1))
    fi
done < "$PROTECT_LIST.raw"
NPROT=$(wc -l < "$PROTECT_LIST")

echo "=============================================================="
if [ $APPLY -eq 1 ]; then
    echo " ★ APPLY 模式：真删！"
else
    echo " DRY-RUN 模式：只预览，不动任何文件"
fi
echo " 保留策略: 最好(best/peak) + 最后(step最大)" \
     $([ "$KEEP_LAST" -ge 2 ] && echo "+ 额外 step 最大 $KEEP_LAST 个")
echo " ★ 活跃 run 保护: $NPROT 个目录（${ACTIVE_HOURS}h 内有新 ckpt 写入）"
[ "$NSTALE" -gt 0 ] && echo "   陈旧保护条目已降级: $NSTALE 个（仍按最好+最后保留，不会裸删）"
[ $INCLUDE_ARCHIVE -eq 0 ] && echo " _archive/ : 跳过（加 --include-archive 才处理）"
echo "=============================================================="

is_protected() {
    local cpdir="$1" p
    while IFS= read -r p; do
        [ -z "$p" ] && continue
        # 精确匹配：保护路径 == 当前目录，或当前目录在其下
        case "$cpdir" in
            "$p"|"$p"/*) return 0 ;;
        esac
    done < "$PROTECT_LIST"
    return 1
}

: > "$MANIFEST"
TOTAL_FREE=0; TOTAL_DEL=0; TOTAL_KEPT=0; NDIR=0; NSKIP=0

while IFS= read -r CPDIR; do
    CPDIR=${CPDIR#./}

    if is_protected "$CPDIR"; then
        echo "[保护] 活跃 run: $CPDIR"
        NSKIP=$((NSKIP + 1))
        continue
    fi
    if [ $INCLUDE_ARCHIVE -eq 0 ]; then
        case "$CPDIR" in _archive/*|*/_archive/*) continue ;; esac
    fi

    mapfile -t ALL < <(find "$CPDIR" -maxdepth 1 -name '*.pt' -printf '%f\n' 2>/dev/null | sort)
    N=${#ALL[@]}
    [ "$N" -eq 0 ] && continue
    NDIR=$((NDIR + 1))

    declare -A KEEP
    # (1) best / peak -> 全保
    for f in "${ALL[@]}"; do
        case "$f" in *best*|*peak*|*BEST*) KEEP["$f"]=1 ;; esac
    done
    # (2)(3) step 最大的 KEEP_LAST 个（KEEP_LAST=1 就是"最后"）
    for ((i=N-1; i>=0 && i>=N-KEEP_LAST; i--)); do KEEP["${ALL[$i]}"]=1; done

    NKEEP=${#KEEP[@]}
    NDEL=$(( N - NKEEP ))
    TOTAL_KEPT=$((TOTAL_KEPT + NKEEP))

    if [ "$NDEL" -le 0 ]; then
        printf '%-72s %4d -> 全留\n' "$CPDIR" "$N"
        unset KEEP; continue
    fi

    BYTES=0
    for f in "${ALL[@]}"; do
        [ -n "${KEEP[$f]:-}" ] && continue
        SZ=$(stat -c%s "$CPDIR/$f" 2>/dev/null || echo 0)
        BYTES=$((BYTES + SZ))
        echo "DEL $CPDIR/$f" >> "$MANIFEST"
    done
    GB=$(awk -v b="$BYTES" 'BEGIN{printf "%.2f", b/1024/1024/1024}')
    TOTAL_FREE=$((TOTAL_FREE + BYTES))
    TOTAL_DEL=$((TOTAL_DEL + NDEL))

    printf '%-72s %4d -> 留 %d 删 %d  释放 %s GB\n' "$CPDIR" "$N" "$NKEEP" "$NDEL" "$GB"

    if [ $APPLY -eq 1 ]; then
        for f in "${ALL[@]}"; do
            [ -n "${KEEP[$f]:-}" ] && continue
            rm -f "$CPDIR/$f" "$CPDIR/$f.done"
            base=${f%.pt}
            rm -rf "$CPDIR/eval_$base" 2>/dev/null
        done
    fi
    unset KEEP
done < <(find . -type d -name 'checkpoints' 2>/dev/null)

echo "=============================================================="
awk -v b="$TOTAL_FREE" 'BEGIN{printf " 可释放: %.2f GB\n", b/1024/1024/1024}'
echo " 扫描目录: $NDIR 个   保护跳过: $NSKIP 个"
echo " 将删除: $TOTAL_DEL 个 ckpt   保留: $TOTAL_KEPT 个"
echo " 清单: $MANIFEST"
if [ $APPLY -eq 0 ]; then
    echo
    echo " → 这是预览。确认真删请加 --apply"
fi
echo "=============================================================="

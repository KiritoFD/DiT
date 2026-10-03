#!/usr/bin/env bash
# 归档两阶段/废弃代码 -> _archive/20261003_twostage/  (移动, 不删除; 带 manifest + 一键还原)
#
# ⚠ 铁律: 绝不移动会被**活代码 import** 的东西 —— 否则 A 跑完后 B 起不来。
#   已知活性依赖 (全部保留):
#     src/model/dit.py:1676   from .legacy.controlnet import ZeroAdaLNInjection
#     src/model/__init__.py:37 from .legacy.controlnet import (...)
#   因此 src/model/legacy/ **整体不动** (只可动其非 controlnet 文件, 本轮先不动)。
set -u
cd /root/Workspace/xy/DiT || exit 1
TS=20261003_twostage
ARC="_archive/$TS"
mkdir -p "$ARC"

# 移动清单: 全部是 "无任何 import 引用" 的脚本 / 配置 / 备份 / 临时目录
ITEMS=(
  "src/train/legacy:legacy-训练脚本(无 import 引用)"
  "src/eval/legacy:legacy-评测脚本(无 import 引用)"
  "src/train/configs.bak_922:2026-09-22 全量配置备份(189 文件, 含中文名)"
  "tmp_revert:临时回滚目录"
  "src/train/train_controlnet.py:两阶段 ControlNet 训练入口"
  "src/train/train_repa.py:REPA 独立训练入口(旧)"
  "_sync_work/train_joint_stage1_stage2.py:两阶段联合训练入口"
  "tools/gen_predskel_stage1.py:两阶段-stage1 产物生成"
  "tools/stage2_qwen_anchor.py:两阶段-stage2 锚定"
  "tools/diag_two_stage.py:两阶段诊断"
)
# 两阶段 shell 脚本 (通配)
SH_GLOB=("_sync_work/launch_two_stage.sh" "_sync_work/run_two_stage.sh" "_sync_work/redo_twostage.sh"
         "_sync_work/two_stage_inner.sh" "_sync_work/run_stage1_serial.sh" "_sync_work/run_stage1_bb.sh"
         "_sync_work/run_gen_stage1.sh" "_sync_work/launch_fs6_stage1.sh" "_sync_work/launch_fs6_stage2.sh"
         "_sync_work/commit_stage1.sh" "_sync_work/eval_stage1_down.sh" "_sync_work/stage1_status.sh")

MAN="$ARC/MANIFEST.tsv"
: > "$MAN"
printf '%-58s %-10s %s\n' "文件/目录" "大小" "说明"
echo "────────────────────────────────────────────────────────────────────────────"

move_one() {
  local src="$1" desc="$2"
  [ -e "$src" ] || { printf '  %-56s %s\n' "$src" "(不存在, 跳过)"; return; }
  local sz; sz=$(du -sh "$src" 2>/dev/null | cut -f1)
  local dst="$ARC/$(echo "$src" | tr '/' '__')"
  mv "$src" "$dst"
  printf '  %-56s %-8s %s\n' "$src" "$sz" "$desc"
  printf '%s\t%s\t%s\n' "$src" "$sz" "$desc" >> "$MAN"
}

for it in "${ITEMS[@]}"; do
  s="${it%%:*}"; d="${it#*:}"
  move_one "$s" "$d"
done
for s in "${SH_GLOB[@]}"; do
  move_one "$s" "两阶段 shell 脚本"
done

echo
echo "  --- __pycache__ 清理 (1532 个, 可自动重建) ---"
before=$(du -sh . 2>/dev/null | cut -f1)
find . -name '__pycache__' -type d -not -path './.git/*' -prune -exec rm -rf {} + 2>/dev/null
after=$(du -sh . 2>/dev/null | cut -f1)
echo "    工作区: $before -> $after"

# 一键还原
cat > "$ARC/RESTORE.sh" <<'EOF'
#!/usr/bin/env bash
# 一键还原本次归档 (按 MANIFEST.tsv 反向 mv)
set -u
cd "$(dirname "$0")/../.." || exit 1
HERE="$(cd "$(dirname "$0")" && pwd)"
while IFS=$'\t' read -r src sz desc; do
  base="$(echo "$src" | tr '/' '__')"
  [ -e "$HERE/$base" ] || { echo "  (缺) $base"; continue; }
  mkdir -p "$(dirname "$src")"
  mv "$HERE/$base" "$src"
  echo "  还原 $src"
done < "$HERE/MANIFEST.tsv"
echo "done"
EOF
chmod +x "$ARC/RESTORE.sh"

echo
echo "  --- 归档结果 ---"
echo "  位置: $ARC"
echo "  条目: $(tail -n +1 "$MAN" | wc -l) 个, 大小 $(du -sh "$ARC" | cut -f1)"
echo "  还原: bash $ARC/RESTORE.sh"
echo
echo "  --- 活性依赖必须保留的 (确认仍在) ---"
for f in src/model/legacy/controlnet.py src/model/dit.py src/model/__init__.py src/model/train.py src/model/controlnet.py; do
  printf '    %-40s %s\n' "$f" "$([ -f "$f" ] && echo '✓ 在' || echo '✗ 不见了!')"
done

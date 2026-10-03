#!/usr/bin/env bash
# 1) 写 .gitignore (纯新增, 安全)  2) 精确的 import 引用扫描 (归档可行性)  3) 生成归档清单(不执行)
set -u
cd /root/Workspace/xy/DiT || exit 1

echo "════ [1] 写 .gitignore ════"
if [ -f .gitignore ]; then
  echo "  已存在, 不动 (内容前 10 行):"; head -10 .gitignore | sed 's/^/    /'
else
  cat > .gitignore <<'EOF'
# ── 大件/缓存 (绝不能进 git) ────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.pt
*.pth
*.ckpt
*.npz
*.npy
*.tar.gz
*.zip

# 训练产物 / 数据
data/
exp-std/data/
exp-std/runs*/
exp-std/runs_AB/
exp-std/runs_smoke/
assets/results/
assets/ink_eval*/
_ot_scratch/
_backups/
/tmp_revert/
src/utils/std_glyph_latent_v2/

# 备份/临时/日志
*.bak
*.bak[0-9]*
*.bak_*
*.bak_oldcols*
*.log
_sync_work/_AB_patch/backup_*/
_sync_work/_inj3_bak/
*.orig
*.rej

# 采样图 / 海报 (可重生成)
*_poster.png
eval_samples_ctrl/
posters/

# 归档目录 (本地留档, 不入 git)
_archive/
EOF
  echo "  ✓ 已创建 .gitignore ($(wc -l < .gitignore) 行)"
fi

echo
echo "════ [2] 精确 import 引用扫描 (归档可行性) ════"
scan() {
  local mod="$1"
  # 精确匹配 import / from ... import 形式, 排除自身与 legacy 内部互引
  local hits
  hits=$(grep -rnE "(^|[^A-Za-z0-9_])(import[[:space:]]+${mod}([[:space:]]|$)|from[[:space:]]+[A-Za-z0-9_.]*${mod}[[:space:]]+import)" \
        --include=*.py --include=*.sh src tools _sync_work scripts 2>/dev/null | grep -v "/${mod}\.py:" | head -4)
  printf '  %-28s %s\n' "$mod" "${hits:-✓ 无任何 import 引用}"
}
for m in train_controlnet train_repa train_joint_stage1_stage2 controlnet repa dit_skel_1cond train_pixel_classifier eval_all_classifiers; do
  scan "$m"
done

echo
echo "  --- legacy 目录是否被非 legacy 代码 import ---"
for d in src/model/legacy src/train/legacy src/eval/legacy; do
  h=$(grep -rn "from .*legacy\|import .*legacy" --include=*.py src tools 2>/dev/null | grep -v "$d" | head -3)
  printf '  %-24s %s\n' "$d" "${h:-✓ 无外部引用}"
done

echo
echo "════ [3] 两阶段(ControlNet/stage1-2) 相关文件清单 ════"
for pat in '*controlnet*' '*stage1*' '*stage2*' '*twostage*' '*two_stage*'; do
  find src tools scripts _sync_work -maxdepth 3 -iname "$pat" -type f 2>/dev/null | head -20 | sed 's/^/    /'
done | sort -u

echo
echo "  --- 归档候选 (体积) ---"
for d in src/model/legacy src/train/legacy src/eval/legacy src/train/configs.bak_922 tmp_revert; do
  [ -e "$d" ] && printf '    %-32s %-8s %s 个文件\n' "$d" "$(du -sh $d 2>/dev/null | cut -f1)" "$(find $d -type f 2>/dev/null | wc -l)"
done
[ -f src/model/train.py ] && printf '    %-32s %-8s %s 行\n' "src/model/train.py" "$(du -sh src/model/train.py | cut -f1)" "$(wc -l < src/model/train.py)"

echo
echo "════ [4] git 现状摘要 (待你点头才提交) ════"
echo "  已跟踪 src/*.py : $(git ls-files 'src/**/*.py' 2>/dev/null | wc -l)"
echo "  未跟踪总数      : $(git status --porcelain 2>/dev/null | grep -c '^??')"
echo "  未跟踪中【会被 .gitignore 挡住】的: 无法离线判定, 提交前用 git status --porcelain 复核"
echo
echo "  建议的提交序列 (未执行):"
echo "    git add .gitignore"
echo "    git add -u                                   # 20 处修改 + 10 处删除"
echo "    git add src/model/dit.py src/eval/in_mem_eval.py src/utils/latent_dataset.py \\"
echo "            src/train/ckpt.py src/utils/deform_aug.py src/train src/train/configs"
echo "    git status --porcelain | head -50            # ★ 先复核, 确认没有 .pt/大件"
echo "    git commit -m 'sync working tree: 核心训练/评测代码入库 + .gitignore'"

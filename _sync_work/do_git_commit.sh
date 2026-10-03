#!/usr/bin/env bash
# 把工作区纳入版本控制 (用户明确要求)。
# 设计: 分 3 个 commit, 每个 commit 前过"大件守卫"(>5MB 直接拒交),
#       第一个 commit 只放 .gitignore —— 之后 add 才受它约束。
#       全程**不改 git config**; 若身份没配就中止并提示。
set -u
cd /root/Workspace/xy/DiT || exit 1

echo "════ [0] 前置检查 ════"
echo "  当前 HEAD: $(git rev-parse --short HEAD 2>/dev/null)  $(git log -1 --format=%s 2>/dev/null | cut -c1-60)"
U_NAME=$(git config user.name 2>/dev/null || true)
U_MAIL=$(git config user.email 2>/dev/null || true)
if [ -z "$U_NAME" ] || [ -z "$U_MAIL" ]; then
  echo "  ✗ git 身份未配置 (user.name='$U_NAME' user.email='$U_MAIL')"
  echo "    我不改 git config。请先执行(任选其一):"
  echo "      git config --global user.name  '你的名字'"
  echo "      git config --global user.email 'you@example.com'"
  exit 2
fi
echo "  身份: $U_NAME <$U_MAIL>"
echo "  未跟踪/修改条目: $(git status --porcelain | wc -l)   .git 体积: $(du -sh .git | cut -f1)"
echo "  .gitignore: $([ -f .gitignore ] && echo "$(wc -l < .gitignore) 行 ✓" || echo '✗ 缺失!')"

# 大件守卫: 检查暂存区里 >5MB 的文件
guard() {
  local big
  big=$(git diff --cached --name-only | while read -r f; do
          [ -f "$f" ] || continue
          local sz; sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
          [ "$sz" -gt 5242880 ] && printf '      %s  %.1f MB\n' "$f" "$(echo "$sz/1048576" | bc -l)"
        done)
  if [ -n "$big" ]; then
    echo "  ✗✗ 暂存区含 >5MB 文件, 拒绝提交:"; echo "$big"
    echo "     -> 请补 .gitignore 或 git reset 后重来"
    return 1
  fi
  echo "  ✓ 大件守卫通过 (暂存 $(git diff --cached --name-only | wc -l) 个文件, 无 >5MB)"
  return 0
}

echo
echo "════ [1/3] 只提交 .gitignore ════"
git add .gitignore
if git diff --cached --quiet; then
  echo "  (无变化, 跳过)"
else
  guard || exit 3
  git commit -q -m "chore(git): 新增 .gitignore —— 排除 *.pt/*.npz/__pycache__(1532 个)/runs*/std_glyph_latent_v2(517M)/*.bak_oldcols 等

背景: 仓库此前无 .gitignore, 下一次 git add 会把 517MB 缓存与上千个 __pycache__ 一起提交。" \
    && echo "  ✓ $(git log -1 --format='%h %s' | cut -c1-90)"
fi

echo
echo "════ [2/3] 已跟踪文件的增删改 + 核心源码 ════"
git add -u
# 核心本体: 显式点名 (它们此前完全不在 git 里)
CORE="src/model/dit.py src/model/injections.py src/model/__init__.py src/model/modules.py
      src/eval/in_mem_eval.py src/eval/model_io.py src/eval/inference.py src/eval/metrics.py
      src/utils/latent_dataset.py src/train/ckpt.py src/utils/deform_aug.py
      src/train/train.py src/train/cli.py src/loss/flow_matching.py"
for f in $CORE; do [ -f "$f" ] && git add "$f" 2>/dev/null; done
guard || exit 3
git diff --cached --stat | tail -4
git commit -q -m "sync: 核心训练/评测本体入库 + 解开 legacy 依赖

- 入库(此前 **完全未跟踪**): src/model/dit.py, src/eval/in_mem_eval.py,
  src/utils/latent_dataset.py, src/train/ckpt.py, src/utils/deform_aug.py 等
- 新增 src/model/injections.py: ZeroAdaLNInjection 从 legacy/controlnet.py 逐字上移,
  切断'训练主路径 -> 遗物'依赖 (已验证: 实现逐字一致 + 历史 adaLN ckpt 键名同构)
- src/model/__init__.py: 去掉 import legacy 的 try/except 隐式回退" \
  && echo "  ✓ $(git log -1 --format='%h %s' | cut -c1-90)"

echo
echo "════ [3/3] 其余源码/配置/脚本/清单 ════"
git add src tools docs scripts _sync_work exp-std/csv assets/*.json 2>/dev/null
guard || exit 3
git diff --cached --stat | tail -4
git commit -q -m "sync: 补齐 src/tools/scripts/_sync_work/docs 与实验配置

含 .gitignore 生效后仍应入库的文本资产 (配置 json / 脚本 sh / 文档 md / 清单 csv)。
⚠ 二进制大件 (*.pt/*.npz/*.png) 一律不入库 —— 用对应 tools/ 脚本重建。" \
  && echo "  ✓ $(git log -1 --format='%h %s' | cut -c1-90)"

echo
echo "════ [4] 提交后核验 ════"
echo "  git ls-files src 数量: $(git ls-files src | wc -l)  (提交前是 13)"
echo "  --- 严禁混入的大件/缓存, 应为空 ---"
git ls-files | grep -E '\.(pt|pth|ckpt|npz|npy|png|jpg|tar\.gz)$|__pycache__|\.log$|bak_oldcols' | head -10 \
  && echo "  ✗ 上面有不该入库的!" || echo "  ✓ 干净: 无二进制/缓存/日志/备份"
echo "  --- 仓库里最大的 5 个已跟踪文件 ---"
git ls-files | xargs -r du -h 2>/dev/null | sort -rh | head -5
echo
echo "  --- 最近 3 个 commit ---"
git --no-pager log --oneline -3
echo
echo "  --- 工作区剩余未跟踪 (应只剩大件/产物/临时) ---"
git status --porcelain | grep '^??' | wc -l
git status --porcelain | grep '^??' | head -8

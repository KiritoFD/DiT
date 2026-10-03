#!/usr/bin/env bash
# 只暂存 + 全量守卫 + 报告, **不提交** (等 git 身份就位)
set -u
cd /root/Workspace/xy/DiT || exit 1

guard() {
  local big
  big=$(git diff --cached --name-only | while read -r f; do
          [ -f "$f" ] || continue
          local sz; sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
          [ "$sz" -gt 5242880 ] && printf '      %s  %.1f MB\n' "$f" "$(echo "scale=1; $sz/1048576" | bc)"
        done)
  if [ -n "$big" ]; then
    echo "  ✗ 暂存区含 >5MB 文件:"; echo "$big"; return 1
  fi
  echo "  ✓ 守卫通过 (暂存 $(git diff --cached --name-only | wc -l) 个, 无 >5MB)"
  return 0
}

git reset -q 2>/dev/null || true      # 确保从干净索引开始

echo "════ [1] .gitignore ════"
git add .gitignore; guard || exit 3

echo
echo "════ [2] 已跟踪增删改 + 核心本体 ════"
git add -u
CORE="src/model/dit.py src/model/injections.py src/model/__init__.py src/model/modules.py
      src/eval/in_mem_eval.py src/eval/model_io.py src/eval/inference.py src/eval/metrics.py
      src/utils/latent_dataset.py src/train/ckpt.py src/utils/deform_aug.py
      src/train/train.py src/train/cli.py src/loss/flow_matching.py"
for f in $CORE; do [ -f "$f" ] && git add "$f" 2>/dev/null; done
guard || exit 3
echo "  --- 被修改的已跟踪文件 (前 8) ---"
git diff --cached --name-status | grep '^M' | head -8 | sed 's/^/    /'
echo "  --- 被删除的已跟踪文件 ---"
git diff --cached --name-status | grep '^D' | head -12 | sed 's/^/    /'

echo
echo "════ [3] 其余源码/配置/脚本/文档/清单 ════"
git add src tools docs scripts _sync_work exp-std/csv 2>/dev/null
guard || exit 3

echo
echo "════ [4] 复核 ════"
echo "  暂存文件总数: $(git diff --cached --name-only | wc -l)"
echo "  其中 src/*.py: $(git diff --cached --name-only | grep -c '^src/.*\.py$')"
echo "  ★ git 仓库里已有的 src 文件数: $(git ls-files src | wc -l) (提交前 13)"
echo
echo "  --- 严禁混入 (应无输出) ---"
git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|npz|npy|png|jpg|tar\.gz)$|__pycache__|\.log$|bak_oldcols' | head -10 || true
echo "  --- 暂存的文件类型分布 ---"
git diff --cached --name-only | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10 | sed 's/^/    /'
echo "  --- 暂存的 src/model 下文件 ---"
git diff --cached --name-only | grep '^src/model/' | head -12 | sed 's/^/    /'

echo
echo "════ [5] 就绪, 只等你给身份 (三选一) ════"
echo "  A) 你自己配:  git config --global user.name 'X' && git config --global user.email 'Y'"
echo "     然后跑:    bash _sync_work/do_git_commit.sh"
echo "  B) 授权我用环境变量提交 (不改 config):"
echo "     GIT_AUTHOR_NAME=X GIT_AUTHOR_EMAIL=Y GIT_COMMITTER_NAME=X GIT_COMMITTER_EMAIL=Y git commit -m '...'"
echo "  C) 先看当前暂存内容: git diff --cached --stat | tail -20"
echo
echo "  当前暂存(可直接 git commit 的三段应分 3 次, 见 do_git_commit.sh)"

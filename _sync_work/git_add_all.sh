#!/usr/bin/env bash
# 重新暂存: git add -A (受 .gitignore 约束), 错误显式打印, 全量守卫
set -u
cd /root/Workspace/xy/DiT || exit 1

git reset -q

echo "════ git add -A ════"
if git add -A; then
  echo "  ✓ add 成功"
else
  echo "  ✗ add 失败 (上面是真实报错) —— 这才是我上次 42 个的真相"
  exit 2
fi

echo
echo "════ 暂存复核 ════"
N=$(git diff --cached --name-only | wc -l)
echo "  暂存文件总数: $N"
echo "  其中 src/**/*.py: $(git diff --cached --name-only | grep -c '^src/.*\.py$')"
echo "  仓库 src 文件总数(暂存后): $(git ls-files src | wc -l)  (提交前 13)"

echo
echo "  --- 类型分布 ---"
git diff --cached --name-only | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -8 | sed 's/^/    /'

echo
echo "════ 大件守卫 (>5MB, 只看新增/修改) ════"
FOUND=0
while read -r st f; do
  case "$st" in
    A|M|AM|MM) ;;
    *) continue ;;
  esac
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt 5242880 ]; then
    printf '    %s  %s MB\n' "$f" "$((sz / 1048576))"
    FOUND=1
  fi
done < <(git diff --cached --name-status)
[ "$FOUND" -eq 0 ] && echo "    ✓ 无 >5MB 文件"

echo
echo "════ 严禁混入 (只查新增/修改; 删除不算) ════"
BAD=$(git diff --cached --name-status | grep -E '^[AM]' \
      | grep -E '\.(pt|pth|ckpt|npz|npy|png|jpg|tar\.gz)$|__pycache__|\.log$|bak_oldcols' | head -10)
if [ -n "$BAD" ]; then echo "$BAD" | sed 's/^/    ✗ /'; else echo "    ✓ 干净"; fi

echo
echo "════ 仓库里最大的 5 个已跟踪文件 (含历史) ════"
git ls-files | xargs -r du -h 2>/dev/null | sort -rh | head -5 | sed 's/^/    /'

echo
echo "════ 仍未跟踪的 (应只剩被 gitignore 的产物) ════"
echo "    未跟踪: $(git status --porcelain | grep -c '^??')"
git status --porcelain | grep '^??' | head -8 | sed 's/^/    /'

echo
echo "════ 下一步 ════"
echo "  只差 git 身份。二选一:"
echo "    A) 你配: git config --global user.name 'X' && git config --global user.email 'Y'"
echo "       然后: bash _sync_work/do_git_commit.sh"
echo "    B) 授权我用环境变量提交(不改 config), 我照做"

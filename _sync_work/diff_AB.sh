#!/usr/bin/env bash
# 安装前 diff: 确认远端没有我们不知道的改动 (只应看到本次补丁)
set -u
cd /root/Workspace/xy/DiT || exit 1
P=_sync_work/_AB_patch

for f in src/model/dit.py src/train/train.py src/eval/model_io.py src/train/cli.py; do
  b=$(basename "$f")
  echo "########## $f ##########"
  if [ ! -f "$f" ]; then
    echo "  (远端不存在 -> 新增)"
  elif diff -q "$f" "$P/$b" >/dev/null 2>&1; then
    echo "  (完全一致 -> 无需安装)"
  else
    n=$(diff "$f" "$P/$b" 2>/dev/null | grep -c '^[<>]')
    echo "  差分 $n 行 (下面前 45 行):"
    diff -u "$f" "$P/$b" 2>/dev/null | head -45
  fi
  echo
done

echo "########## 新配置文件是否已存在 ##########"
for c in v50_A_space_xattn_style_adaLN_top10.json v51_B_joint_kv_k4_top10.json; do
  if [ -f "src/train/configs/$c" ]; then
    echo "  src/train/configs/$c 已存在 -> 差分:"
    diff -u "src/train/configs/$c" "$P/$c" | head -25
  else
    echo "  src/train/configs/$c (远端不存在 -> 新增)"
  fi
done

echo
echo "########## 远端 dit.py 里是否已有 attn_tau (他处改动痕迹) ##########"
grep -n 'attn_tau' src/model/dit.py | head -10 || echo "  (无)"
echo "########## 远端 train.py 里是否已有 w_style_ortho ##########"
grep -n 'w_style_ortho\|style-ortho' src/train/train.py | head -10 || echo "  (无)"

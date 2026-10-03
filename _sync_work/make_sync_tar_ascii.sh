#!/usr/bin/env bash
# 重打同步包: 只含 **ASCII 文件名** (Windows bsdtar 解不了中文名)
set -u
cd /root/Workspace/xy/DiT || exit 1
OUT=/root/_sync_out
TAR="$OUT/DiT_src_ascii.tar.gz"
LIST="$OUT/ascii_files.txt"

ITEMS=(.gitignore src tools docs scripts _sync_work exp-std/csv)

# 1) 列出所有候选文件, 只留 ASCII 路径
find "${ITEMS[@]}" -type f 2>/dev/null \
  | grep -v -E '(__pycache__|\.pyc$|\.pt$|\.pth$|\.ckpt$|\.npz$|\.npy$|\.bin$|\.png$|\.jpg$|\.jpeg$|\.gif$)' \
  | grep -v -E '\.(ttf|otf|woff|woff2|so|a|o|dll|exe|tgz|zip|tar\.gz|whl)$' \
  | grep -v -E '\.log$|\.bak($|_|[0-9])' \
  | grep -v -E '(std_glyph_latent_v2|cloudflared|mccd)' \
  | grep -v -E '(_AB_patch/backup_)' \
  | LC_ALL=C grep -E '^[ -~]+$' \
  > "$LIST"

N=$(wc -l < "$LIST")
NONASCII=$(find "${ITEMS[@]}" -type f 2>/dev/null | LC_ALL=C grep -cvE '^[ -~]+$' || echo 0)
echo "  候选(ASCII)文件: $N"
echo "  因中文名被排除: $NONASCII 个"

tar --exclude='*.pt' --exclude='*.npz' --exclude='*.png' --exclude='*.ttf' \
    --exclude='*.TTF' --exclude='*.tgz' --exclude='__pycache__' \
    -czf "$TAR" -T "$LIST" 2>/dev/null

RC=$?
if [ "$RC" -ne 0 ] || [ ! -s "$TAR" ]; then echo "✗ 打包失败 rc=$RC"; exit "$RC"; fi
echo "✓ $TAR  ($(du -h "$TAR" | cut -f1), $(tar -tzf "$TAR" | wc -l) 个成员)"

echo
echo "  --- 体积前 6 (确认无大件) ---"
tar -tzvf "$TAR" 2>/dev/null | sort -k3 -n -r | head -6 | awk '{printf "    %8.1f KB  %s\n", $3/1024, $NF}'
echo "  --- 中文名残留检查 (应为 0) ---"
tar -tzf "$TAR" | LC_ALL=C grep -cvE '^[ -~]+$' || echo "    0 ✓"
echo "  --- 关键文件点检 ---"
for f in .gitignore src/model/dit.py src/model/injections.py src/model/__init__.py \
         src/eval/in_mem_eval.py src/utils/latent_dataset.py src/train/train.py \
         src/train/ckpt.py src/train/cli.py tools/plot_curve.py \
         src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
         src/train/configs/v51_B_joint_kv_k4_top10.json; do
  tar -tzf "$TAR" | grep -qx "$f" && printf '    ✓ %s\n' "$f" || printf '    ✗ %s\n' "$f"
done

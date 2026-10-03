#!/usr/bin/env bash
# 打"该入库的文本资产"包 (严格排除二进制/大件/数据), 供拉回本地提交
set -u
cd /root/Workspace/xy/DiT || exit 1
OUT=/root/_sync_out
mkdir -p "$OUT"
TAR="$OUT/DiT_src.tar.gz"

ITEMS=(.gitignore src tools docs scripts _sync_work exp-std/csv)

tar \
  --exclude='*.pt'   --exclude='*.pth'  --exclude='*.ckpt' \
  --exclude='*.npz'  --exclude='*.npy'  --exclude='*.bin'  --exclude='*.tgz' \
  --exclude='*.png'  --exclude='*.jpg'  --exclude='*.jpeg' --exclude='*.gif' \
  --exclude='*.ttf'  --exclude='*.otf'  --exclude='*.woff*' \
  --exclude='*.tar.gz' --exclude='*.zip' --exclude='*.whl' \
  --exclude='*.so'   --exclude='*.a'    --exclude='*.o'    --exclude='*.dll' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' --exclude='.git' \
  --exclude='*.log'  --exclude='*.bak' --exclude='*.bak_*' --exclude='*.bak[0-9]*' \
  --exclude='std_glyph_latent_v2' --exclude='cloudflared' --exclude='mccd*' \
  --exclude='data'   --exclude='runs*' --exclude='results' --exclude='ink_eval*' \
  --exclude='*_AB_patch/backup_*' \
  -czf "$TAR" "${ITEMS[@]}" 2>/dev/null

RC=$?
if [ "$RC" -ne 0 ] || [ ! -s "$TAR" ]; then echo "✗ 打包失败 rc=$RC"; exit "$RC"; fi

echo "✓ $TAR  ($(du -h "$TAR" | cut -f1), $(tar -tzf "$TAR" | wc -l) 个成员)"
echo
echo "--- 体积前 8 大成员 (确认无大件) ---"
tar -tzvf "$TAR" 2>/dev/null | sort -k3 -n -r | head -8 | awk '{printf "  %8.1f KB  %s\n", $3/1024, $NF}'
echo
echo "--- 关键文件点检 ---"
for f in .gitignore src/model/dit.py src/model/injections.py src/model/__init__.py \
         src/eval/in_mem_eval.py src/eval/model_io.py src/utils/latent_dataset.py \
         src/train/train.py src/train/ckpt.py src/train/cli.py \
         src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
         src/train/configs/v51_B_joint_kv_k4_top10.json \
         tools/plot_curve.py _sync_work/launch_AB_v2.sh; do
  if tar -tzf "$TAR" | grep -qx "$f"; then printf '  ✓ %s\n' "$f"; else printf '  ✗ %s\n' "$f"; fi
done
echo
echo "--- 根级成员 (只应看到 .gitignore 之类) ---"
tar -tzf "$TAR" | grep -E '^[^/]+$' | head -10 | sed 's/^/  /'
echo
echo "--- 文件类型分布 ---"
tar -tzf "$TAR" | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10 | sed 's/^/  /'

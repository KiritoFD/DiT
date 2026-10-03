#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. 探针是否在跑 / 走了什么设备 ##########"
tmux ls 2>&1 | head -4
ps -eo pid,etime,pcpu,cmd | grep -E 'vae_space_probe|latent_signal' | grep -v grep | head -3
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
grep -aE '^\[(in|load|pixel|pairs|split)\]|^ *(SD-4ch|FLUX-16ch|pixel-64)|^A/B/C|^D\.|^判读' \
     exp-std/logs_purestd/vae_probe_latest.log 2>/dev/null | tail -30

echo
echo "########## 2. K=4 DINO 质心 / 风格 token 表在哪 ##########"
ls -la assets/ 2>/dev/null | grep -iE 'dino|kmeans|centroid|style|pool|k4|tok' | head -20
find assets data -maxdepth 3 -iname '*k4*' -o -maxdepth 3 -iname '*centroid*' -o -maxdepth 3 -iname '*kmeans*' 2>/dev/null | head -12

echo
echo "########## 3. v15 系列配置 ##########"
ls src/train/configs/ | grep -iE 'v15' | head -20

echo
echo "########## 4. 配置/代码里的风格注入口 ##########"
grep -rlnE 'dino_pool|style_tokens|dino_centroid|k4|style_cross|ip_adapter|style_enc' \
     src/train/configs/*.json src/model/*.py src/train/*.py 2>/dev/null | head -14
echo "--- 关键字段实际值 ---"
grep -rnE '"(dino_pool|n_style_tokens|style_token|dino_centroid_[a-z]*|multistyle_k|style_cross_attn|[a-z_]*ip_[a-z_]*)"' \
     src/train/configs/*.json 2>/dev/null | head -20

echo
echo "########## 5. v15 的实测成绩(ratio_style 等) ##########"
ls -d assets/results/v15* 2>/dev/null | head -20
grep -rn 'ratio_style' docs/ *.md 2>/dev/null | head -12

echo
echo "########## 6. 模型里有没有 cross-attn 注入口(现成) ##########"
grep -rnE 'class .*(StyleTok|IPAdapter|StyleCross|DinoPool)' src/model/*.py 2>/dev/null | head -10
grep -rn 'glyph_inject_mode' src/model/dit.py 2>/dev/null | head -12

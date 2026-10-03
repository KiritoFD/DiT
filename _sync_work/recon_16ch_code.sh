#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. 所有 decode 站点 (sf 用法; 16ch Flux 需要 + shift 0.1159) ##########"
grep -rnE '/ *sf|/sf|18215|scaling_factor|vae_shift' src/eval/*.py src/train/*.py \
     tools/build_eval_real200_cache.py 2>/dev/null | head -30

echo
echo "########## 2. save_input_g (poster 第 1 行) ##########"
grep -n 'def save_input_g' -A 26 src/eval/in_mem_eval.py 2>/dev/null | head -34

echo
echo "########## 3. K=4 质心构建器 ##########"
sed -n '1,45p' tools/build_multistyle_k4.py 2>/dev/null

echo
echo "########## 4. cli 里的相关开关 ##########"
grep -nE 'callig.style.ca|style.ctx.every.lyaer|callig.multi.style.k|latent.channels|vae.shift|vae.scaling|callig.emb.pretrained|freeze.callig' \
     src/train/cli.py 2>/dev/null | head -24

echo
echo "########## 5. v47 配置全文(作为新配置的基底) ##########"
cat src/train/configs/v47_purestd_xattn12_top10.json 2>/dev/null | head -80

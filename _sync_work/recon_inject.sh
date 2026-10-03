#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. v15_multistyle_k4.json (K=4 注入的正式配置) ##########"
cat src/train/configs/v15_multistyle_k4.json 2>/dev/null

echo
echo "########## 2. d5/d3 v15a 变体 ##########"
echo "--- d5_v15a_multistyle_k4.json"; cat assets/d5_v15a_multistyle_k4.json 2>/dev/null
echo "--- d3_v15a_multistyle_k4_any.json (前 1200 字)"; head -c 1200 assets/d3_v15a_multistyle_k4_any.json 2>/dev/null

echo
echo "########## 3. 模型/训练里多风格 tok 的注入实现 ##########"
grep -rnE 'multistyle|style_tok|n_style|dino_pool|dino_tok|k4' \
     src/model/*.py src/train/train.py src/utils/*.py src/eval/*.py 2>/dev/null | head -40

echo
echo "########## 4. v15a/b 实测数字 ##########"
for f in assets/t1_v15a_multistyle_k4__0150000.json assets/t1_v15b_multistyle_k4__0125000.json \
         assets/t2_v15a_multistyle_k4__0150000.json assets/t2_v15b_multistyle_k4__0125000.json \
         assets/t1_v13_styletok__0195000.json assets/t2_v13_styletok__0195000.json; do
  if [ -f "$f" ]; then echo "--- $f"; head -c 900 "$f"; echo; fi
done

echo
echo "########## 5. ink_eval 家底 ##########"
ls assets/ink_eval/ 2>/dev/null | head -20
echo "--- 逐样本 csv ---"
ls assets/ink_eval_raw/ 2>/dev/null | head -20
for f in assets/ink_eval_raw/v15a_multistyle_k4__seen.csv assets/ink_eval_raw/v15b_multistyle_k4__seen.csv; do
  if [ -f "$f" ]; then echo "--- $f"; head -2 "$f"; fi
done

echo
echo "########## 6. 条件/骨架 shard 家底 (换 16ch 要重编哪些) ##########"
for d in data/top10_style23/gt_skel_png_w7 data/top10_style23/std \
         exp-std/data/shards_gtskel_w7 exp-std/data/shards_std_w7 exp-std/data/shards_img_flux16 \
         exp-std/data/shards_std_flux16; do
  n=$(ls "$d" 2>/dev/null | wc -l)
  echo "  $d : $n 个文件"
done

echo
echo "########## 7. 当前 v47 配置的通道/模型字段 ##########"
grep -nE '"(in_channels|latent_channels|latent_shards_dir|image_size|model|patch_size|vae_[a-z_]*|glyph_inject[a-z_]*|use_char|no_char)' \
     src/train/configs/v47_purestd_xattn12_top10.json 2>/dev/null

echo
echo "########## 8. dino 特征缓存 ##########"
ls -la data/dino_cache/ 2>/dev/null | head
ls -la data/dino_cache/top10_v1/ 2>/dev/null | head -6

#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1

echo "########## 3. 多风格/CROSS-ATTN 注入的代码位置 ##########"
grep -rnE 'MultiStyleEmbedder|CalligStyleCrossAttn|callig_multi_style_k' \
     src/model/*.py src/train/train.py src/eval/*.py 2>/dev/null | head -24

echo
echo "########## 3b. 现有注入模式有哪些 ##########"
grep -rnE 'glyph_inject_mode|inject_mode ==|== "adaln"|== "xattn"' src/model/dit.py 2>/dev/null | head -14

echo
echo "########## 4. v15a/b 实测 ##########"
ls -la assets/ink_eval/ 2>/dev/null | head -14
for f in assets/ink_eval_raw/v15a_multistyle_k4__seen.csv assets/ink_eval_raw/v15a_multistyle_k4__strict.csv \
         assets/ink_eval_raw/v15b_multistyle_k4__seen.csv assets/ink_eval_raw/v15b_multistyle_k4__strict.csv; do
  if [ -f "$f" ]; then echo "--- $f"; head -2 "$f"; fi
done
echo "--- 所有 ink_eval_raw ---"; ls assets/ink_eval_raw/ 2>/dev/null | head -20

echo
echo "########## 5. ratio_style 记录 ##########"
grep -rn 'ratio_style' docs/ assets/*.json 2>/dev/null | head -10
ls assets/*style*.json 2>/dev/null | head

echo
echo "########## 6. 条件/骨架 shard 家底 ##########"
for d in data/top10_style23/gt_skel_png_w7 data/top10_style23/std \
         exp-std/data/shards_gtskel_w7 exp-std/data/shards_std_w7 \
         exp-std/data/shards_img_flux16 exp-std/data/shards_std_flux16; do
  echo "  $d : $(ls "$d" 2>/dev/null | wc -l) 个文件"
done
ls exp-std/data/shards_img_flux16/ 2>/dev/null | head -3

echo
echo "########## 7. v47 配置里的通道/模型字段 ##########"
grep -nE 'in_channels|latent_channels|latent_shards_dir|patch_size|"model"|vae_' \
     src/train/configs/v47_purestd_xattn12_top10.json 2>/dev/null
echo "--- base 模型注册表 ---"
grep -rn 'DiT_2Cond_models' src/model/__init__.py src/model/dit.py 2>/dev/null | head -6

echo
echo "########## 8. DINO 缓存 ##########"
ls -la data/dino_cache/ 2>/dev/null | head -8
ls data/dino_cache/top10_v1/ 2>/dev/null | head -4
echo -n "top10_v1 文件数 = "; ls data/dino_cache/top10_v1/ 2>/dev/null | wc -l

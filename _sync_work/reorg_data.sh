#!/bin/bash
# reorg_data.sh — 数据/脚本大整理 (幂等, 活跃数据不动)
cd /root/Workspace/xy/DiT || exit 1
mkdir -p data/archive/{legacy_latents,legacy_images,legacy_skeletons,legacy_plans,std_legacy,mid_data,results_legacy} \
         scripts/{scratch,legacy_py,legacy_sh,exp_configs} logs

mvq() { for f in $1; do [ -e "$f" ] && mv "$f" "$2/" 2>/dev/null; done; }

# ---- 1) 数据集归档 (活跃 fame v8 资产不动) ----
A=data/archive
mvq "final_latents data/latents/final_latents_f4 data/latents/final_latents_fame_clean data/latents/final_latents_mid_clean" $A/legacy_latents
mvq "data/imgs/final_imgs_256 data/imgs/final_imgs_256_clean data/imgs/final_imgs_256_clean_v2 data/imgs/final_imgs_256_v7backup data/imgs/final_imgs_mid_clean final_canny final_canny_d3 final_local_imgs.tar.gz" $A/legacy_images
mvq "data/skel/final_skeleton data/skel/final_skeleton_d3 data/skel/final_skel1 data/skel/final_skel3 data/skel/final_skel_latents_eval data/skel/final_skel_latents_eval_1px data/skel/final_skel_latents_eval_v2 data/skel/final_skel_latents_fame_v8 data/skel/final_skel_latents_mid_common data/skel/final_skel_latents_train_1px new_data_skel0 data/skel/std_skel data/skel/std_skel.tgz data/skel/std_skel1_latents_eval data/skel/std_skel1_latents_fame data/skel/std_skeleton_d3" $A/legacy_skeletons
mvq "final_img_plan.json final_latent_plan.json final_manifest_split.json final_id_maps.json final_eval.csv final_train.csv final_train_small.csv final_test.csv" $A/legacy_plans
mvq "std_glyph_latent std_glyph_latent.tgz std_gt std_gt_report.json std_vs_noise_dist.json" $A/std_legacy
mvq "new_data" $A/mid_data
[ -d results ] && [ ! -d $A/results_legacy/results ] && mv results $A/results_legacy/ 2>/dev/null

# ---- 2) 根目录脚本/日志归类 ----
mvq "*.log" logs
mvq "_chk*.py _check*.py _scan*.py _smoke*.py _verify*.py _diag*.py _probe*.py nohup.out" scripts/scratch
mvq "exp_*.json" scripts/exp_configs
ls *.sh 2>/dev/null | grep -vE "^(launch|run_)" | head -200 > /tmp/_sh_list
while read f; do [ -e "$f" ] && mv "$f" scripts/legacy_sh/ 2>/dev/null; done < /tmp/_sh_list
ls *.py 2>/dev/null | grep -vE "^(launch|run_)" > /tmp/_py_list
while read f; do [ -e "$f" ] && mv "$f" scripts/legacy_py/ 2>/dev/null; done < /tmp/_py_list
mvq "probe_*.json gen_*.json" scripts/exp_configs

# ---- 3) assets legacy csv 归档 (活跃 csv 留下) ----
mkdir -p data/archive/legacy_csv
for f in assets/*.csv; do
  b=$(basename "$f")
  case "$b" in
    train_fame_clean_v8.csv|eval_fame_strict_clean_v8.csv|master_results.csv) ;;
    *) mv "$f" data/archive/legacy_csv/ 2>/dev/null ;;
  esac
done
mvq "assets/*.json" data/archive/legacy_csv 2>/dev/null
for f in assets/*.json; do
  b=$(basename "$f")
  case "$b" in
    v89_eval_series.json|data_path.json) ;;
    *) mv "$f" data/archive/legacy_csv/ 2>/dev/null ;;
  esac
done

echo "=== 整理后根目录 ==="; ls -p | grep -v / | wc -l
echo "=== data/ ==="; du -sh data/ data/archive/* 2>/dev/null | head -12
echo "=== 活跃资产在位检查 ==="
for d in data/imgs/final_imgs_fame_v8 data/latents/final_latents_fame_v8 data/skel/final_skel_latents_fame_1px_v8 data/skel/final_skel1_fame_v8 data/skel/final_skel3_fame_v8 dataset pretrained_models; do
  [ -e "$d" ] && echo "OK $d" || echo "MISSING $d"
done
ls assets/train_fame_clean_v8.csv assets/eval_fame_strict_clean_v8.csv assets/master_results.csv >/dev/null 2>&1 && echo "OK 活跃csv"
REORG_DONE=1; echo REORG_COMPLETE

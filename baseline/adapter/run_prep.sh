#!/bin/bash
# Data preparation for all three baselines (idempotent).
# Run: bash /root/Workspace/xy/DiT/baseline/adapter/run_prep.sh
set -e
cd /root/Workspace/xy/DiT/baseline/adapter/common
PY=/opt/conda/envs/baseline/bin/python

echo "== 01 render content font (Deng.ttf, 256) =="
$PY 01_render_content.py
echo "== 02 FontDiffuser layout =="
$PY 02_prep_fontdiffuser.py
echo "== 03 VQ-Font LMDB + meta =="
$PY 03_prep_vqfont.py
echo "== 04 DG-Font layout =="
$PY 04_prep_dgfont.py
echo "== 05 eval GT + refs protocol =="
$PY 05_make_eval_pairs.py
echo "== 06 content-font latent shards (project vae_io) =="
$PY 06_prep_content_latents.py
echo "== 07 DG-Font patches (att_to_use, torchvision DCN, gates, compile hook) =="
$PY ../dgfont/patch_dgfont.py
echo "== 08 FontDiffuser patches (png dataset, StyleRSI pairing, SDPA) =="
$PY ../fontdiffuser/patch_fontdiffuser_256.py
$PY ../fontdiffuser/patch_sdpa.py
echo ALL_PREP_DONE

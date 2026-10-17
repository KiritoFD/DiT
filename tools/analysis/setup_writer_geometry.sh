#!/usr/bin/env bash
set -euo pipefail
base=/root/Workspace/xy/DiT
work="$base/scratch/writer_geometry_d2d3_20261016_01"
mkdir "$work"
cp -a "$base/scratch/writer_latent_d0d1_20261016_01/src" "$base/scratch/writer_latent_d0d1_20261016_01/tools" "$base/scratch/writer_latent_d0d1_20261016_01/tests" "$work/"
mkdir "$work/assets" "$work/scratch"
ln -s "$base/data" "$work/data"
ln -s "$base/exp-std" "$work/exp-std"
ln -s "$base/pretrained_models" "$work/pretrained_models"
ln -s "$base/assets/writer_latent_data_20261016" "$work/assets/data"
ln -s "$base/scratch/disc_lat_full.npz" "$work/scratch/disc_lat_full.npz"
tar -xzf "$base/scratch/writer_geometry_overlay_20261016.tar.gz" -C "$work"
cd "$work"
/opt/conda/envs/cu121/bin/python -m pytest -q tests/test_writer_style_geometry.py tests/test_writer_style_latent.py tests/test_latent_style_gradient.py
echo WRITER_GEOMETRY_SETUP_COMPLETE

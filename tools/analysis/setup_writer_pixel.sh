#!/usr/bin/env bash
set -euo pipefail
base=/root/Workspace/xy/DiT
work="$base/scratch/writer_pixel_d0d1_20261016_01"
mkdir "$work"
cp -a "$base/scratch/v89_teacher_v3_20261016_01/src" "$base/scratch/v89_teacher_v3_20261016_01/tools" "$base/scratch/v89_teacher_v3_20261016_01/tests" "$work/"
mkdir "$work/assets" "$work/scratch"
ln -s "$base/data" "$work/data"
ln -s "$base/exp-std" "$work/exp-std"
ln -s "$base/scratch/disc_lat_full.npz" "$work/scratch/disc_lat_full.npz"
tar -xzf "$base/scratch/writer_pixel_overlay_20261016.tar.gz" -C "$work"
cd "$work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python=/opt/conda/envs/cu121/bin/python
"$python" -m pytest -q tests/test_writer_style_pixel.py
"$python" tools/train_writer_style_pixel.py --prepare --out assets/data
"$python" -u tools/train_writer_style_pixel.py --data-root assets/data --out assets/benchmark --benchmark --arm D1 --steps 80 --batch 512
echo PIXEL_SETUP_AND_BENCHMARK_COMPLETE

#!/usr/bin/env bash
set -euo pipefail
base=/root/Workspace/xy/DiT
old="$base/scratch/v89_encoder_latest_20261016_01"
work="$base/scratch/v89_encoder_v2_20261016_01"
mkdir "$work"
cp -a "$old/src" "$old/tools" "$old/tests" "$work/"
mkdir "$work/assets" "$work/scratch"
ln -s "$base/data" "$work/data"
ln -s "$base/exp-std" "$work/exp-std"
ln -s "$base/scratch/disc_lat_full.npz" "$work/scratch/disc_lat_full.npz"
ln -s "$base/assets/writer_brush_critic.pt" "$work/assets/writer_brush_critic.pt"
tar -xzf "$base/scratch/v89_encoder_v2_overlay.tar.gz" -C "$work"
cd "$work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>"$base/logs/strong_style_queue.lock"
flock -n 9
/opt/conda/envs/cu121/bin/python -m pytest -q tests/test_prototype_style_encoder.py tests/test_v89_encoder_acceptance.py tests/test_style_encoder_local_stats.py
/opt/conda/envs/cu121/bin/python -u tools/train_prototype_style_encoder.py --arch local_stats --out assets/smoke --steps 4 --eval-every 4
/opt/conda/envs/cu121/bin/python -u tools/train_prototype_style_encoder.py --arch local_stats --out assets/encoder --steps 20000 --eval-every 1000
/opt/conda/envs/cu121/bin/python -u tools/analysis/accept_v89_encoder.py --run assets/encoder
echo V89_ENCODER_V2_COMPLETE

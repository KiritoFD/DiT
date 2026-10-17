#!/usr/bin/env bash
# Fully isolated source snapshot; existing remote checkout and weights untouched.
set -euo pipefail
base=/root/Workspace/xy/DiT
snapshot="$base/scratch/v89_encoder_latest_20261016_01"
mkdir "$snapshot"
mkdir "$snapshot/assets" "$snapshot/scratch"
tar -xzf "$base/scratch/v89_encoder_latest_source_20261016_01.tar.gz" -C "$snapshot"
ln -s "$base/data" "$snapshot/data"
ln -s "$base/exp-std" "$snapshot/exp-std"
ln -s "$base/scratch/disc_lat_full.npz" "$snapshot/scratch/disc_lat_full.npz"
ln -s "$base/assets/writer_brush_critic.pt" "$snapshot/assets/writer_brush_critic.pt"
cd "$snapshot"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum "$base/scratch/v89_encoder_latest_source_20261016_01.tar.gz" > source_archive.sha256
exec 9>"$base/logs/strong_style_queue.lock"
flock -n 9
/opt/conda/envs/cu121/bin/python -m pytest -q -o testpaths=tests tests/test_prototype_style_encoder.py
/opt/conda/envs/cu121/bin/python -u tools/train_prototype_style_encoder.py \
    --out assets/encoder --steps 20000 --batch 512 --lr .0001 --eval-every 1000
/opt/conda/envs/cu121/bin/python -u tools/analysis/accept_v89_encoder.py --run assets/encoder
echo V89_LATEST_FULL_EXPERIMENT_COMPLETE

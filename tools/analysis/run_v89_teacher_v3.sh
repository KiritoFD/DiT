#!/usr/bin/env bash
set -euo pipefail
base=/root/Workspace/xy/DiT
old="$base/scratch/v89_encoder_latest_20261016_01"
work="$base/scratch/v89_teacher_v3_20261016_01"
stats="$base/scratch/v89_encoder_v2_20261016_01/assets/encoder/ink_width_stats.npz"
test -f "$stats"
mkdir "$work"
cp -a "$old/src" "$old/tools" "$old/tests" "$work/"
mkdir "$work/assets" "$work/scratch"
ln -s "$base/data" "$work/data"
ln -s "$base/exp-std" "$work/exp-std"
ln -s "$base/scratch/disc_lat_full.npz" "$work/scratch/disc_lat_full.npz"
ln -s "$base/assets/writer_brush_critic.pt" "$work/assets/writer_brush_critic.pt"
tar -xzf "$base/scratch/v89_teacher_v3_overlay.tar.gz" -C "$work"
cd "$work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>"$base/logs/strong_style_queue.lock"
flock -n 9
/opt/conda/envs/cu121/bin/python -m pytest -q tests/test_prototype_style_encoder.py tests/test_style_encoder_local_stats.py tests/test_crosschar_supervision.py tests/test_teacher_hard_nce.py
/opt/conda/envs/cu121/bin/python -u tools/train_prototype_style_encoder.py --arch local_stats --hard-weight 1 --ink-width-stats "$stats" --out assets/smoke --steps 4 --eval-every 4
/opt/conda/envs/cu121/bin/python -u tools/train_prototype_style_encoder.py --arch local_stats --hard-weight 1 --ink-width-stats "$stats" --out assets/encoder --steps 20000 --eval-every 1000
/opt/conda/envs/cu121/bin/python -u tools/analysis/accept_v89_encoder.py --run assets/encoder
# Copy only image-derived statistics, NOT evaluation embeddings or target styles.
cp "$stats" assets/encoder/ink_width_stats.npz
/opt/conda/envs/cu121/bin/python -u tools/analysis/accept_teacher_matched.py --teacher-run assets/encoder --model-name v3
echo V89_TEACHER_V3_COMPLETE

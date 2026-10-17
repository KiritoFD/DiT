#!/usr/bin/env bash
set -euo pipefail
base=/root/Workspace/xy/DiT
cd "$base/scratch/writer_latent_d0d1_20261016_01"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>"$base/logs/strong_style_queue.lock"
flock -n 9
python=/opt/conda/envs/cu121/bin/python
"$python" -m pytest -q tests/test_writer_style_pixel.py tests/test_writer_style_latent.py
"$python" -u tools/analysis/check_writer_latent_bridge.py
for arm in D0 D1; do
  "$python" -u tools/train_writer_style_pixel.py --domain latent --data-root assets/data --out "assets/$arm" --arm "$arm" --batch 2048 --steps 10000 --eval-every 500
done
for arm in D0 D1; do
  "$python" -u tools/analysis/accept_writer_style_pixel.py --data-root assets/data --run "assets/$arm"
done
echo WRITER_LATENT_D0_D1_COMPLETE

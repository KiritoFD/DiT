#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
PY=${PYTHON_BIN:-/opt/conda/envs/cu121/bin/python}
mkdir -p logs
exec 9>logs/strong_style_queue.lock
flock -n 9 || { echo 'Another queue owns the lock'; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU occupied, refusing concurrent training'; exit 1
fi
"$PY" tools/analysis/prepare_full_followups.py
if [[ ! -f assets/style_eval_full_A100_20261015/DINO-S/style_report.json ||
      ! -f assets/style_eval_full_A100_20261015/CLIP-S/style_report.json ]]; then
echo "WAIT verified encoder-cache transfer $(date -Is)"
dino=/root/.cache/huggingface/hub/models--timm--vit_large_patch14_dinov2.lvd142m/snapshots/4741e1cafbf45415e77074bb0cb42dba76c8684a/model.safetensors
clip=/root/.cache/huggingface/hub/models--timm--vit_large_patch14_clip_224.openai/snapshots/18d0535469bb561bf468d76c1d73aa35156c922b/pytorch_model.bin
until [[ -f "$dino" && -f "$clip" ]] && \
      [[ $(stat -c%s "$dino") == 1217502758 && $(stat -c%s "$clip") == 1710633285 ]]; do sleep 10; done
sha256sum "$dino" "$clip"
echo "EVAL START $(date -Is)"
root=assets/results/v86_A100_full_fixed/20261015-034828-v86-A100-full-fixed/eval_samples_ctrl
"$PY" tools/v85_compare.py \
  --set v84A200k=assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix \
  --set A100_5k="$root/step0005000/eval200fix" \
  --set A100_10k="$root/step0010000/eval200fix" \
  --set A100_15k="$root/step0015000/eval200fix" \
  --set A100_20k="$root/step0020000/eval200fix" \
  --max-ref 3000 --bsd-ref-max 1000 --out assets/style_eval_full_A100_20261015
fi
# The historical wrapper can return zero after a failed subprocess. Require both reports.
"$PY" -c 'import json; from pathlib import Path; root=Path("assets/style_eval_full_A100_20261015"); names={"v84A200k","A100_5k","A100_10k","A100_15k","A100_20k"}; [(lambda d: (names <= set(d["sets"]) or (_ for _ in ()).throw(RuntimeError("incomplete evaluation"))))(json.loads((root/enc/"style_report.json").read_text())) for enc in ("DINO-S","CLIP-S")]'
echo "EVAL DONE $(date -Is)"
arms=("$@")
if ((${#arms[@]} == 0)); then arms=(B50 C D); fi
for arm in "${arms[@]}"; do
  case "$arm" in B25|B50|C|D) ;; *) echo "Unknown arm $arm"; exit 1;; esac
  log="logs/v86_full_fixed_10k_${arm}_$(date +%Y%m%d-%H%M%S).log"
  echo "START $arm $(date -Is) $log"
  MASTER_PORT=12791 "$PY" -m src.train.train --config "src/train/configs/v86_full_fixed_${arm}.json" --num-workers 4 > "$log" 2>&1
  steps=10000
  grep -q "Reached max_steps=$steps" "$log"
  echo "DONE $arm $(date -Is)"
done

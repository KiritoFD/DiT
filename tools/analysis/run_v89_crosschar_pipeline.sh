#!/usr/bin/env bash
# Run only after a teacher passes the preregistered discrimination gate.
set -euo pipefail
base=/root/Workspace/xy/DiT
work="$base/scratch/v89_crosschar_20261016_01"
teacher="$base/scratch/v89_teacher_v3_20261016_01/assets/encoder"
plan="$base/assets/v89_crosschar_pilot_20261016_01"
cd "$work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
while [[ ! -s "$teacher/matched_teacher.json" ]]; do
  if ! pgrep -f '[r]un_v89_teacher_v3.sh' > /dev/null; then
    echo 'TEACHER_JOB_ENDED_WITHOUT_MATCHED_ACCEPTANCE'; exit 2
  fi
  sleep 30
done
exec 9>"$base/logs/strong_style_queue.lock"
flock 9
py=/opt/conda/envs/cu121/bin/python
"$py" -m pytest -q tests/test_crosschar_supervision.py
"$py" tools/analysis/prepare_v89_crosschar_pilot.py --teacher-run "$teacher" --out "$plan"
"$py" tools/analysis/build_crosschar_bank.py --teacher-run "$teacher" --out "$plan/train_features.pt"
# Verify actual full-batch auxiliary gradient and memory before spending 10k.
"$py" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.loads((p/"crosschar.json").read_text()); c.update(max_steps=4,epochs=4,ckpt_every=4,epoch_steps=4,in_mem_eval=False,log_every=1,compile=False,experiment_name="v89-crosschar-smoke",results_dir="assets/results/v89_crosschar_smoke"); (p/"smoke.json").write_text(json.dumps(c))' "$plan"
MASTER_PORT=12891 "$py" -u -m src.train.train --config "$plan/smoke.json" > "$plan/smoke.log" 2>&1
MASTER_PORT=12892 "$py" -u -m src.train.train --config "$plan/control.json" > "$plan/control.log" 2>&1
MASTER_PORT=12893 "$py" -u -m src.train.train --config "$plan/crosschar.json" > "$plan/crosschar.log" 2>&1
"$py" -u tools/analysis/eval_v89_crosschar_pilot.py --plan "$plan" > "$plan/evaluation.log" 2>&1
echo V89_ID_ONLY_CROSSCHAR_PIPELINE_COMPLETE

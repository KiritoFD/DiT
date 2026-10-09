#!/bin/bash
# smoke_v73_batch.sh -- probe max feasible micro-batch for REPA-E on a 24G card.
# NOTE: the DiT trainer has NO gradient accumulation, so batch must fit outright.
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
CFG=src/train/configs/v73_repae_calli_joint.json
OUT=logs/v73_batch_smoke
mkdir -p "$OUT"

echo "=================================================================="
echo "REPA-E (v73) 显存/步速冒烟：找 24G 上能放下的最大 batch"
echo "  配置: $CFG   (latent_shards_dir 已清空 -> 在线上编码)"
echo "  每档 30 步，不写 ckpt、不做评测"
echo "=================================================================="
printf '%-6s %-7s %-10s %-9s %-6s %s\n' batch steps peak_mem steps_per_s rc note

for B in 32 64 96 128; do
  L="$OUT/b${B}.log"
  rm -rf "/tmp/_smoke_v73_b${B}"
  timeout 1200 env PYTHONPATH=. "$PY" -u -m src.train.train --config "$CFG" \
      --global-batch-size "$B" --max-steps 30 --ckpt-every 100000 \
      --in-mem-eval false --results-dir "/tmp/_smoke_v73_b${B}" > "$L" 2>&1
  rc=$?
  peak=$(grep -a -o 'Mem: [0-9.]*G' "$L" | sed 's/Mem: //' | tr -d 'G' | sort -n | tail -1)
  sps=$(grep -a -o 'Steps/Sec: [0-9.]*' "$L" | tail -1 | sed 's|Steps/Sec: ||')
  nstep=$(grep -ac '(step=' "$L")
  note=""
  if [ "$rc" -ne 0 ]; then
    if grep -qa 'out of memory' "$L"; then note="OOM"; else note="RC=$rc"; fi
  fi
  printf '%-6s %-7s %-10s %-9s %-6s %s\n' "$B" "$nstep" "${peak:-?}" "${sps:-?}" "$rc" "$note"
  if [ -n "$note" ]; then
    echo "     ---- tail ----"
    tail -6 "$L" | sed 's/^/     /'
  fi
  sleep 5
done

echo
echo "===== 结论（人工读）====="
echo "  取 peak_mem 明显低于 24G（建议留 1.5G 余量）且 rc=0 的最大 batch 作为 v73 的 global_batch_size"
echo "  日志目录: $OUT"

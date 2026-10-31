#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
for t in abl_none abl_off0 abl_off1 abl_off01; do
  f=assets/f_20261101/eval/t3_$t/t3_report.json
  if [ -f "$f" ]; then
    echo "=== $t"
    /opt/conda/envs/cu121/bin/python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print('  grid   ', d.get('grid'))
print('  ocr    ', d.get('ocr'))
print('  direction', d.get('direction'), ' amplitude', d.get('amplitude'),
      ' n_pairs', d.get('n_pairs'))
PY
  else
    echo "=== $t  MISSING"
  fi
done
echo
echo "=== K_abl_off01 content eval ==="
f=assets/f_20261101/eval/K_abl_off01/eval.json
[ -f "$f" ] && /opt/conda/envs/cu121/bin/python -c "
import json;d=json.load(open('$f'));print(d['content']);print('ocr',d['ocr']);print('ink',d['ink'])"
echo
echo "=== s1 log tail ==="
tail -5 assets/f_20261101/logs/f_driver.log

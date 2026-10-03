#!/bin/bash
# 盘: 每个 run 有没有记录指标曲线(轨迹); seen20 覆盖哪些槽位; train.csv 是否含这些 seen id
cd /root/Workspace/xy/DiT || exit 1

echo "########## A. 一个 run 目录完整内容 ##########"
run=$(find assets/results -maxdepth 2 -name 'checkpoints' -type d 2>/dev/null | head -1 | xargs -r dirname)
echo "run = $run"
ls -la "$run" 2>/dev/null | head -20
echo "--- checkpoints 里除 .pt 还有什么 ---"
ls -la "$run/checkpoints" 2>/dev/null | head -12

echo
echo "########## B. 所有 run 里非 .pt 文件的类型统计 ##########"
find assets/results _archive/20261003_twostage -maxdepth 3 -type f ! -name '*.pt' 2>/dev/null \
  | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10

echo
echo "########## C. 疑似指标/曲线文件 ##########"
find assets/results _archive/20261003_twostage -maxdepth 3 -type f \
  \( -name '*metric*' -o -name '*summary*' -o -name '*hist*' -o -name '*.log' -o -name '*curve*' \) 2>/dev/null | head -20
echo "--- logs/ ---"
ls logs/ 2>/dev/null | head -40

echo
echo "########## D. seen20 slotmap 覆盖的槽位 ##########"
echo "槽位计数:"; cut -d, -f13 assets/eval_top10_seen_20_slotmap.csv 2>/dev/null | tail -n +2 | sort | uniq -c
echo "slot 总数 = $(cut -d, -f13 assets/eval_top10_seen_20_slotmap.csv | tail -n +2 | sort -u | wc -l)"

echo
echo "########## E. 23 槽位表全名 ##########"
/opt/conda/envs/cu121/bin/python - <<'PY'
import json
m = json.load(open("assets/callig_script_id_map_top10.json", encoding="utf-8"))
pm = m.get("pair_map", m)
print("keys:", list(m.keys()))
print("槽位数 =", len(pm))
for k, v in list(pm.items())[:25]:
    print("   ", k, "->", v)
PY

echo
echo "########## F. seen20 的 img_id 是否都在 train.csv ##########"
/opt/conda/envs/cu121/bin/python - <<'PY'
import csv
tr = {r["img_id"] for r in csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8"))}
ev = {r["img_id"] for r in csv.DictReader(open("exp-std/csv/eval200.csv", encoding="utf-8"))}
for name in ("assets/eval_top10_seen_20_slotmap.csv", "assets/eval_top10_seen_20.csv"):
    try:
        s = list(csv.DictReader(open(name, encoding="utf-8")))
    except Exception as e:
        print(name, "读取失败", e); continue
    ids = {r["img_id"] for r in s}
    print(f"{name}: n={len(s)}  在train={len(ids & tr)}/{len(ids)}  与eval200重叠={len(ids & ev)}")
print(f"train={len(tr)}  eval200={len(ev)}  交集={len(tr & ev)}")
PY

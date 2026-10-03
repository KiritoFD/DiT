#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. DINO 缓存格式 ##########"
ls -la data/dino_cache/ 2>/dev/null | head
echo "--- top10_v1 ---"
ls -la data/dino_cache/top10_v1/ 2>/dev/null | head -8
echo -n "  文件数 = "; ls data/dino_cache/top10_v1/ 2>/dev/null | wc -l
/opt/conda/envs/cu121/bin/python - <<'PY'
import glob, os, numpy as np, torch as th
d = "data/dino_cache/top10_v1"
fs = sorted(glob.glob(os.path.join(d, "*")))
print("  样例文件:", [os.path.basename(f) for f in fs[:5]])
if fs:
    f = fs[0]
    try:
        if f.endswith(".npz"):
            z = np.load(f)
            print("  npz keys:", {k: (z[k].shape, z[k].dtype) for k in z.files})
        elif f.endswith((".pt", ".pth")):
            o = th.load(f, map_location="cpu", weights_only=False)
            if isinstance(o, dict):
                print("  pt dict keys:", list(o.keys())[:8])
                for k in list(o.keys())[:3]:
                    v = o[k]
                    print(f"    {k}: {tuple(v.shape) if hasattr(v,'shape') else type(v)}")
            else:
                print("  pt 类型:", type(o), getattr(o, "shape", ""))
    except Exception as e:
        print("  读取失败:", e)
PY

echo
echo "########## 2. 本地 DINOv2 权重 ##########"
ls ~/.cache/huggingface/hub/ 2>/dev/null | grep -i dino
ls -la ~/.cache/torch/hub/checkpoints/ 2>/dev/null | head -5

echo
echo "########## 3. 23 槽位 pair_map 格式 vs 87 pair 格式 ##########"
/opt/conda/envs/cu121/bin/python - <<'PY'
import json
for p in ("exp-std/csv/callig_script_id_map_top10.json", "assets/callig_script_id_map.json"):
    try:
        d = json.load(open(p, encoding="utf-8"))
        print(f"--- {p}: keys={list(d.keys())}")
        for k in ("num_pairs", "num_calligraphers", "pair_map", "callig_map"):
            if k in d:
                v = d[k]
                print(f"    {k}: " + (f"{len(v)} 项, 前3: {list(v.items())[:3]}" if isinstance(v, dict) else str(v)))
    except Exception as e:
        print(p, "读取失败", e)
PY

echo
echo "########## 4. build_multistyle_k4 的入参 ##########"
grep -nE 'add_argument' tools/build_multistyle_k4.py 2>/dev/null | head -12
echo "--- 它怎么用 npz/map ---"
grep -nE 'npz|id_map|pair_map|callig_script_map|np\.load' tools/build_multistyle_k4.py 2>/dev/null | head -14

echo
echo "########## 5. 16ch 骨架编码是否完成 ##########"
tail -6 exp-std/logs_purestd/encode16_rest_latest.log 2>/dev/null
for d in exp-std/data/shards_gtskel_flux16 exp-std/data/shards_std_flux16_fixed_eval200; do
  echo "  $d : $(ls "$d" 2>/dev/null | wc -l) 文件"
done

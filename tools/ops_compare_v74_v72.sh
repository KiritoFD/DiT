#!/usr/bin/env bash
# ops_compare_v74_v72.sh -- v72@120k vs v74@120k：判�?是不是采�?目标)的问�?
#
#   判据两把尺（都在同一�?40 字上�?
#     * in-mem 曲线（两者同口径 cfg1.0/γ0，训练时自动落盘）—�?�?SSIM/LPIPS/frag
#     * census_ckpt（高频×GT、逐字墨量回归斜率/r）—�?�?�?�?均值化"有没有改�?#
# 用法:
#   bash scratch/ops_compare_v74_v72.sh            # 只读曲线（不占卡�?#   STOP=1 bash scratch/ops_compare_v74_v72.sh     # 先停 v74 再跑普查（普查需�?GPU�?set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
V72_CFG=src/train/configs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot.json
V74_CFG=src/train/configs/v74_callivae_mode_sp_c2ot.json
VAE=$PWD/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv
V72_CK=$(ls assets/runs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/*/checkpoints/0085000.pt 2>/dev/null | head -1)
V74_CK=$(ls assets/runs/v74_callivae_mode_sp_c2ot/*/checkpoints/0085000.pt 2>/dev/null | head -1)

echo "############ A. �?step �?in-mem 曲线（cfg1.0/γ0，直接可比） ############"
$PY - <<'PYEOF'
import glob, json, os, re
def curve(pat, label):
    fs = glob.glob(pat)
    if not fs:
        print(f"  {label}: (�?"); return {}
    d = {}
    for p in fs:
        try: j = json.load(open(p, encoding="utf-8"))
        except Exception: continue
        if j.get("set") == "eval200fix":
            d[j["step"]] = (j.get("ssim"), j.get("lpips"), j.get("frag"), j.get("ink_iou"))
    return d
a = curve("assets/runs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/*/checkpoints/eval_auto_*.json", "v72")
b = curve("assets/runs/v74_callivae_mode_sp_c2ot/*/checkpoints/eval_auto_*.json", "v74")
for lab, d in (("v72 (sample 目标)", a), ("v74 (mode 目标)", b)):
    print(f"\n  --- {lab} ---")
    if not d: print("    (无评测点)"); continue
    print(f"    {'step':>7} {'ssim':>8} {'lpips':>8} {'frag':>7} {'ink_iou':>8}")
    for s in sorted(d):
        v = d[s]
        print(f"    {s:>7} " + " ".join(f"{(x if x is not None else 0):>8.4f}" for x in v))
print("\n  --- �?step 差�?(v74 - v72) ---")
common = sorted(set(a) & set(b))
if not common:
    print("    (还没有共同评测点)")
for s in common:
    d0, d1 = a[s], b[s]
    print(f"    {s:>7}  dSSIM={d1[0]-d0[0]:+.4f}  dLPIPS={d1[1]-d0[1]:+.4f}  dFrag={d1[2]-d0[2]:+.3f}  dInk={d1[3]-d0[3]:+.4f}")
PYEOF

echo
echo "############ B. v74 进度 ############"
L=$(ls -t logs/v74_train_*.log | head -1)
tail -1 "$L" | tr -d '\033' | sed 's/\[[0-9;]*m//g' | sed 's/^/  /'
echo "  gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

if [ "${STOP:-0}" = "1" ]; then
  echo
  echo "############ C. �?v74（用 120k ckpt 做普查；需�?GPU�?############"
  [ -n "$V74_CK" ] || { echo "  �?找不�?v74 �?0085000.pt，先不动"; exit 1; }
  tmux kill-session -t v74 2>/dev/null && echo "  tmux v74 killed"
  pkill -f 'src.train.train --config .*v74_callivae_mode' 2>/dev/null && echo "  pkill sent"
  sleep 8
  echo "  gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

  echo
  echo "############ D. 两个 ckpt 的普查（高频 + 墨量斜率�?############"
  for pair in "v72:$V72_CK:$V72_CFG" "v74:$V74_CK:$V74_CFG"; do
    tag=${pair%%:*}; rest=${pair#*:}; ck=${rest%%:*}; cfg=${rest#*:}
    echo "  === $tag  $ck"
    PYTHONPATH=. $PY -u tools/census_ckpt.py --ckpt "$ck" --config "$cfg" \
      --vae-path "$VAE" --step 85000 --n 40 --cfg 0.7 --gamma 0.0 \
      --out "experiments/census_${tag}_120k" 2>&1 | tail -18 | sed 's/^/    /'
  done
  echo
  echo "  判读: HF×GT 升（0.51~0.67 �??）、ink 斜率升（0.55 �??）、r 升（0.53 �??�?
  echo "        若都改善 -> 确认�?采样/目标'的问题（mode 目标治均值化�? 若不�?-> 是容�?损失层面"
else
  echo
  echo "  (只跑了只读曲线。普查需�?GPU：用 STOP=1 再跑一�?"
fi

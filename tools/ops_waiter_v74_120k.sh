#!/usr/bin/env bash
# waiter_120k.sh -- 等 v74 到 120k，然后自动完成对照（不改训练，只在到点后停它）
#   1) 等到 <v74>/checkpoints/0120000.pt 出现（+ 等评测落盘）
#   2) 只读曲线对比: v74 vs v72 同 step（两者同口径 cfg1.0/γ0）-> REPORT.md
#   3) 停 v74（先确认 ckpt 在位）
#   4) 对 v72@120k 与 v74@120k 各跑 census_ckpt.py（cfg0.7/γ0, N=40）
#   5) 写 DONE 标记
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
V72_RUN=$(ls -d assets/runs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/*/ | head -1)
V74_RUN=$(ls -d assets/runs/v74_callivae_mode_sp_c2ot/*/ | head -1)
TARGET=${V74_RUN}checkpoints/0120000.pt
OUT=experiments/compare_120k
REPORT=$OUT/REPORT.md
mkdir -p "$OUT"

echo "[waiter] 开始守候 $TARGET  $(date '+%F %T')" | tee -a "$OUT/waiter.log"
# ★ 必须等 ckpt **写完**：torch.save 是边写边出现的（文件先出现、后变大）。
#   判据 = 存在 && 大小 >900MB && 30 秒内大小不变（否则会停在一个写了一半的 ckpt 上）
while true; do
  if [ -f "$TARGET" ]; then
    s1=$(stat -c%s "$TARGET" 2>/dev/null || echo 0)
    sleep 30
    s2=$(stat -c%s "$TARGET" 2>/dev/null || echo 0)
    if [ "$s1" = "$s2" ] && [ "$s1" -gt 900000000 ]; then
      echo "[waiter] ckpt 已写完: ${s2} bytes" >> "$OUT/waiter.log"
      break
    fi
    echo "[waiter] $(date '+%H:%M') ckpt 仍在写 (${s1} -> ${s2})" >> "$OUT/waiter.log"
  else
    sleep 240
    echo "[waiter] $(date '+%H:%M') 仍在等 … 当前 $(ls -1 ${V74_RUN}checkpoints/*.pt 2>/dev/null | wc -l) 个 ckpt" >> "$OUT/waiter.log"
  fi
done
echo "[waiter] 再等 90s 让 in-mem 评测落盘  $(date '+%F %T')" | tee -a "$OUT/waiter.log"
sleep 90

# ---------- 2. 曲线对比（只读） ----------
{
echo "# v72 vs v74 @120k 对照（max=200k 同调度，唯一差异 = 编码目标 sample vs mode）"
echo
echo "生成时间: $(date '+%F %T')"
echo
echo "## A. 同口径 in-mem 曲线（cfg1.0/γ0, N=187, heun-50）"
echo
$PY - <<'PYEOF'
import glob, json
def curve(pat):
    d = {}
    for p in glob.glob(pat):
        try: j = json.load(open(p, encoding="utf-8"))
        except Exception: continue
        if j.get("set") == "eval200fix":
            d[j["step"]] = (j.get("ssim"), j.get("lpips"), j.get("frag"), j.get("ink_iou"))
    return d
A = curve("assets/runs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/*/checkpoints/eval_auto_*.json")
B = curve("assets/runs/v74_callivae_mode_sp_c2ot/*/checkpoints/eval_auto_*.json")
for lab, d in (("v72 (sample 目标)", A), ("v74 (mode 目标)", B)):
    if not d:
        print(f"### {lab}\n(无评测点)\n"); continue
    print(f"### {lab}\n")
    print("| step | ssim | lpips | frag | ink_iou |")
    print("|---|---|---|---|---|")
    for s in sorted(d):
        v = d[s]
        print("| {} | {:.4f} | {:.4f} | {:.3f} | {:.4f} |".format(s, *(x if x is not None else 0 for x in v)))
    print()
print("### 同 step 差值 (v74 - v72)\n")
print("| step | dSSIM | dLPIPS | dFrag | dInk_iou |")
print("|---|---|---|---|---|")
for s in sorted(set(A) & set(B)):
    a, b = A[s], B[s]
    print("| {} | {:+.4f} | {:+.4f} | {:+.3f} | {:+.4f} |".format(
        s, b[0]-a[0], b[1]-a[1], b[2]-a[2], b[3]-a[3]))
PYEOF
echo
} > "$REPORT" 2>&1

# ---------- 3. 停 v74 ----------
{
echo
echo "## B. 停 v74（已确认 120k ckpt 在位）"
tmux kill-session -t v74 2>/dev/null && echo "- tmux v74 killed"
pkill -f 'config src/train/configs/v74_callivae_mode_sp_c2ot.json' 2>/dev/null && echo "- pkill sent"
pkill -f 'v74_callivae_mode_sp_c2ot' 2>/dev/null && echo "- pkill(2) sent"
for i in 1 2 3 4 5 6; do
  sleep 10
  free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -dc '0-9')
  echo "- ${i}0s 后 GPU used = ${free} MiB"
  [ "${free:-9999}" -lt 3000 ] && break
done
echo
} >> "$REPORT" 2>&1

# ---------- 4. 两 ckpt 普查 ----------
{
echo "## C. 两 ckpt 普查（cfg0.7/γ0, N=40, 同一批字）"
echo
echo "判读: 高频×GT 越大越好（v72@130k 参照 0.51~0.67）；"
echo "      ink_slope 越接近 1、ink_r 越高 = 墨量越跟着字形走（v72@130k 参照 斜率 0.55 / r 0.53）"
echo
for tag in v72 v74; do
  if [ "$tag" = "v72" ]; then CK=${V72_RUN}checkpoints/0120000.pt; CFG=src/train/configs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot.json
  else CK=${V74_RUN}checkpoints/0120000.pt; CFG=src/train/configs/v74_callivae_mode_sp_c2ot.json; fi
  echo "### $tag  ($CK)"
  echo '```'
  PYTHONPATH=. $PY -u tools/census_ckpt.py --ckpt "$CK" --config "$CFG" \
    --vae-path "$PWD/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv" \
    --step 120000 --n 40 --cfg 0.7 --gamma 0.0 --out "experiments/census_${tag}_120k" 2>&1 | tail -20
  echo '```'
  echo
done
} >> "$REPORT" 2>&1

date '+%F %T' > "$OUT/DONE"
echo "[waiter] 完成 $(date '+%F %T')" | tee -a "$OUT/waiter.log"
tail -40 "$REPORT"

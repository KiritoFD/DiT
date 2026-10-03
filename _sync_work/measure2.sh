#!/usr/bin/env bash
# 稳态归因 v2:
#  [1] 等过编译期, 采稳态功率/利用率
#  [2] 读实时 Steps/Sec
#  [3] 从旧 v47 日志**量出 inline eval 的阻塞时长** (跨 2500 步落点的时间差 vs 正常步)
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python

echo "########## [1] 等 210 秒过编译, 再采 5 次稳态 ##########"
sleep 210
for i in 1 2 3 4 5; do
  nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,clocks.current.sm --format=csv,noheader
  sleep 3
done

echo
echo "########## [2] 实时 Steps/Sec (A 路) ##########"
LOG=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
echo "log=$LOG"
grep -aE 'Steps/Sec|Step/s' "$LOG" | tail -6
echo "--- 最后 4 行 ---"
tail -4 "$LOG"

echo
echo "########## [3] inline eval 阻塞时长 (从旧 v47 日志实测) ##########"
$PY - <<'PY'
import glob, re, statistics
cands = sorted(glob.glob("exp-std/logs_purestd/stage1_p1.0_*.log"))
if not cands:
    print("(没找到旧日志)"); raise SystemExit
p = cands[-1]
print("log =", p)
rx = re.compile(r"\[(?:\x1b\[34m)?(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\x1b\[0m)?\] \(step=(\d+)\)")
rows = []
for ln in open(p, encoding="utf-8", errors="ignore"):
    m = rx.search(ln)
    if m:
        from datetime import datetime
        rows.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2))))
print(f"  解析到 {len(rows)} 条 step 记录, step 范围 {rows[0][1]}..{rows[-1][1]}")
dts = []
for (t0, s0), (t1, s1) in zip(rows, rows[1:]):
    dts.append(((t1 - t0).total_seconds(), s1 - s0, s1))
norm = [d for d, ds, s in dts if ds == 50 and s % 2500 != 0 and (s - 50) % 2500 != 0]
big = [(d, ds, s) for d, ds, s in dts if d > 5 * (statistics.median(norm) if norm else 1)]
if norm:
    med = statistics.median(norm)
    print(f"  正常 50 步耗时: 中位 {med:.1f}s  (≈ {50/med:.2f} steps/s)")
print("  ⚠ 明显停顿 (>5x 正常):")
tot = 0.0
for d, ds, s in sorted(big, key=lambda z: -z[0])[:12]:
    print(f"    step->{s}: 间隔 {d:.1f}s  (含 {ds} 步)")
    tot += d
print(f"  大停顿合计(前 12 个) = {tot:.0f}s")
allbig = sum(d for d, ds, s in big)
print(f"  ★ 所有大停顿合计 = {allbig:.0f}s = {allbig/60:.1f} 分钟")
if rows:
    span = (rows[-1][0] - rows[0][0]).total_seconds()
    print(f"  日志总跨度 {span/60:.1f} 分钟 / {rows[-1][1]-rows[0][1]} 步")
    print(f"  ★ 估算: 若全是 eval 阻塞, 它占总时间的 {100*allbig/max(span,1):.1f}%")
PY

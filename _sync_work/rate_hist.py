"""算 xattn 训练每 50 步的间隔，定位有没有掉速窗口。"""
import os
import re
import glob

os.chdir("/root/Workspace/xy/DiT")
L = sorted(glob.glob("logs/v12_series/v12_xattn/train_*.log"),
           key=os.path.getmtime)[-1]
pat = re.compile(r"^\[2026-09-17 (\d\d:\d\d:\d\d)\].*\(step=(\d+)\).*Steps/Sec: ([\d.]+)")
rows = []
for line in open(L, encoding="utf-8", errors="replace"):
    m = pat.match(line)
    if m:
        h, mi, s = map(int, m.group(1).split(":"))
        rows.append((h * 3600 + mi * 60 + s, int(m.group(2)), float(m.group(3))))

print(f"{'时间':>10} {'step':>8} {'秒/50步':>9} {'Steps/Sec':>10}")
prev = None
worst = (0, None)
for t, st, rate in rows:
    d = (t - prev) if prev is not None else 0
    prev = t
    if d > worst[0]:
        worst = (d, (t, st))
    if st % 500 == 0 or d > 12:            # 每 500 步 + 所有异常慢的
        hh, rem = divmod(t, 3600)
        mm, ss = divmod(rem, 60)
        flag = "  <<< 慢" if d > 12 else ""
        print(f"{hh:02d}:{mm:02d}:{ss:02d} {st:>8} {d:>9} {rate:>10}{flag}")

print()
print(f"最慢一次: {worst[0]} 秒 (step {worst[1][1] if worst[1] else '-'})")
rates = [r for _, _, r in rows]
print(f"Steps/Sec: min={min(rates):.2f} med={sorted(rates)[len(rates)//2]:.2f} "
      f"max={max(rates):.2f}")

# -*- coding: utf-8 -*-
"""量化 v12 日志里 "epoch 边界" 的真实代价。

只做测量, 不改任何东西。产出:
  1) 每 50 步窗口的 dt, 标记窗口内是否含 "Beginning epoch"
  2) 边界窗口 vs 普通窗口的 dt 分布 (排除 in_mem_eval 的大间隔)
  3) 若把 epoch 拉长到 N 步, 预计省多少时间
"""
import io
import re
import sys
import statistics as st

LOG = sys.argv[1] if len(sys.argv) > 1 else r"G:\GitHub\DiT\_review\v12_train.log"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TSTEP = re.compile(r"\[.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\]\s*\(step=(\d+)\)")
TEPOCH = re.compile(r"\[.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\]\s*Beginning epoch (\d+)")


def ts(s):
    h, m, sec = s.split(" ")[1].split(":")
    d = int(s.split(" ")[0].split("-")[2])
    return ((d * 24 + int(h)) * 60 + int(m)) * 60 + int(sec)


steps, epochs, raw = [], [], []
for line in io.open(LOG, encoding="utf-8", errors="replace"):
    line = ANSI.sub("", line)
    m = TSTEP.search(line)
    if m:
        steps.append((ts(m.group(1)), int(m.group(2))))
        continue
    m = TEPOCH.search(line)
    if m:
        epochs.append((ts(m.group(1)), int(m.group(2))))
        continue

print(f"log={LOG}")
print(f"step 记录 {len(steps)} 条, epoch 记录 {len(epochs)} 条")
if not steps:
    sys.exit("no step lines")

print(f"步数范围 {steps[0][1]} .. {steps[-1][1]}  "
      f"墙钟 {steps[-1][0] - steps[0][0]} s")
print(f"epoch 范围 {epochs[0][1]} .. {epochs[-1][1]}")
print(f"→ 每 epoch 步数 = {(steps[-1][1]-steps[0][1])/max(1,(epochs[-1][1]-epochs[0][1])):.1f}")
print()

# ---- 窗口: 相邻两条 step 记录 ----
bnd = []   # (dt, dstep, n_epoch, tag)
norm = []
for i in range(1, len(steps)):
    t0, s0 = steps[i - 1]
    t1, s1 = steps[i]
    dt, ds = t1 - t0, s1 - s0
    n_ep = sum(1 for (te, _) in epochs if t0 < te <= t1)
    tag = "bnd" if n_ep else "norm"
    rec = (dt, ds, n_ep)
    if ds == 50 and dt <= 25:          # 排除 in_mem_eval 等大间隔
        (bnd if n_ep else norm).append(rec)

print(f"干净窗口: 含 epoch 边界 {len(bnd)} 个 / 不含 {len(norm)} 个")
if bnd and norm:
    mb = st.mean(x[0] for x in bnd)
    mn = st.mean(x[0] for x in norm)
    print(f"  含边界 dt: mean={mb:.2f}s median={st.median(x[0] for x in bnd):.2f}s "
          f"min={min(x[0] for x in bnd):.2f} max={max(x[0] for x in bnd):.2f}")
    print(f"  不含边界 dt: mean={mn:.2f}s median={st.median(x[0] for x in norm):.2f}s "
          f"min={min(x[0] for x in norm):.2f} max={max(x[0] for x in norm):.2f}")
    print(f"  每个含边界窗口多花 = {mb - mn:+.3f} s / 50 步  "
          f"({(mb-mn)/50*1000:+.1f} ms/step)")
    # 统计显著: 简单 t 检验
    import math
    nb, nn = len(bnd), len(norm)
    vb = st.variance(x[0] for x in bnd)
    vn = st.variance(x[0] for x in norm)
    se = math.sqrt(vb / nb + vn / nn)
    print(f"  Welch t = {(mb-mn)/se:+.2f}  (SE={se:.3f}s)  "
          f"{'显著' if abs((mb-mn)/se) > 2 else '不显著'}")
print()

# ---- 每个 epoch 边界的额外代价 ----
ep_per_window = None
if bnd and norm:
    eph = st.mean(x[0] for x in bnd) - st.mean(x[0] for x in norm)
    steps_per_epoch = (steps[-1][1] - steps[0][1]) / max(1, (epochs[-1][1] - epochs[0][1]))
    n_ep_total = epochs[-1][1] - epochs[0][1]
    wall = steps[-1][0] - steps[0][0]
    print(f"每个 epoch 边界额外 = {eph:+.3f} s  (窗口里平均含 "
          f"{st.mean(x[2] for x in bnd):.2f} 个边界)")
    per_bnd = eph / max(1e-9, st.mean(x[2] for x in bnd))
    print(f"  → 单个边界 ≈ {per_bnd*1000:+.1f} ms")
    print()
    print(f"当前: {steps_per_epoch:.0f} 步/epoch, 全程 {n_ep_total} 个 epoch, "
          f"墙钟 {wall}s ({wall/3600:.2f}h)")
    print(f"  epoch 边界总代价 ≈ {per_bnd*n_ep_total:.1f} s "
          f"= {per_bnd*n_ep_total/wall*100:.2f}% of wall")
    print()
    print("若把 epoch 拉长到 N 步 (边界数随之减少):")
    for N in (500, 1000, 2500, 5000, 100000):
        nb_new = (steps[-1][1] - steps[0][1]) / N
        saved = per_bnd * (n_ep_total - nb_new)
        print(f"  epoch={N:>6} 步 -> 边界 {nb_new:>6.0f} 个, "
              f"省 {saved:>7.1f} s ({saved/wall*100:>5.2f}%)")
    print()
    # 每步绝对时间
    tot_steps = steps[-1][1] - steps[0][1]
    print(f"当前平均 step/s = {tot_steps/wall:.3f}")
    tot_norm = sum(x[0] for x in norm)
    print(f"仅看非边界窗口 step/s = {sum(x[1] for x in norm)/tot_norm:.3f}"
          f"  (边界窗口 step/s = {sum(x[1] for x in bnd)/sum(x[0] for x in bnd):.3f})")

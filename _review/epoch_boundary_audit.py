# -*- coding: utf-8 -*-
"""从真实训练日志量化「epoch 边界」的代价 (不需要卡, 纯文本分析)。

原理: 日志每 50 步打一行, 带秒级时间戳; epoch 边界又单独打一行
"Beginning epoch N..."。于是可以**直接 A/B**:
  窗口不含边界  -> 50 步的 Δt = 基线
  窗口含 1 个边界 -> 50 步的 Δt = 基线 + 该边界的暴露代价
两者样本量都很大 (474 个边界), 且全来自真实跑过的 run -> 不是微基准推测。

⚠ 必须排除 in_mem_eval 窗口 (每 ckpt_every=2500 步一次, 会占几十秒),
   否则均值被污染。eval 期间日志会出现 [in-mem-eval] / sampling 字样。

用法: python epoch_boundary_audit.py <train.log> [more.log ...]
"""
import re
import sys
import statistics as st

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*(.*)$")
STEP = re.compile(r"\(step=(\d+)\)")
EPOCH = re.compile(r"Beginning epoch (\d+)")
EVAL = re.compile(r"in-mem-eval|\[eval\]|sampling")


def ts2sec(s):
    hh, mm, ss = s.split(" ")[1].split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def parse(path):
    """-> steps:[(t,step)], bounds:[t], evals:[t]"""
    steps, bounds, evals = [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = ANSI.sub("", line).rstrip("\n")
            m = TS.match(line)
            if not m:
                continue
            t, body = ts2sec(m.group(1)), m.group(2)
            if EPOCH.search(body):
                bounds.append(t)
            elif EVAL.search(body):
                evals.append(t)
            else:
                ms = STEP.search(body)
                if ms:
                    steps.append((t, int(ms.group(1))))
    return steps, sorted(bounds), sorted(evals)


def audit(path):
    steps, bounds, evals = parse(path)
    print("=" * 78)
    print("LOG: %s" % path)
    print("  step 行 %d 行 | epoch 边界 %d 个 | eval 相关行 %d 行"
          % (len(steps), len(bounds), len(evals)))
    if len(steps) < 10:
        print("  样本太少, 跳过")
        return

    # 同一秒内多行 -> 以 step 递增去重, 只保留 step 变化的那条
    seq = []
    for t, s in steps:
        if seq and seq[-1][1] == s:
            continue
        seq.append((t, s))

    import bisect
    base, one, multi, skipped = [], [], [], 0
    for (t0, s0), (t1, s1) in zip(seq, seq[1:]):
        nstep = s1 - s0
        if nstep <= 0 or nstep > 200:          # 只取相邻的 50 步窗口
            continue
        i0 = bisect.bisect_right(bounds, t0)
        i1 = bisect.bisect_right(bounds, t1)
        nb = i1 - i0
        # 排除 eval: 窗口内任意 eval 行 -> 丢弃
        if bisect.bisect_right(evals, t1) != bisect.bisect_right(evals, t0):
            skipped += 1
            continue
        dt = t1 - t0
        if s1 - s0 == 50:
            (base if nb == 0 else one if nb == 1 else multi).append(dt)

    print("  已丢弃 eval 窗口: %d 个" % skipped)
    if not base or not one:
        print("  A/B 样本不足 (base=%d one=%d)" % (len(base), len(one)))
        return

    mb, mo = st.median(base), st.median(one)
    print("\n  -- 50 步窗口 wall time --")
    print("  不含 epoch 边界 : n=%4d  median %.1fs  mean %.1fs  (%.2f step/s)"
          % (len(base), mb, st.mean(base), 50 / mb))
    print("  含 1 个 epoch 边界: n=%4d  median %.1fs  mean %.1fs  (%.2f step/s)"
          % (len(one), mo, st.mean(one), 50 / mo))
    if multi:
        print("  含 >1 个边界     : n=%4d  median %.1fs" % (len(multi), st.median(multi)))
    print("  => 每个 epoch 边界的暴露代价 = %.2f s  (median 差)" % (mo - mb))
    lost_ms = (mo - mb) * 1000 / 50
    print("  => 摊到每步 = %.2f ms/step, 占 178ms/step 的 %.2f%%"
          % (lost_ms, lost_ms / 178 * 100))
    print("  => 现状 119 步/epoch: 每步付 %.2f ms" % (lost_ms))
    print("  => 若拉到 2500 步/epoch: 每步付 %.3f ms -> 净省 %.2f ms/step (%.2f%%)"
          % (lost_ms * 119 / 2500, lost_ms - lost_ms * 119 / 2500,
             (lost_ms - lost_ms * 119 / 2500) / 178 * 100))
    # 边界分布: 每 119 步一次 -> 检查 50 步窗口命中率是否符合预期 50/119
    print("\n  校验: 边界命中率 %.3f (理论 %.3f, 50 步窗口对 %d 步 epoch)"
          % (len(one) / (len(one) + len(base)), 50 / 119, 119))


for p in sys.argv[1:]:
    try:
        audit(p)
    except Exception as e:                                    # noqa: BLE001
        print("FAILED %s: %r" % (p, e))

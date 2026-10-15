#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phase_profile.py —— **训练/eval 路径的阶段级 profile**（monkey-patch ✓ 零源码改动 ✓）

依据 `docs/infra_1015.md` §6.3:
  "建议把性能监控分成三层:
     吞吐   | samples/s、完整 step ms
     **GPU 时间 | forward、backward、optimizer、H2D、REPA/critic 的 CUDA 时间 → 判断慢在哪里**
     硬件利用 | kernel gap、occupancy …" ✓

★★ 做法（良好工程 ✓）:
  · **不改远端源码** ✗ —— 用 monkey-patch 包装**真实**函数 ✓
    （远端与本地是**两套无共同祖先的仓库** ✗ 无法 pull ✓ 见本轮实测 ✓）
  · 阶段用 `perf_counter` 测**墙钟**（回答"时间花在哪" ✓）
    另用 `cuda.Event` 测**纯 GPU** 段 ✓
  · ★ 同步纪律: **每步只在边界同步一次** ✓（审查 §十一 ✓）
  · 自动发现类路径 ✓（不硬编码 ✗ 免得远端版本变动就失效 ✓）

用法::
    python tools/prof/phase_profile.py --config src/train/configs/v84A_std_cond.json \
        --global-batch-size 288 --max-steps 30 --skip 10 --measure 10
"""
from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

ACC = defaultdict(float)        # 阶段名 -> 累计**自耗时**（秒 ✓ 嵌套不重复计 ✓）
ACC_TOT = defaultdict(float)    # 阶段名 -> 累计**含子阶段**耗时（秒 ✓）
CNT = defaultdict(int)          # 阶段名 -> 次数
ORDER = []                      # 首次出现顺序（保证报告可读 ✓）
DEPTH = [0]                     # ★ 嵌套深度 ✓ —— 只有**最外层**才累计 ✓
CUR = [None]                    # 当前最外层阶段名 ✓

# ★★ 方法论（本轮实测纠正 ✓）:
#   用 `perf_counter` 包 Python 调用是**错的** ✗ —— CUDA 异步 ⇒ 测到的是
#   入队/反压等待 ✗ 不是 GPU 执行 ✗（实测 `model_forward` 报 1591ms > 整步 594ms ✗✗）
#   且嵌套调用（fm_loss ⊃ model_forward）会**重复计数** ✗（实测合计 601% ✗）
#   ⇒ 改用 **CUDA Event** ✓（在流内排队 ✓ `elapsed_time` 才是真 GPU 时长 ✓）
#      + **嵌套感知** ✓（只有最外层累计"自耗时" ✓）
EV = []                         # [(name, start_event, end_event)] 最外层 ✓
CPU_SELF = defaultdict(float)   # 最外层阶段的 **CPU 墙钟**（含 Python ✓ 单独列 ✓）
# ★★ 2026-10-16 实测修正 ✓: 必须**只在测量窗内**记事件 ✗
#   首版把全部 24 步都记了 ✓ 却按 10 步（测量窗）平均 ✗
#   ⇒ `fm_loss` 报 1685 ms/步 ✗ 而真实 ≈ 1685/2.4 ≈ 702 ms ✓（差 2.4 倍 ✗✗）
#   根因: warmup 步（更慢 ✓）与测量步混在一起 ✓ 且步数口径不一致 ✗
MEASURING = [False]


@contextlib.contextmanager
def phase(name):
    """阶段计时 ✓ —— GPU 用 CUDA Event ✓ CPU 用 perf_counter ✓ **嵌套感知** ✓
    且**只在测量窗内记录** ✓（否则步数口径不一致 ⇒ 数值虚高 ✗ 实测 ✓）。
    """
    outer = DEPTH[0] == 0
    rec = outer and MEASURING[0]
    if outer:
        DEPTH[0] = 0                      # 防止跨步残留 ✓
    if rec:
        if name not in ORDER:             # ★ 判重按 ORDER ✓（ACC 在报告时才填 ✗）
            ORDER.append(name)
        CNT[name] += 1
        s = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if s is not None:
            s.record()
        t0 = time.perf_counter()
    DEPTH[0] += 1
    try:
        yield
    finally:
        DEPTH[0] -= 1
        if rec:
            if torch.cuda.is_available():
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                EV.append((name, s, e))
            CPU_SELF[name] += time.perf_counter() - t0


def _wrap(obj, attr, label):
    """把 `obj.attr` 包一层计时 ✓（保留原语义与签名 ✓）。返回是否成功 ✓。"""
    orig = getattr(obj, attr, None)
    if orig is None or getattr(orig, "_phased", False):
        return False
    if isinstance(obj, type):
        def wrapper(self, *a, **kw):
            with phase(label):
                return orig(self, *a, **kw)
    else:
        def wrapper(*a, **kw):
            with phase(label):
                return orig(*a, **kw)
    wrapper._phased = True
    wrapper._orig = orig
    try:
        setattr(obj, attr, wrapper)
    except Exception:                                                 # noqa: BLE001
        return False
    return True


def install_patches():
    """装全部计时钩子 ✓（打印每项是否成功 ✓ 不静默 ✗）。"""
    ok = []

    # ① DataLoader 取数等待（审查 §7.2 的关键指标 ✓）
    #    ★ 修正: `DataLoader` **没有** `__next__` ✗（实测报错 ✓ 钩子如实返回 ✗ ✓）
    #      迭代器才有 ⇒ 必须包 `_BaseDataLoaderIter` ✓
    try:
        from torch.utils.data.dataloader import _BaseDataLoaderIter
        ok.append(("_BaseDataLoaderIter.__next__",
                   _wrap(_BaseDataLoaderIter, "__next__", "dataloader_wait")))
    except Exception:                                                 # noqa: BLE001
        ok.append(("_BaseDataLoaderIter.__next__", False))

    # ② 反向（全部 backward ✓ 含 FM 与 REPA 的分次反向 ✓）
    ok.append(("Tensor.backward", _wrap(torch.Tensor, "backward", "backward_total")))

    # ③ 优化器
    import torch.optim as _opt
    ok.append(("Optimizer.step", _wrap(_opt.Optimizer, "step", "opt_step")))
    ok.append(("Optimizer.zero_grad", _wrap(_opt.Optimizer, "zero_grad", "opt_zero_grad")))

    # ④ 模型 forward（自动发现 ✓）
    for mod, cls in (("src.model.dit", "DiT_2Cond"), ("src.model.dit_2cond", "DiT_2Cond"),
                     ("src.model.dit", "DiT"), ("src.model.dit_2cond", "DiT")):
        try:
            m = __import__(mod, fromlist=[cls])
            c = getattr(m, cls, None)
            if c is not None and _wrap(c, "forward", "model_forward"):
                ok.append(("%s.%s.forward" % (mod, cls), True))
                break
        except Exception:                                             # noqa: BLE001
            continue

    # ⑤ FM 损失 / 采样器 / **C2OT**（审查 §五 明确要求分段计时 ✓）
    #    ★ 实测: FM 损失入口是 `training_losses` ✓ **不是** `forward` ✗（远端 grep 确认 ✓）
    try:
        fm = __import__("src.loss.flow_matching", fromlist=["FlowMatching"])
        F = getattr(fm, "FlowMatching", None)
        if F is not None:
            for meth, lab in (("ddim_sample_loop", "eval_sampler"),
                              ("training_losses", "fm_loss_and_forward"),
                              ("_apply_c2ot", "c2ot_solver")):
                ok.append(("FlowMatching.%s" % meth, _wrap(F, meth, lab)))
    except Exception:                                                 # noqa: BLE001
        pass

    # ⑥ VAE 解码（eval 路径 ✓）
    for mod in ("src.utils.vae_fast", "src.eval.vae_fast", "src.utils.vae"):
        try:
            m = __import__(mod, fromlist=["x"])
            for cname in dir(m):
                c = getattr(m, cname)
                if isinstance(c, type) and _wrap(c, "decode", "eval_vae_decode"):
                    ok.append(("%s.%s.decode" % (mod, cname), True))
                    break
        except Exception:                                             # noqa: BLE001
            continue

    # ⑦ 梯度裁剪
    ok.append(("clip_grad_norm_", _wrap(torch.nn.utils, "clip_grad_norm_", "grad_clip")))

    # ⑧ ★ **REPA 损失**（自动发现 ✓ 名字含 REPA 的类 ✓）
    for mod in ("src.loss.repa", "src.loss.repa_loss", "src.loss.repa_align"):
        try:
            m = __import__(mod, fromlist=["x"])
        except Exception:                                             # noqa: BLE001
            continue
        for cname in dir(m):
            c = getattr(m, cname)
            if isinstance(c, type) and "REPA" in cname.upper():
                for meth in ("forward", "__call__"):
                    if _wrap(c, meth, "repa_loss"):
                        ok.append(("%s.%s.%s" % (mod, cname, meth), True))
                        break

    # ⑨ ★ **EMA 更新**（函数 ✓ 来自 trainer.py `update_ema(...)` ✓）
    for mod, fn in (("src.train.ema", "update_ema"), ("src.utils.ema", "update_ema")):
        try:
            m = __import__(mod, fromlist=[fn])
            if _wrap(m, fn, "ema_update"):
                ok.append(("%s.%s" % (mod, fn), True))
                break
        except Exception:                                             # noqa: BLE001
            continue

    # ⑩ ★★ **H2D 搬运** —— 只统计带 `non_blocking=True` 的 `.to()` ✓
    #    为何这样过滤（良好工程 ✓）: `.to()` 每步被调用**数百次** ✗
    #    全量包一层会引入巨大开销 ✗；而只有**大批量搬运**（batch→device ✓）
    #    才显式传 `non_blocking=True` ✓ ⇒ 过滤后每步只剩**几十次** ✓ 开销可忽略 ✓
    _orig_to = torch.Tensor.to

    def to_prof(self, *a, **kw):
        if kw.get("non_blocking"):
            with phase("h2d_nonblocking"):
                return _orig_to(self, *a, **kw)
        return _orig_to(self, *a, **kw)

    to_prof._phased = True
    torch.Tensor.to = to_prof
    ok.append(("Tensor.to(non_blocking=True)", True))

    print("  ═══ 计时钩子安装情况（不静默 ✓）═══")
    for name, good in ok:
        print("     %s %s" % ("✓" if good else "✗", name))
    return sum(1 for _n, g in ok if g)


def report(steps_measured, step_times, device):
    """★ 报告: 以 **CUDA Event** 的 GPU 时长为主 ✓（异步下唯一可信的口径 ✓）。"""
    # 把 Event 全部解析成 GPU 毫秒 ✓（此处才**同步一次** ✓ 审查 §十一 ✓）
    gpu = defaultdict(float)
    if device == "cuda" and EV:
        torch.cuda.synchronize()          # ★ 唯一一次同步 ✓（窗口边界 ✓）
        for name, s, e in EV:
            try:
                gpu[name] += s.elapsed_time(e)      # 毫秒 ✓
            except Exception:                                     # noqa: BLE001
                pass
    print()
    print("  ═══ 阶段级分解（%d 步实测 ✓）═══" % steps_measured)
    total_wall = sum(step_times)
    gpu_tot = sum(gpu.values())
    print("  %-22s %11s %10s %9s %9s %7s" %
          ("阶段", "GPU 累计ms", "GPU/步ms", "占GPU比", "CPU/步ms", "次数"))
    for name in ORDER:
        g = gpu.get(name, 0.0)
        print("  %-22s %11.1f %10.2f %8.1f%% %9.2f %7d"
              % (name, g, g / max(1, steps_measured),
                 100.0 * g / gpu_tot if gpu_tot else 0.0,
                 CPU_SELF[name] * 1000.0 / max(1, steps_measured), CNT[name]))
    print("  %-22s %11.1f %10.2f %8.1f%%" %
          ("（GPU 阶段合计）", gpu_tot, gpu_tot / max(1, steps_measured), 100.0))
    med = statistics.median(step_times)
    print()
    print("  ★ 中位步时（墙钟 ✓）= **%.1f ms** ⇒ %.2f step/s ✓"
          % (med * 1000, 1.0 / med if med else 0))
    if gpu_tot:
        print("  ★ **GPU 忙碌合计 = %.1f ms/步** ✓ ⇒ GPU 占空比 ≈ **%.1f%%** ✓"
              % (gpu_tot / max(1, steps_measured),
                 100.0 * (gpu_tot / max(1, steps_measured)) / (med * 1000)))
        idle = med * 1000 - gpu_tot / max(1, steps_measured)
        print("  ★ **GPU 空转/等待 ≈ %.1f ms/步（%.1f%%）** %s"
              % (idle, 100.0 * idle / (med * 1000),
                 "✗ ⇒ **CPU/同步瓶颈**（审查 §7.3 P0 ✓）" if idle > 0.25 * med * 1000
                 else "✓ ⇒ 基本算力受限 ✓"))
    print()
    print("  ★ GPU 阶段占比最高的三项（**优化重点** ✓）:")
    for n in sorted(ORDER, key=lambda x: -gpu.get(x, 0.0))[:3]:
        g = gpu.get(n, 0.0)
        print("     %-22s %6.1f%% ✓（每步 %.2f ms ✓）"
              % (n, 100.0 * g / gpu_tot if gpu_tot else 0.0, g / max(1, steps_measured)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--global-batch-size", type=int, default=288)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--skip", type=int, default=10, help="前 N 步不计入（warmup ✓）")
    ap.add_argument("--measure", type=int, default=10, help="计时的步数 ✓")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    n_ok = install_patches()
    print("  ✓ 共装 %d 个钩子 ✓（0 个也能跑 ✓ 但报告会缺阶段 ✓）" % n_ok)

    # ★ 用**真实入口**跑训练 ✓ 不改一行源码 ✓
    from src.train import train as T
    argv = ["--config", args.config,
            "--global-batch-size", str(args.global_batch_size),
            "--max-steps", str(args.max_steps),
            "--num-workers", str(args.num_workers),
            "--log-every", "1", "--in-mem-eval-sets", ""] + list(args.extra)
    print("  ═══ 启动真实训练（%s ✓）═══" % " ".join(argv))
    print("  ⚠ 前 %d 步为 warmup（不计入 ✓）；第 %d..%d 步计时 ✓"
          % (args.skip, args.skip + 1, args.skip + args.measure))

    # 用 DataLoader 迭代器的 __next__ 计数来界定测量窗口 ✓
    #   ★ 必须包 `_BaseDataLoaderIter` ✓（`DataLoader` 没有 `__next__` ✗ 实测 ✓）
    step_times = []
    state = {"n": 0, "t0": None}
    from torch.utils.data.dataloader import _BaseDataLoaderIter
    _orig_next = getattr(_BaseDataLoaderIter.__next__, "_orig",
                         _BaseDataLoaderIter.__next__)

    def tick(*a, **kw):
        t = time.perf_counter()
        # ★ 测量窗: 第 skip+1 .. skip+measure 步 ✓（与事件记录同一口径 ✓）
        MEASURING[0] = args.skip < state["n"] <= args.skip + args.measure
        if MEASURING[0] and state["t0"] is not None:
            step_times.append(t - state["t0"])
        state["t0"] = t
        state["n"] += 1
        return _orig_next(*a, **kw)

    _BaseDataLoaderIter.__next__ = tick

    t_start = time.perf_counter()
    try:
        T.main_from_cli(argv)
    except SystemExit:
        pass
    wall = time.perf_counter() - t_start
    print()
    print("  ═══ 总墙钟 %.1f s ✓（含启动 ✓）═══" % wall)
    if step_times:
        report(len(step_times), step_times, "cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("  ⚠ 未采集到步时 ✗（步数不足或钩子未生效 ✓）—— 已计阶段仍如下 ✓")
        report(1, [wall], "cuda" if torch.cuda.is_available() else "cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

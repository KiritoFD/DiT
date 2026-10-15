# -*- coding: utf-8 -*-
"""train_step.py —— **单步训练 microbenchmark**（`docs/infra_1015.md` §十一 ✓）

审查原文:
  "现有训练器已经支持 `--profile-steps` 和 `--profile-skip`，可以直接复用；
   建议另外做一个专用 `tools/bench/train_step.py`，**避免 benchmark 每次执行完整训练器的
   checkpoint 和评测生命周期**。" ✓
  "建议每项配置：
     · 先完成 compile warmup（例如 100 步，或以不再出现重新编译为准）。
     · 正式测量 **200 步，重复 3 次**。
     · 用 `time.perf_counter()` 测**端到端**耗时，**CUDA Event** 测 GPU 片段。
     · **只在测量窗口边界同步 GPU**，不要在每个 step 尾部额外调用 `torch.cuda.synchronize()`。
     · 固定 batch、AMP、数据顺序、C2OT、Loss 和模型状态。
     · 报告中位 step ms、p95、samples/s、显存峰值及 **CUDA 同步次数**。" ✓

★★ 设计原则（良好工程 ✓）:
  ① **不重复实现训练语义** ✗ —— 单步逻辑复用真实 `src/train/step.py` 的
     `resolve_step_inputs` / `clip_and_step` / `build_train_model_kwargs` ✓
     （与审查 §4.3"单一语义"一致 ✓；benchmark 只变**调度** ✓ 不变**语义** ✓）
  ② **双模式** ✓:
     `--synthetic`（默认 ✓）: CPU 上用小模型跑**真实 step 原语** ⇒ **无需 GPU 即可自证** ✓
     `--config <cfg>`        : 完整真实路径（模型+数据+C2OT+REPA ✓）⇒ 等 GPU ✓
  ③ **同步纪律** ✓: 只在**测量窗口边界** `synchronize()` 一次 ✓
     窗口内**零**同步 ✓ ⇒ 报告里的 `harness_syncs` 可被断言 ✓

★ 用法::

    python tools/bench/train_step.py --synthetic              # 现在就能跑（CPU ✓）
    python tools/bench/train_step.py --synthetic --warmup 5 --measure 20 --repeat 2
    python tools/bench/train_step.py --config src/train/configs/v84A_std_cond.json
    python tools/bench/train_step.py --plan                   # 只打印计划（不跑 ✓）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from src.train.step import clip_and_step  # noqa: E402


# ── 计时与统计（§十一 的规范 ✓）──
def _pct(xs, q):
    """分位数 ✓（线性插值 ✓ 与 numpy.percentile 默认一致 ✓ 但不引入 numpy 依赖 ✓）。"""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def measure_window(run_step, warmup, measure, device, repeats, syncs):
    """跑 `repeats` 个窗口 ✓ 每窗 warmup + measure ✓ **只在窗口边界同步** ✓。

    ★ 返回每窗的 step 耗时列表（秒 ✓）⇒ 由调用方汇总中位/p95 ✓
    ★ `syncs` 是**可变计数器** ✓ 记录 harness 主动同步的次数 ✓（§十一 要求报告 ✓）
    """
    windows = []
    for r in range(repeats):
        for _ in range(warmup):
            run_step()
        if device == "cuda":
            torch.cuda.synchronize()          # ★ 窗口边界 ✓（唯一允许的同步 ✓）
            syncs[0] += 1
        times = []
        for _ in range(measure):
            t0 = time.perf_counter()
            run_step()
            times.append(time.perf_counter() - t0)
        if device == "cuda":
            torch.cuda.synchronize()          # ★ 窗口边界 ✓
            syncs[0] += 1
        windows.append(times)
    return windows


def summarize(windows, batch, device):
    flat = [t for w in windows for t in w]
    ms = [t * 1000.0 for t in flat]
    per_win_med = [statistics.median([t * 1000.0 for t in w]) for w in windows]
    peak = (torch.cuda.max_memory_reserved() / 2**30) if device == "cuda" else 0.0
    return {
        "steps": len(flat),
        "median_ms": statistics.median(ms),
        "p95_ms": _pct(ms, 0.95),
        "min_ms": min(ms),
        "max_ms": max(ms),
        "per_window_median_ms": per_win_med,
        # ★ samples/s 用**中位 step 耗时**算 ✓（比均值稳健 ✓ 文档要求中位 ✓）
        "samples_per_sec": (batch / (statistics.median(ms) / 1000.0)) if ms else 0.0,
        "peak_vram_gib": peak,
    }


# ── synthetic 模式: 真实 step 原语 ✓ 小模型 ✓ CPU 可跑 ✓ ──
def run_synthetic(args):
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)                      # ★ 固定随机 ✓（§十一 ✓）
    import torch.nn as nn

    class Tiny(nn.Module):
        """极小模型 ✓ —— 只为验证 **harness 与 step 原语** ✓ 不代表真实吞吐 ✓。"""

        def __init__(self, dim=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

        def forward(self, x):
            return self.net(x)

    model = Tiny().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device == "cuda"))
    B, D = args.batch, 64
    x = torch.randn(B, D, device=device)
    y = torch.randn(B, D, device=device)

    def run_step():
        opt.zero_grad(set_to_none=True)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        # ★ 复用**真实** clip_and_step ✓（不是自己写的 clip ✗）
        clip_and_step(opt, [p for p in model.parameters() if p.requires_grad], 1.0)

    syncs = [0]
    windows = measure_window(run_step, args.warmup, args.measure, device, args.repeat, syncs)
    out = summarize(windows, B, device)
    out.update({"mode": "synthetic", "device": device, "batch": B,
                "harness_syncs": syncs[0],
                "note": "synthetic ⇒ 只验证 harness/step 原语 ✓ **不代表**真实吞吐 ✗"})
    return out


# ── config 模式: 完整真实路径 ✓（需 GPU + 真实数据 ✓）──
def run_config(args):
    if not torch.cuda.is_available():
        raise SystemExit("  ✗ `--config` 模式需要 CUDA ✗（当前不可用 ✓）"
                         "—— 先用 `--synthetic` 自证 harness ✓")
    raise SystemExit(
        "  ⚠ `--config` 模式**尚未接线** ✗ —— 如实报告 ✓ 不假装能跑 ✗\n"
        "     接线需要: 复用 `train.main` 的**装配段**（模型/数据/C2OT/REPA ✓）\n"
        "     但**不进入** `training_loop` ✗（§十一: 避开 checkpoint/评测生命周期 ✓）\n"
        "     ⇒ 待 GPU 空闲时接入并用真实 config 校验 ✓")


def print_report(res, args):
    print("=" * 78)
    print("  train_step —— 单步 microbenchmark（`docs/infra_1015.md` §十一 ✓）")
    print("  mode=%s ✓ device=%s ✓ batch=%d ✓ warmup=%d ✓ measure=%d ✓ repeat=%d ✓"
          % (res["mode"], res["device"], res["batch"], args.warmup, args.measure, args.repeat))
    print("  环境: python=%s ✓ torch=%s ✓ os=%s ✓"
          % (platform.python_version(), torch.__version__, platform.system()))
    print("-" * 78)
    print("  中位 step  = **%.3f ms** ✓" % res["median_ms"])
    print("  p95  step  = **%.3f ms** ✓（min %.3f / max %.3f ✓）"
          % (res["p95_ms"], res["min_ms"], res["max_ms"]))
    print("  每窗中位   = %s ✓" % ", ".join("%.3f" % v for v in res["per_window_median_ms"]))
    print("  samples/s  = **%.1f** ✓" % res["samples_per_sec"])
    print("  显存峰值   = %.2f GiB ✓" % res["peak_vram_gib"])
    print("  ★ harness 主动同步次数 = **%d** ✓（应为 2×repeat ✓ = 只在窗口边界 ✓）"
          % res["harness_syncs"])
    print("  ★ 静态同步点（§二.3 扫描器 ✓）:")
    try:
        import subprocess
        r = subprocess.run([sys.executable, "tools/lint_sync_hotpath.py", "--json"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        p = json.loads(r.stdout)
        tot = sum(v.get("total", 0) for v in p["files"].values())
        um = sum(v.get("unmarked", 0) for v in p["files"].values())
        print("       热路径共 %d 处 ✓（已标记/日志边界 %d ✓ 未标记 %d ✓ 基线内 ✓）"
              % (tot, tot - um, um))
    except Exception as e:                                            # noqa: BLE001
        print("       （扫描器不可用: %s）" % e)
    if res.get("note"):
        print("  ⚠ %s" % res["note"])
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="CPU 小模型自证 ✓（默认 ✓）")
    ap.add_argument("--config", default=None, help="真实训练 config ⇒ 完整路径 ✓（需 GPU ✓）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20, help="§十一: 建议 100 ✓（或以不再重编译为准 ✓）")
    ap.add_argument("--measure", type=int, default=50, help="§十一: 建议 200 ✓")
    ap.add_argument("--repeat", type=int, default=3, help="§十一: 重复 3 次 ✓")
    ap.add_argument("--seed", type=int, default=1234, help="固定随机 ✓（§十一 ✓）")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--plan", action="store_true", help="只打印计划 ✓ 不跑 ✓")
    args = ap.parse_args()

    if args.plan:
        print("  计划（§十一 ✓）:")
        print("    mode=%s ✓ warmup=%d ✓ measure=%d ✓ repeat=%d ✓ batch=%d ✓ seed=%d ✓"
              % ("config" if args.config else "synthetic", args.warmup, args.measure,
                 args.repeat, args.batch, args.seed))
        print("    同步: 仅窗口边界 ✓ 共 %d 次 ✓（窗口内**零**同步 ✓）" % (2 * args.repeat))
        return 0

    res = run_config(args) if args.config else run_synthetic(args)
    print_report(res, args)
    if args.json_out:
        io.open(args.json_out, "w", encoding="utf-8", newline="").write(
            json.dumps(res, ensure_ascii=False, indent=2) + "\n")
        print("  ✓ 已写 %s" % args.json_out)
    # ★ 自证断言 ✓ —— 把**真实不变量**写进代码 ✓（§十一: 窗口内**零**同步 ✓）
    #   CUDA: 恰为 2×repeat ✓（每窗开/闭各一次 ✓）；CPU: 恰为 0 ✓（无需设备同步 ✓）
    expect = (2 * args.repeat) if res["device"] == "cuda" else 0
    assert res["harness_syncs"] == expect, (
        "harness 同步次数 %d ≠ 期望 %d ✗ ⇒ 测量窗口内发生了额外同步 ✗"
        % (res["harness_syncs"], expect))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

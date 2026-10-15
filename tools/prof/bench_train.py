# -*- coding: utf-8 -*-
"""bench_train.py —— **单卡训练 profiling harness**（2026-10-14 [重构 PR6] ✓）

审查 §6.3 原文:
  "特别建议做两套基准：
   **Benchmark A：纯基模 FM+REPA。** 固定 B=192/240/288，关闭评测、梯度探针和额外 Critic，
   测试真正的主干效率。
   **Benchmark B：强风格后训练。** 同样固定 B 和训练步数，逐项打开 Writer、Brush、
   反事实分支及梯度探针，测量各自的额外成本。
   两套结果不要使用同一个未经校正的 MFU 值评价。" ✓
审查 §7.2 原文:
  "我建议做一个真实吞吐矩阵: A num_workers=0 / B 2 / C 4 / D 8(prefetch 2) / E 8(prefetch 4)"
  "每项使用相同的 B、compile 模式、数据源和随机种子，比较稳定 samples/s、CPU 开销和
   GPU kernel gap" ✓

★★ 设计原则（审查 §4.3"单一语义" ✓）:
  **不自己实现训练循环** ✗ —— 直接启动**真实入口** `python -m src.train.train` ✓
  吞吐与显存**从它自己的日志解析** ✓（日志已打印 `Steps/Sec:` 与 `Mem: X/Y` ✓）
  ⇒ 零重复实现 ✓ 测的就是真实路径 ✓

★ 产出:
  · 每个 cell: `samples/s` ✓ `step ms` ✓ **峰值显存** ✓
  · **校正后的代理 MFU** ✓（用审查 §6.1/§6.2 的公式 ✓ 分母 165.2 TFLOP/s ✓）
  · **实验身份** ✓（git rev ✓ config sha ✓ batch ✓ workers ✓ compile ✓）

用法::

    # 先 dry-run 看命令（无需 GPU ✓）
    python tools/prof/bench_train.py --dry-run
    # 正式跑（GPU 空闲时 ✓）
    python tools/prof/bench_train.py --benchmark A --batch 192,240,288 --steps 30
    python tools/prof/bench_train.py --benchmark B --batch 288 --steps 30
    python tools/prof/bench_train.py --matrix dataloader      # §7.2 的 A–E 矩阵
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── 审查 §6.1 的静态 FLOPs 估算（**保持同一公式** ✓ 才能跨配置比较 ✓）──
F_SAMPLE_GFLOPS = 96.84          # ≈ 6·P·N + 12·L·N²·d ✓ P=59.9M, N=256, L=12, d=512 ✓
PEAK_TFLOPS_BF16 = 165.2         # 4090 BF16 + FP32 累加 ✓（**不是** 330.3 ✗ 那是 FP16 累加 ✓）

# 审查 §7.2 的 DataLoader 矩阵 ✓
DL_MATRIX = {"A": (0, None), "B": (2, 2), "C": (4, 2), "D": (8, 2), "E": (8, 4)}


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "?"
    except Exception:                                                 # noqa: BLE001
        return "?"


def file_sha(path: str) -> str:
    try:
        return hashlib.sha256(io.open(path, "rb").read()).hexdigest()[:16]
    except Exception:                                                 # noqa: BLE001
        return "?"


def proxy_mfu(batch: int, steps_per_sec: float) -> float:
    """审查 §6.2 的代理 MFU ✓（**同一个未经校正的公式** ⇒ 只在同套基准内可比 ✓）。"""
    return (F_SAMPLE_GFLOPS * 1e9 * batch * steps_per_sec) / (PEAK_TFLOPS_BF16 * 1e12)


# ── 日志解析（用真实日志 ✓ 不另造计数器 ✗）──
RE_STEPS = re.compile(r"Steps/Sec:\s*([0-9.]+)")
RE_MEM = re.compile(r"Mem:\s*([0-9.]+)G/([0-9.]+)G")
RE_DIFF = re.compile(r"Diff:\s*([0-9.]+)")


def parse_log(text: str, skip_first: int = 2) -> dict:
    """从**真实日志**解析吞吐 ✓（`infra_1015.md` §十一 ✓）。

    ★ `skip_first`（默认 2 ✓）: 丢掉**前几步**的日志 ✓
      实测发现: 首步会打印 `Steps/Sec: 0.02` ✗（compile/预热/首次分配的假值 ✓）
      ⇒ 若不丢，均值会被严重拉低 ✗ ⇒ 报告失真 ✗
    ★ 主指标改用**后段中位**（而非全均值 ✓）:
      与审查 §十一 要求一致（"报告中位 step ms" ✓）；均值对尾部抖动敏感 ✗
    """
    sps = [float(x) for x in RE_STEPS.findall(text)]
    mem = [(float(a), float(b)) for a, b in RE_MEM.findall(text)]
    loss = [float(x) for x in RE_DIFF.findall(text)]
    sps_use = sps[skip_first:] if len(sps) > skip_first else sps
    med = sorted(sps_use)[len(sps_use) // 2] if sps_use else 0.0
    return {
        "n_log_lines": len(sps),
        "n_used": len(sps_use),
        "skipped_warmup": min(skip_first, len(sps)),
        "steps_per_sec": med,              # ★ 后段**中位** ✓（稳健 ✓）
        "sps_mean_used": (sum(sps_use) / len(sps_use)) if sps_use else 0.0,
        "sps_first_raw": sps[0] if sps else 0.0,
        "sps_last": sps[-1] if sps else 0.0,
        "peak_reserved_gb": max((b for _a, b in mem), default=0.0),
        "mem_cur_gb": mem[-1][0] if mem else 0.0,
        "loss_last": loss[-1] if loss else None,
    }


def build_cmd(args, batch: int, workers: int, prefetch):
    cmd = [sys.executable, "-u", "-m", "src.train.train",
           "--config", args.config,
           "--global-batch-size", str(batch),
           "--max-steps", str(args.steps),
           "--num-workers", str(workers)]
    if args.log_every:
        cmd += ["--log-every", str(args.log_every)]
    # ⚠ `prefetch_factor` **没有暴露成 CLI 开关** ✗（2026-10-14 用 train CLI 的
    #   actions 清单**核实**过 ✓）⇒ 审查 §7.2 矩阵的 D(8,2) / E(8,4)
    #   **当前无法从命令行区分** ✗ ⇒ 要跑那两行必须先在配置里暴露该键 ✓
    #   （已如实记为待办 ✓ 不要假装测过 ✗）
    # ── Benchmark A: **关闭**评测 / 梯度探针 / 额外 Critic ✓（审查 §6.3 ✓）──
    if args.benchmark == "A":
        cmd += ["--in-mem-eval-sets", "", "--w-critic-writer", "0", "--w-critic-away", "0"]
        if args.no_probe:
            cmd += ["--strong-style-probe-every", "0"]
    # ── Benchmark B: **逐项打开**强风格（审查 §6.3 ✓ 用 args.extra 传入 ✓）──
    if args.benchmark == "B" and args.enable:
        cmd += list(args.enable)
    if args.extra:
        cmd += list(args.extra)
    return cmd


def run_cell(args, batch: int, workers: int, prefetch, tag: str) -> dict:
    cmd = build_cmd(args, batch, workers, prefetch)
    if args.dry_run:
        print("    [dry-run] %s" % " ".join(cmd))
        return {"tag": tag, "batch": batch, "workers": workers, "prefetch": prefetch,
                "cmd": " ".join(cmd), "dry_run": True}
    log = os.path.join(args.out_dir, "bench_%s.log" % tag)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    with io.open(log, "w", encoding="utf-8", newline="") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    wall = time.time() - t0
    text = io.open(log, encoding="utf-8", errors="replace").read()
    res = parse_log(text)
    res.update({"tag": tag, "batch": batch, "workers": workers, "prefetch": prefetch,
                "cmd": " ".join(cmd), "exit": proc.returncode, "wall_s": round(wall, 1),
                "log": log})
    if res["steps_per_sec"]:
        res["samples_per_sec"] = res["steps_per_sec"] * batch
        res["ms_per_step"] = 1000.0 / res["steps_per_sec"]
        res["proxy_mfu"] = proxy_mfu(batch, res["steps_per_sec"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["A", "B"], default="A",
                    help="A=纯基模 FM+REPA（关评测/探针/Critic ✓）；B=强风格（逐项打开 ✓）")
    ap.add_argument("--config", default="src/train/configs/v84A_std_cond.json")
    ap.add_argument("--batch", default="192,240,288", help="逗号分隔 ✓ 审查 §6.3 用这三档 ✓")
    ap.add_argument("--steps", type=int, default=30, help="每档总步数（含 warmup ✓）")
    ap.add_argument("--log-every", type=int, default=5, help="日志周期（吞吐从这里取 ✓）")
    ap.add_argument("--workers", type=int, default=None, help="单点测试时的 num_workers")
    ap.add_argument("--prefetch", type=int, default=None)
    ap.add_argument("--matrix", choices=["batch", "dataloader"], default="batch",
                    help="batch=审查 §6.3 的 B 档；dataloader=审查 §7.2 的 A–E 矩阵")
    ap.add_argument("--enable", nargs="*", default=None,
                    help="Benchmark B: 逐项打开的 CLI 开关（如 --w_critic_writer 0.01 ✓）")
    ap.add_argument("--extra", nargs="*", default=None, help="附加 CLI 开关 ✓")
    ap.add_argument("--no-probe", action="store_true", help="A 档关闭梯度探针 ✓")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令（无需 GPU ✓）")
    ap.add_argument("--out-dir", default="logs/prof")
    args = ap.parse_args()

    batches = [int(x) for x in str(args.batch).split(",") if x.strip()]
    cells = []
    if args.matrix == "dataloader":
        for k, (w, pf) in DL_MATRIX.items():
            cells.append((batches[0], w, pf, "dl%s" % k))
    else:
        for b in batches:
            cells.append((b, args.workers if args.workers is not None else 8,
                          args.prefetch if args.prefetch is not None else 4, "b%d" % b))

    ident = {"git": git_rev(), "config": args.config, "config_sha": file_sha(args.config),
             "benchmark": args.benchmark, "steps": args.steps,
             "f_sample_gflops": F_SAMPLE_GFLOPS, "peak_tflops_bf16": PEAK_TFLOPS_BF16}
    print("=" * 78)
    print("  bench_train —— Benchmark %s ✓（审查 §6.3 ✓）" % args.benchmark)
    print("  身份: git=%s config=%s sha=%s" % (ident["git"], args.config, ident["config_sha"]))
    print("  FLOPs 模型: F_sample=%.2f GFLOPs ✓ 峰值 %.1f TFLOP/s(BF16+FP32累加 ✓)"
          % (F_SAMPLE_GFLOPS, PEAK_TFLOPS_BF16))
    print("  ⚠ 代理 MFU 用**同一未经校正的公式** ✓ ⇒ 只在**同套基准内**可比 ✗")
    print("=" * 78)

    rows = []
    for b, w, pf, tag in cells:
        print("  ▶ %s: batch=%d workers=%s prefetch=%s" % (tag, b, w, pf))
        r = run_cell(args, b, w, pf, tag)
        rows.append(r)
        if not args.dry_run and r.get("steps_per_sec"):
            print("      %.2f steps/s ✓ %.0f samples/s ✓ %.1f ms/step ✓ 峰值 %.2f GB ✓ MFU≈%.1f%%"
                  % (r["steps_per_sec"], r["samples_per_sec"], r["ms_per_step"],
                     r["peak_reserved_gb"], 100 * r["proxy_mfu"]))
            print("        （统计口径: 已丢前 %d 条 warmup ✓ 用 %d 条的**中位** ✓ 首条原值 %.2f ✓）"
                  % (r["skipped_warmup"], r["n_used"], r["sps_first_raw"]))
        elif not args.dry_run:
            print("      ✗ 未解析到吞吐（exit=%s ✓ 看 %s）" % (r.get("exit"), r.get("log")))

    out = {"identity": ident, "rows": rows}
    if not args.dry_run:
        p = os.path.join(args.out_dir, "bench_%s.json" % args.benchmark)
        os.makedirs(args.out_dir, exist_ok=True)
        io.open(p, "w", encoding="utf-8", newline="").write(json.dumps(out, indent=2, ensure_ascii=False))
        print("  ✓ 结果已写 %s" % p)
    else:
        print("  （dry-run ✓ 未启动任何训练 ✓）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

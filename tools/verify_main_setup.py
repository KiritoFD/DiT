#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_main_setup.py — `train.main()` **setup 路径**的行为指纹护栏 ✓

为什么需要
----------
`tools/smoke_train.py` 自己建模型/建 diffusion, **不走 main()** ⇒ main() 的
setup 路径（args 解析/校验、callig 映射、triple 表、数据/分片、ckpt 恢复、DDP、
optim/scheduler 组装、日志）**零覆盖** ✗。而 main() 是 2500+ 行的上帝函数 ——
按重构铁律「没有测试覆盖的核心代码不重构」, 必须先给它建指纹 ✓。

做法
----
用真实 config 跑 main() **2 步**, 捕获 setup 阶段打印的诊断日志, 做**归一化**
(去掉时间戳/路径/耗时/显存等易变项) 后取 sha256 = **行为指纹** ✓。
指纹变化 = setup 行为变了 ✗（比"跑通没报错"强得多 ✓）。

用法（在 4090 上跑, 因为 main() 断言 CUDA ✓）:
  python tools/verify_main_setup.py --config <cfg> --golden <p> --save
  python tools/verify_main_setup.py --config <cfg> --golden <p> --check
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 归一化: 抹掉一切"每次都会变"的东西 ✓
SUBS = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?", re.I), "<TIME>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|sec|seconds|min|ms|it/s|s/it)\b"), "<DUR>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:GiB|MiB|GB|MB|B)\b"), "<MEM>"),
    (re.compile(r"(?:/root|/tmp|/home|G:|C:)[^\s,;:)\]]*"), "<PATH>"),
    (re.compile(r"\b[0-9a-f]{16,}\b"), "<HEX>"),
    (re.compile(r"\b\d+\.\d{4,}\b"), "<FLOAT>"),          # 长小数（loss 等）
    (re.compile(r"\bstep\s*\d+\b"), "step<N>"),
    (re.compile(r"\b\d+/\d+\b"), "<N/N>"),
]


# 训练期开始的标志（实测取自真实日志 ✓ —— 别猜 ✗）:
#   "[<TIME>] Begin epoch 0 (steps 1-5000) — ckpt+eval 落点..."
#   以及任何 traceback / loss 行（异常时也要切断, 否则报错栈会进指纹 ✗）
TRAIN_MARK = re.compile(r"(Begin epoch|Traceback \(most recent|"
                        r"loss_dict\s*=|it\s*\d+.*loss|\[ckpt\] async writer)", re.I)


def setup_only(text: str) -> str:
    """只保留**训练开始之前**的日志 —— setup 阶段的数字就是"解析后的配置值",
    必须保留 ✗不能抹（否则护栏会漏掉 in_channels 之类真变化 ✗）; 训练期的 loss/耗时
    才是易变项, 整体丢弃 ✓"""
    out = []
    for ln in text.splitlines():
        if TRAIN_MARK.search(ln):
            break
        out.append(ln)
    return "\n".join(out)


def normalize(text: str) -> str:
    for pat, rep in SUBS:
        text = pat.sub(rep, text)
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("[cond-route]"):          # 该行是常量, 保留 ✓
            out.append(ln)
        elif re.search(r"[A-Za-z\u4e00-\u9fff]", ln):
            out.append(ln)
    return "\n".join(out)


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(setup_only(text)).encode("utf-8")).hexdigest()


def run_setup(cfg_path, steps=2):
    """跑 main() 的前若干步, 返回 (归一化日志, 是否推进到 step)"""
    from src.train import cli as train_cli
    from src.train import train as train_mod
    a = train_cli.parse_args([]) if hasattr(train_cli, "parse_args") else None
    # 用 config 解析（与 main_from_cli 同路径 ✓）
    argv = ["--config", cfg_path]
    if hasattr(train_cli, "parse_args"):
        try:
            a = train_cli.parse_args(argv)
        except TypeError:
            a = train_cli.parse_args()
    if a is None:
        raise SystemExit("无法解析 config: %s" % cfg_path)
    tmp = tempfile.mkdtemp(prefix="vms_")
    # ⚠ 实测: 用完整 batch(288) 会与并行的 OCR 训练争显存 -> OOM -> 报错栈混进
    #   指纹 -> 指纹不稳定 ✗。故用极小 batch/0 workers ✓
    for k, v in (("max_steps", steps), ("results_dir", tmp), ("eval_every", 0),
                 ("save_every", 0), ("ckpt_every", 0), ("num_workers", 0),
                 ("preload_workers", 0), ("global_batch_size", 2), ("epoch_steps", 100000),
                 ("auto_eval", False), ("eval_auto", False), ("in_mem_eval", False),
                 ("use_wandb", False)):
        if hasattr(a, k):
            setattr(a, k, v)
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            train_mod.main(a)
    except SystemExit as e:                    # 训练正常结束可能 SystemExit ✓
        err = "SystemExit(%s)" % e.code
    except Exception as e:                     # noqa: BLE001
        err = "%s: %s" % (type(e).__name__, e)
    log = buf.getvalue()
    return log, err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--golden")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    log, err = run_setup(args.config, args.steps)
    fp = fingerprint(log)
    nlines = len(normalize(setup_only(log)).splitlines())
    has_step = bool(re.search(r"\bstep\b", log, re.I))
    _cut = next((ln.strip()[:70] for ln in log.splitlines() if TRAIN_MARK.search(ln)), "(无)")
    print("  setup 段 %d 行 · 指纹 %s · 推进到 step=%s · 异常=%s"
          % (nlines, fp[:16], has_step, err or "无"))
    print("  切点(训练开始) = %s" % _cut)
    if err:
        print("  ⚠ 运行期异常（可能是 DDP/环境退出）:", err)
    if nlines < 5:
        print("  ✗ 捕获到的 setup 日志过少 —— 护栏无效, 不要当通过 ✗")
        return 3

    # ★★ 崩溃检测：切点只截到 [Begin epoch]，若崩溃发生在其**之后**，指纹照样"一致" ✗✗
    #    ⇒ 必须独立断言：无 traceback / 无 Error / 真的推进到训练步 ✓
    tb = ("Traceback (most recent call last)" in log) or ("Error" in log and "OutOfMemory" in log)
    real_step = bool(re.search(r"\bstep\s*[1-9]\d*\b", log)) and ("loss" in log.lower())
    if tb:
        print("  ✗ 运行中出现 traceback/Error —— 护栏判定**失败** ✗（切点之后崩溃也会被抓 ✓）")
        return 4
    if not real_step:
        print("  ✗ 未能确认真的跑过训练步（未见 step>=1 的 loss 行）⇒ 护栏无效 ✗")
        return 4
    meta = {"config": args.config, "steps": args.steps, "sha256": fp, "nlines": nlines,
            "has_step": has_step, "real_step": real_step}
    if args.save:
        if not args.golden:
            print("  ✗ --save 需要 --golden"); return 2
        io.open(args.golden + ".txt", "w", encoding="utf-8").write(normalize(log))
        json.dump(meta, io.open(args.golden, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("  ✓ 指纹已保存 ->", args.golden)
        return 0
    if args.check:
        if not args.golden or not os.path.isfile(args.golden):
            print("  ✗ 黄金不存在:", args.golden); return 2
        g = json.load(io.open(args.golden, encoding="utf-8"))
        if g["sha256"] == fp:
            print("  ✓ setup 行为指纹**一致** —— 重构未改变 setup ✓✓")
            return 0
        print("  ✗ setup 指纹变了（黄金 %s != 当前 %s）" % (g["sha256"][:16], fp[:16]))
        gold_lines = io.open(args.golden + ".txt", encoding="utf-8").read().splitlines()
        now_lines = normalize(log).splitlines()
        import difflib
        diff = list(difflib.unified_diff(gold_lines, now_lines, "golden", "now", lineterm="", n=1))
        for l in diff[:40]:
            print("     " + l)
        print("     ... 共 %d 行差异" % len(diff))
        return 1
    print("  （未指定 --save / --check）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

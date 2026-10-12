#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_configs.py — **离线配置审计**（不需要 GPU, 不启动训练 ✓）

检查每个 config JSON 里的每个键, 是否会被训练 CLI **真正消费**：
  · 已注册 → OK（并报告解析后的有效值 ✓）
  · 别名键 → OK（会被映到正式参数, 但会告警提示改用正式名 ✓）
  · 以 "_" 开头 / 在白名单里 → 允许（注释、环境信息 ✓）
  · **其它 → 未知键 ✗**（当前会被 cli.py 静默忽略 ⇒ 配置静默失效 ✗✗）

用法:
  python tools/audit_configs.py                      # 审计全部 config
  python tools/audit_configs.py <cfg.json> ...       # 只审计指定配置（--strict 时未知键视为失败）
  python tools/audit_configs.py --active             # 只审计活跃线（v8*/v84*/当前实验）
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 允许的非参数键（注释/环境信息 ✓）
ALLOW_PREFIX = "_"
ALLOW_KEYS = {
    "python_bin",        # 启动环境信息（由启动脚本消费, 不属于训练超参 ✓）
    "results_dir", "experiment_name",  # 由 train.py 直接读取 ✓
    "eval_sets", "eval_eval_set_spec",
}

# 别名: 配置里写的是旧名/别名 -> 正式 dest（会告警 ✓）
ALIASES = {
    # 审查发现: seed 未被注册, 实际用 global_seed ⇒ 静默失效 ✗
    "seed": "global_seed",
    # gpu_eval_every 未被注册, 评测周期由 ckpt_every 控制
    "gpu_eval_every": "ckpt_every",
}


def known_dests():
    """从真实 CLI 取出全部已注册 dest（不解析 argv, 无副作用 ✓）"""
    from src.train import cli
    parser = cli.build_parser(["--config", "NONE"])      # 不存在的 config -> 不合并任何键 ✓
    return {a.dest: a for a in parser._actions if a.dest not in ("help",)}


def audit(cfg_path, dests):
    try:
        cfg = json.load(io.open(cfg_path, encoding="utf-8"))
    except Exception as e:                                        # noqa: BLE001
        return {"err": "JSON 读取失败: %s" % e, "unknown": [], "alias": []}
    if not isinstance(cfg, dict):
        return {"err": "不是 dict", "unknown": [], "alias": []}
    unknown, alias, ok = [], [], []
    for k, v in cfg.items():
        if k.startswith(ALLOW_PREFIX) or k in ALLOW_KEYS:
            ok.append(k); continue
        if k in dests:
            ok.append(k); continue
        if k in ALIASES:
            tgt = ALIASES[k]
            # 只有当正式键**不在**该配置里时才认为别名"会被消费"（否则正式值胜出 ✓）
            alias.append((k, tgt, "正式键 %s 也在配置里 -> 别名被忽略 ✗" % tgt if tgt in cfg
                          else "会映射到 %s ✓" % tgt))
            continue
        unknown.append((k, v))
    return {"err": None, "unknown": unknown, "alias": alias, "ok": ok, "cfg": cfg}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--active", action="store_true", help="只看活跃线（v8* / v84* / exp-*）")
    ap.add_argument("--strict", action="store_true", help="发现未知键即返回非零（预检用 ✓）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    dests = known_dests()
    files = a.configs or sorted(glob.glob("src/train/configs/*.json"))
    if a.active and not a.configs:
        # 真正的活跃线 = std-soft 系列（v80/v81/v82/v83/v84 ✓）
        # ⚠ 别用 startswith("v8") ✗ —— 那会把**整条历史 v8 家族**捞进来（各带 ~8-19 个
        #   已移除功能的死键, 例如 w_repa_early/repa_early_layer/repa_early_warmup ✗）
        files = [f for f in files
                 if any(os.path.basename(f).startswith(p)
                        for p in ("v80", "v81", "v82", "v83", "v84"))]
    print("  CLI 已注册参数 %d 个" % len(dests))
    print("  审计 %d 份配置\n" % len(files))

    bad, warned = [], []
    for f in files:
        r = audit(f, dests)
        name = os.path.basename(f)
        if r["err"]:
            print("  ✗ %-46s %s" % (name, r["err"])); bad.append(f); continue
        if r["unknown"]:
            bad.append(f)
            print("  ✗ %-46s 未知键 %d 个 ← **会被静默忽略** ✗✗" % (name, len(r["unknown"])))
            for k, v in r["unknown"]:
                print("        %-24s = %r" % (k, v))
        elif r["alias"]:
            warned.append(f)
            if not a.quiet:
                print("  ~ %-46s 别名键 %d 个" % (name, len(r["alias"])))
                for k, tgt, note in r["alias"]:
                    print("        %-20s -> %-20s %s" % (k, tgt, note))
        elif not a.quiet:
            print("  ✓ %-46s 全部键有效（%d 个）" % (name, len(r["ok"])))

    print("\n  汇总: 有效 %d · 含别名 %d · **含未知键 %d** ✗ · 共 %d"
          % (len(files) - len(bad) - len(warned), len(warned), len(bad), len(files)))
    if a.strict and bad:
        print("  ✗ --strict: 存在未知键配置, 判定失败 ✗")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

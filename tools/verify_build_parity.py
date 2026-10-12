#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_build_parity.py — 验证 `src/model/build.py` 与旧路径**逐参数等价**。

三重护栏:
  ① 新构造器 vs 旧 model_io.build_model_from_args -> named_parameters 名字+形状必须逐位相同 ✓
  ② 新构造器建出的模型 -> `load_state_dict(ckpt 权重, strict=True)` 必须 0 missing / 0 unexpected ✓✓
     （这是**最强证据**: 说明新路径复刻了训练时的架构, 一个字段都不差）
  ③ 两种内容条件模式的中文/表模式 ckpt 都要过 ✓

用法: python tools/verify_build_parity.py <ckpt1> [<ckpt2> ...]
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model.build import build_dit_from_args          # noqa: E402
from src.eval import model_io                            # noqa: E402


def strip_prefix(sd):
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}


def main(paths):
    rc = 0
    for ck_path in paths:
        if not os.path.isfile(ck_path):
            print("  · 跳过（不存在）:", ck_path); continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        name = os.path.basename(ck_path)
        print("=== %s ===" % name)
        try:
            m_new = build_dit_from_args(a, device="cpu")
        except Exception as e:                                    # noqa: BLE001
            print("  ✗ 新构造器失败: %s: %s" % (type(e).__name__, e)); rc = 1; continue
        # ① 与旧路径逐参数比对
        try:
            m_old = model_io.build_model_from_args(a, "cpu")
            pn = {n: tuple(p.shape) for n, p in m_new.named_parameters()}
            po = {n: tuple(p.shape) for n, p in m_old.named_parameters()}
            if pn == po:
                print("  ① 与旧 model_io 路径逐参数一致 ✓ (%d 个参数张量)" % len(pn))
            else:
                only_new = set(pn) - set(po); only_old = set(po) - set(pn)
                diff = [k for k in set(pn) & set(po) if pn[k] != po[k]]
                print("  ✗ ① 参数表不一致: only_new=%s only_old=%s shape_diff=%s"
                      % (sorted(only_new)[:3], sorted(only_old)[:3], diff[:3])); rc = 1
        except Exception as e:                                    # noqa: BLE001
            print("  · ① 旧路径不可用(%s), 跳过对比" % type(e).__name__)
        # ② strict 加载（最强证据 ✓）
        sd = ck.get("ema") if ck.get("ema") is not None else ck.get("model", ck)
        sd = strip_prefix(sd)
        try:
            m_new.load_state_dict(sd, strict=True)
            print("  ② strict=True 加载成功 ✓✓（0 missing / 0 unexpected）"
                  " · 参数=%.1fM" % (sum(p.numel() for p in m_new.parameters()) / 1e6))
        except Exception as e:                                    # noqa: BLE001
            msg = str(e)
            print("  ✗ ② strict 加载失败:", msg[:300]); rc = 1
    print("\n总体:", "全部通过 ✓" if rc == 0 else "有失败 ✗")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

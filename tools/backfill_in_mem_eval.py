#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对一批训练 ckpt 重跑「在训评测」, 用于回填失败/缺失的 eval 点。

背景 (2026-10-04): v53 在 step 10000..60000 的 eval 全部因 `_get_cache` 里
v53 remap 块**在 cache 命中时二次重映射**而 KeyError 失败 (已修)。修好后用本脚本
对每个 ckpt 补评; 结果写进同一 results_dir 的 summary/batch csv + poster, 并由
csv 的 (step, set) 去重 —— 已评过的点会自动跳过, 可安全重复运行。

用法:
  python tools/backfill_in_mem_eval.py \
      --ckpt-dir exp-std/runs_AB/<run>/checkpoints \
      --steps 10000,15000,20000      # 省略 = 全部 ckpt
"""
import argparse
import glob
import os
import sys

import torch

os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))
sys.path.insert(0, os.getcwd())

from src.eval.in_mem_eval import run_in_mem_eval                     # noqa: E402
from src.eval.model_io import build_model_from_args, apply_post_construction  # noqa: E402


class _AttrDict(dict):
    """Namespace 与 dict 双支持 (下游两种访问方式都有)。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--steps", default="", help="逗号分隔; 空 = 全部")
    ap.add_argument("--device", default="cuda")
    aa = ap.parse_args()

    dev = torch.device(aa.device)
    want = {int(s) for s in aa.steps.split(",") if s.strip()} if aa.steps else None
    cks = sorted(glob.glob(os.path.join(aa.ckpt_dir, "*.pt")))
    print(f"[backfill] {len(cks)} 个 ckpt, steps={sorted(want) if want else 'ALL'}", flush=True)

    for ck in cks:
        step_file = int(os.path.splitext(os.path.basename(ck))[0])
        if want is not None and step_file not in want:
            continue
        print(f"\n[backfill] ===== {os.path.basename(ck)} =====", flush=True)
        d = torch.load(ck, map_location="cpu", weights_only=False)
        raw = d.get("args", {})
        if hasattr(raw, "__dict__"):
            raw = vars(raw)
        a = _AttrDict(raw if isinstance(raw, dict) else {})

        model = build_model_from_args(a, dev)
        apply_post_construction(model, a, verbose=False)
        sd = _strip(d.get("ema") or d.get("delta") or d)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"[backfill] unexpected keys: {list(unexpected)[:5]}")
        if missing:
            print(f"[backfill] ⚠ missing keys ({len(missing)}): {list(missing)[:5]}", flush=True)
        model.eval()

        rd = str(getattr(a, "results_dir", "") or "exp-std/runs_AB")
        res = run_in_mem_eval(model, a, int(d.get("train_steps") or step_file), dev,
                              results_dir=rd, logger=print)
        print(f"[backfill] step {step_file} 完成, sets={sorted((res or {}).keys())}", flush=True)

        del model
        torch.cuda.empty_cache()
    print("[backfill] all done", flush=True)


if __name__ == "__main__":
    main()

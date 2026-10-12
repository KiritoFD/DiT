#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_char_table_intervention.py — 推理期"不给字表"的单变量干预。

为什么不用 `no_char_cond=True`: 那会**改变结构**（`cond_fusion.weight` 输入维度从 256+32+16
降到 32+16）⇒ strict 加载直接失配 ✗（已实测）。所以正确做法是**保持结构不变，只替换字表权重**
—— 这才是干净的单变量干预 ✓。

四种模式:
  baseline : 原样
  zero     : 字表权重全零       -> "字条件 = 常数(无信息)"
  null     : 全部行 = 最后一行（表尾的 null token，若存在）-> "训练支持的'无字'状态"
  other    : 全部行 = 另一个字的行 -> "字条件 = 恒定但错误的字"

用法:
  PY=/opt/conda/envs/cu121/bin/python
  $PY tools/audit_char_table_intervention.py --ckpt <ckpt> --vae-path <vae> \
      --sets "eval200fix:exp-std/csv/_audit_eval_orig.csv:3" --out-root assets/_audit/ct --modes baseline,zero,null,other
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(R, "src")):
    os.chdir(R)
sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.model_io import load_model_from_ckpt          # noqa: E402
from src.eval.in_mem_eval import run_in_mem_eval            # noqa: E402


def prep_args(args, cfg_defaults, sets):
    args.sde_gamma = 0.0
    args.flow_sampler = "heun"
    args.eval_steps = 50
    args.eval_cfg = 1.0
    args.eval_blend_alpha = 0.0
    args.in_mem_eval_batch = 1
    args.in_mem_eval_vae_batch = 1
    args.in_mem_eval_save_samples = True
    args.in_mem_eval_lpips = False
    args.in_mem_eval_sets = sets
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--modes", default="baseline,zero,null,other")
    ap.add_argument("--other-id", type=int, default=-1, help="other 模式用的字 id（默认取中间某行）")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]

    model, args = load_model_from_ckpt(a.ckpt, device=dev, use_ema=True, verbose=False)
    emb = getattr(model, "y_char_embedder", None)
    if emb is None:
        raise SystemExit("模型没有 y_char_embedder —— 该 ckpt 根本没字表")
    table = emb.embedding_table if hasattr(emb, "embedding_table") else emb
    W = table.weight
    n_rows, dim = W.shape
    orig = W.detach().clone()
    args.eval_vae_path = str(a.vae_path)
    prep_args(args, None, a.sets)
    print("[ct] 字表: rows=%d dim=%d use_char_cond=%s" % (n_rows, dim, getattr(model, "use_char_cond", None)))

    step = int(os.path.basename(a.ckpt).replace(".pt", ""))
    res = {}
    for mode in modes:
        with torch.no_grad():
            if mode == "baseline":
                W.copy_(orig)
            elif mode == "zero":
                W.zero_()
            elif mode == "null":
                W.copy_(orig[n_rows - 1:n_rows].expand(n_rows, dim))
            elif mode == "other":
                oid = a.other_id if a.other_id >= 0 else max(0, n_rows // 2)
                W.copy_(orig[oid:oid + 1].expand(n_rows, dim))
            else:
                raise SystemExit("未知模式 %s" % mode)
        out = os.path.join(a.out_root, mode)
        os.makedirs(out, exist_ok=True)
        print("[ct] === 模式 %s -> %s ===" % (mode, out), flush=True)
        r = run_in_mem_eval(model, args, step, dev, out, logger=print)
        res[mode] = r
        # 记录该模式下字表权重的统计（证明干预确实生效）
        res[mode]["_char_table"] = {"rows": int(n_rows), "dim": int(dim),
                                    "weight_norm": float(W.detach().norm()),
                                    "row0_norm": float(W[0].detach().norm())}
    json.dump(res, open(os.path.join(a.out_root, "intervention.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[ct] done ->", os.path.join(a.out_root, "intervention.json"))


if __name__ == "__main__":
    main()

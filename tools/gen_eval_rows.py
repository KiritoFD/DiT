#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_eval_rows.py — 为海报生成"新行"图集：eval/<name>/{idx}.png

背景: docs/04_experiments 的三张分级海报 (best/mid/worst) 的行 = 模型,
      列 = 该档代表字 (idx = eval200_fixed.csv 行号), 图来源 = 根目录 eval/<model>/{idx}.png。
      现有 15 行全是 cfg1.0/γ0；**没有任何 SDE 行, 也没有 v72** -> 本脚本补这两类行。

做法 (复用已验证的采样链路, 不自己重写 sampler):
  1. 从 eval200_fixed.csv 抽出指定 idx 子集 -> 临时 CSV
  2. run_in_mem_eval(..., save_samples=True) 采样并落盘
  3. 把 g{k}.png 改名到 eval/<name>/{idx_k}.png (k = 子集内序号)

用法:
  python tools/gen_eval_rows.py --name v68_sde \
      --ckpt _local_ckpt/v68_200k/0200000.pt --config src/train/configs/v68_aug_sp_c2ot.json \
      --vae-path pretrained_models/sd-vae-ft-ema --cfg 0.75 --gamma 0.5 --indices 100,170,...
"""

import argparse
import csv
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model import DiT_2Cond_models                      # noqa: E402
from src.eval.in_mem_eval import run_in_mem_eval            # noqa: E402

EVAL_CSV = "exp-std/csv/eval200_fixed.csv"
TABLES = "assets/triple_tables_best_minimal"


class A:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="输出到 eval/<name>/")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--indices", required=True, help="逗号分隔的 eval200_fixed 索引，或 @文件")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--step-tag", type=int, default=0)
    a = ap.parse_args()

    if a.indices.startswith("@"):
        idx = [int(x) for x in open(a.indices[1:], encoding="utf-8").read().strip().split(",") if x.strip()]
    else:
        idx = [int(x) for x in a.indices.split(",") if x.strip()]
    rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))
    sel = [rows[i] for i in idx]
    os.makedirs("_local_ckpt", exist_ok=True)
    sub = f"_local_ckpt/genrows_{a.name}.csv"
    with open(sub, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(sel)
    n = len(sel)
    print(f"[{a.name}] idx={idx}\n  子集 {n} 行 -> {sub}", flush=True)

    cfg = json.load(open(a.config, encoding="utf-8"))
    cfg.update({
        "eval_csv": EVAL_CSV, "in_mem_eval_sets": f"genrows:{sub}:{n}",
        "triple_table_prefix": TABLES + "/",
        "char_remap_json": f"{TABLES}/char_remap.json",
        "callig_remap_json": f"{TABLES}/callig_remap.json",
        "font_remap_json": f"{TABLES}/font_remap.json",
        "vae_path": a.vae_path, "eval_vae_path": a.vae_path,
        "eval_cfg": a.cfg, "sde_gamma": a.gamma, "eval_steps": a.steps,
        "in_mem_eval_batch": a.batch, "in_mem_eval_vae_batch": a.batch,
        "in_mem_eval_save_samples": True,
    })
    dev = a.device
    model = DiT_2Cond_models[cfg.get("model", "DiT-2Cond-Sp/2")](
        learn_sigma=False, norm_type=cfg.get("norm_type", "layer"), mlp_type=cfg.get("mlp_type", "gelu"),
        qk_norm=cfg.get("qk_norm", 0), rope=cfg.get("rope", 0),
        condition_fusion=cfg.get("condition_fusion", "factorized_cat"),
        cond_fusion_norm=cfg.get("cond_fusion_norm", "split"),
        num_calligraphers=cfg.get("num_calligraphers", 10), num_characters=cfg.get("num_characters", 4690),
        num_script_classes=cfg.get("num_script_classes", 3), use_script_cond=cfg.get("use_script_cond", True),
        char_embed_dim=cfg.get("char_embed_dim", 256), callig_embed_dim=cfg.get("callig_embed_dim", 32),
        script_embed_dim=cfg.get("script_embed_dim", 16),
        glyph_inject_layers=cfg.get("glyph_inject_layers", 0),
        cond_inject_at=cfg.get("cond_inject_at", "2,4,5,6"),
        cond_inject_scale=cfg.get("cond_inject_scale", True)).to(dev)
    import torch
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st = ck.get("ema") or ck.get("model") or ck
    if hasattr(st, "state_dict"):
        st = st.state_dict()
    st = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in st.items()}
    m, u = model.load_state_dict(st, strict=False)
    print(f"  [ckpt] {os.path.basename(a.ckpt)} missing={m} unexpected={u}", flush=True)
    model.eval()

    res_dir = f"_local_ckpt/gen_{a.name}_n{n}"
    # ★ 必须清空: run_in_mem_eval 有"(step,set) 已完成则跳过评测"的逻辑, 复用 results_dir
    #   会导致本轮**跳过采样**, 然后在拷贝阶段把上一轮的残留 g{k}.png 当成本轮结果复制
    #   （图被贴错标签, 且不报错）。2026-10-10 实测踩到, 故此处强制清场。
    shutil.rmtree(res_dir, ignore_errors=True)
    run_in_mem_eval(model, A(cfg), step=a.step_tag, device=dev, results_dir=res_dir, logger=print)

    src = os.path.join(res_dir, "eval_samples_ctrl", f"step{a.step_tag:07d}", "genrows")
    dst = os.path.join("eval", a.name)
    shutil.rmtree(dst, ignore_errors=True)      # ★ 同样清空目标, 防新旧混在一起
    os.makedirs(dst, exist_ok=True)
    ok = 0
    for k, i in enumerate(idx):
        p = os.path.join(src, f"g{k}.png")
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(dst, f"{i}.png"))
            ok += 1
        else:
            print(f"  ✗ 缺 g{k}.png (idx={i})")
    print(f"[{a.name}] 写入 {ok}/{n} 张 -> {dst}/  (cfg={a.cfg} gamma={a.gamma} steps={a.steps})")


if __name__ == "__main__":
    main()

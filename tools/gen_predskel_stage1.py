#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_predskel_stage1.py —— 用 **stage1 (v31)** 生成 predskel shards, 喂给 stage2。

与 tools/gen_predskel_dit.py 的区别 (那条线是旧的自写 trainer):
  · 模型 = src/train/train.py 的产物 (v31_stage1_skel), 用 src/eval/model_io 严格复刻构造
  · 采样 = **噪声起步 + 骨架当条件** (skel_as_glyph_cond / adaln 注入),
           而不是旧的"g 起步 + hide-g" bridge。两者的采样不能混用。
  · 去噪用模型自己的 flow 求解器 (create_diffusion_or_flow + ddim_sample_loop),
    与训练/评测同一份 flow_kwargs, 避免静默不一致。

产物与 GT 骨架 shards 同构: npz{latents(N,4,32,32) f16, img_ids(N)},
可直接喂 tools/run_skel_calibration.py 的 --pred-seen / --pred-strict。

用法:
  python tools/gen_predskel_stage1.py --ckpt <v31 ckpt> --set strict84 --out <dir>
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SETS = {
    "seen20": ("assets/eval_top10_seen_20.csv",
               "data/top10_style23/shards_std"),
    "strict84": ("assets/eval_top10_strict_subset84.csv",
                 "data/50k_v2_glyph15k/shards_std"),
}


def load_index(d):
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                idx[int(i)] = (f, j)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="stage1 的 train.py 格式 ckpt")
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warm-seed", type=int, default=0)
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    from src.eval import model_io
    from src.loss import create_diffusion_or_flow, flow_kwargs_from
    from src.utils.callig_script_map import map_callig_script

    model, args = model_io.load_model_from_ckpt(a.ckpt, device=dev, use_ema=True)
    model.eval()
    print(f"[1] 载入 {a.ckpt}")
    print(f"    model={getattr(args,'model',None)} "
          f"skel_as_glyph_cond={getattr(args,'skel_as_glyph_cond',None)} "
          f"inject={getattr(args,'glyph_inject_layers',None)}/"
          f"{getattr(args,'glyph_inject_mode',None)} "
          f"steps={a.steps}")
    assert getattr(args, "skel_as_glyph_cond", False), \
        "该 ckpt 不是 skel_as_glyph_cond 模型, 本脚本不适用"

    diff = create_diffusion_or_flow(timestep_respacing="",
                                    diffusion_type=getattr(args, "diffusion_type", "flow"),
                                    **flow_kwargs_from(args))
    print(f"[2] flow: {diff}")

    csvp, std_dir = SETS[a.set]
    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    sidx = load_index(std_dir)
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    print(f"[3] {csvp}: {len(rows)} 行; 标准骨架 {std_dir} ({len(sidx)} ids)")

    ids, gs, ys = [], [], []
    for r in rows:
        m = re.search(r"(\d+)\.png$", r.get("image_path", ""))
        if not m:
            continue
        i = int(m.group(1))
        if i not in sidx:
            continue
        pid = int(map_callig_script(int(r["calligrapher_id"]),
                                    int(r["script_id"]), csmap))
        f, j = sidx[i]
        with np.load(f) as z:
            gs.append(np.asarray(z["latents"][j], np.float32))
        ids.append(i)
        ys.append(pid)
    print(f"[4] 可生成 {len(ids)} 条")

    th.manual_seed(a.warm_seed)
    outs = []
    for s in range(0, len(ids), a.batch):
        b = slice(s, min(s + a.batch, len(ids)))
        g = th.from_numpy(np.stack(gs[b])).to(dev)
        y = th.tensor(ys[b], dtype=th.long, device=dev)
        shape = (g.shape[0], 4, 32, 32)
        with th.no_grad():
            z = diff.ddim_sample_loop(
                model, shape,
                model_kwargs={"y_callig": y,
                              "y_char": th.zeros_like(y),
                              "g": g},
                device=dev, progress=False)
        outs.append(z.float().cpu().numpy())
        print(f"    {min(s + a.batch, len(ids))}/{len(ids)}", flush=True)
    o = np.concatenate(outs).astype(np.float16)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "shard_00000.npz"),
                        latents=o, img_ids=np.array(ids, dtype=np.int64))
    print(f"[5] -> {a.out}/shard_00000.npz  n={len(ids)}  "
          f"ink占位 {float((o > 0).mean()):.4f}")


if __name__ == "__main__":
    main()

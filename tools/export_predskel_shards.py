#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/export_predskel_shards.py — 用 v31_stage1_skel @ 80k 全量离线生成预测骨架潜变量分片"""
import os, sys, glob, json, time, csv, re
import numpy as np
import pandas as pd
import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval import model_io
from src.eval.inference import build_diffusion
from src.utils.callig_script_map import load_callig_script_map, map_callig_script

def main():
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("=== 开始使用 v31_stage1_skel 全量导出训练集预测骨架 Latent 分片 ===")
    
    ckpt_p = "exp/v31_stage1_skel/20261001-005831-v31-stage1-skel/checkpoints/0080000.pt"
    if not os.path.exists(ckpt_p):
        ckpts = sorted(glob.glob("exp/v31_stage1_skel/*/checkpoints/*.pt"))
        ckpt_p = ckpts[-1]
    print(f"载入 Checkpoint: {ckpt_p}")
    
    model, args = model_io.load_model_from_ckpt(ckpt_p, device=dev, use_ema=True)
    model.eval()
    
    out_dir = "data/top10_style23/shards_predskel_v31"
    os.makedirs(out_dir, exist_ok=True)
    
    csmap = load_callig_script_map("assets/callig_script_id_map_top10.json")
    diff = build_diffusion(20, "flow") # 20 步 Euler 采样
    
    # 读入全量训练集 CSV 建立行映射
    csv_p = "assets/train_top10_style23.csv"
    rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))
    print(f"训练集总样本数: {len(rows)}")
    
    # 逐分片处理 (保持与 shards_std 100% 结构一致)
    std_shards = sorted(glob.glob("data/top10_style23/shards_std/shard_*.npz"))
    print(f"输入 shards_std 分片数: {len(std_shards)}")
    
    # 建立 id -> (pair_id, glyph_id) 映射
    id_to_cond = {}
    for r in rows:
        m = re.search(r"(\d+)\.png$", r["image_path"])
        if m:
            iid = int(m.group(1))
            cid = int(r["calligrapher_id"])
            sid = int(r.get("script_id", 0))
            pair_id = map_callig_script(cid, sid, csmap)
            gid = int(r.get("glyph_id", r.get("character_id", 0)))
            id_to_cond[iid] = (pair_id, gid)
            
    t_start = time.time()
    total_generated = 0
    batch_size = 128
    
    for s_idx, sp in enumerate(std_shards):
        fn = os.path.basename(sp)
        out_sp = os.path.join(out_dir, fn)
        with np.load(sp) as z:
            lats_std = z["latents"] # (M, 4, 32, 32), float16
            iids = z["img_ids"]     # (M,)
            names = z["names"] if "names" in z else None
            
        M = len(iids)
        print(f"\n[分片 {s_idx+1}/{len(std_shards)}] {fn}: 包含 {M} 条样本 ...")
        
        preds_list = []
        t0 = time.time()
        for i in range(0, M, batch_size):
            j = min(i + batch_size, M)
            cur_bs = j - i
            
            sub_std = th.from_numpy(lats_std[i:j].astype(np.float32)).to(dev)
            cur_iids = iids[i:j]
            cur_pairs = [id_to_cond[int(iid)][0] for iid in cur_iids]
            cur_glyphs = [id_to_cond[int(iid)][1] for iid in cur_iids]
            
            cond_tuples = list(zip(cur_pairs, cur_glyphs))
            noise = th.randn(cur_bs, 4, 32, 32, device=dev)
            
            # 使用 sample_latents 批量生成
            from src.eval.inference import sample_latents
            with th.no_grad():
                with th.autocast("cuda", dtype=th.bfloat16):
                    g_pred = sample_latents(model, diff, noise, cond_tuples, 1.0, cur_bs, dev, skel=sub_std, seed=0)
            preds_list.append(g_pred.cpu().numpy().astype(np.float16))
            
        all_preds = np.concatenate(preds_list, axis=0)
        save_dict = {"latents": all_preds, "img_ids": iids}
        if names is not None:
            save_dict["names"] = names
        np.savez_compressed(out_sp, **save_dict)
        
        total_generated += M
        print(f"  ✓ {fn} 生成完成! 耗时: {time.time()-t0:.1f}s | latents min={all_preds.min():.2f}, max={all_preds.max():.2f}, mean={all_preds.mean():.3f}, std={all_preds.std():.3f}")
        
    print("\n" + "=" * 65)
    print(f"🎉 全部 38,583 条预测骨架分片导出成功！总耗时: {time.time()-t_start:.1f}s")
    print(f"输出目录: {out_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()

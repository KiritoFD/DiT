# -*- coding: utf-8 -*-
"""
eval_sweep_12ch.py — 12ch 以来全部实验统一复评 (SIGSTOP 训练, in-mem, skel-iou).

对 6 个 run 的 best/latest ckpt:
  seen n=10 + strict n=50 (self-cond 0.5, cfg 0.7, Heun50) → decode →
  ssim / mse / skel_iou 逐样本 → samples 落盘 eval_samples_ctrl/step{N}/{set}/
输出: assets/eval_sweep_12ch.csv (run, step, set, n, ssim_mean, skel_iou_mean, mse_mean, ...)
显存: 训练 SIGSTOP (不释放显存但让出算力); eval 需 <3.4G → dit_batch 4 / vae_batch 4。
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

from src.model import DiT_2Cond_models
from src.eval.inference import (make_eval_cache, load_eval_vae, sample_latents,
                                sample_latents_self_cond, build_diffusion,
                                _mse, _ssim, _skel_iou)

RUNS = [
    ("S2_ref12ch", "assets/results/v11_pretrain_S2_ref12ch/20260912-125224-v11-pretrain-S2-ref12ch/checkpoints/0090000.pt"),
    ("S2_v8_aux02", "assets/results/v11_pretrain_S2_v8_aux02/20260912-172942-v11-pretrain-S2-v8-aux02/checkpoints/0120000.pt"),
    ("M432_sk3", "assets/results/v11_pretrain_M432_v8_sk3/20260913-023329-v11-pretrain-M432-v8-sk3/checkpoints/0152500.pt"),
    ("M432_adaln4_sym", "assets/results/v11_pretrain_M432_adaln4_sym/20260913-164820-v11-pretrain-M432-adaln4-sym/checkpoints/0042500.pt"),
    ("M432_sym_noise400k", "assets/results/v11_pretrain_M432_adaln4_sym_noise400k/20260914-022115-v11-pretrain-M432-adaln4-sym-noise400k/checkpoints/0270000.pt"),
    ("Sp2_base", "assets/results/v11_pretrain_Sp2_base/20260914-153302-v11-pretrain-Sp2-base/checkpoints/0017500.pt"),
]

SETS = [("seen", "assets/eval_seen_v10.csv", 10),
        ("strict", "assets/eval_fame3_strict_clean_v9.csv", 50)]


def strip(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def build_model(ck):
    ck = torch.load(ck, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    if not isinstance(a, dict):
        import argparse as _ap
        a = vars(a)
    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)), attn_impl=a.get("attn_impl", "sdpa"))
    n_aux = len([s for s in str(a.get("aux_latent_shards_dirs") or "").split(",") if s])
    model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
        num_calligraphers=int(a.get("num_calligraphers") or 1013),
        num_characters=int(a.get("num_characters") or 35130),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim") or 128),
        char_embed_dim=int(a.get("char_embed_dim") or 384),
        char_proj_mode=(a.get("char_proj_mode") or "mlp"),
        freeze_char_table=bool(a.get("freeze_char_table", False)),
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, cond_drop_which_glyph_prob=0.5,
        use_checkpoint=False, learn_sigma=False, use_glyph_cond=True,
        use_char_cond=not bool(a.get("no_char_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init") or 0.4), glyph_drop_prob=0.0,
        glyph_embedder_depth=int(a.get("glyph_embedder_depth") or 0),
        glyph_inject_layers=int(a.get("glyph_inject_layers") or 0),
        callig_style_attn=bool(a.get("callig_style_attn", False)),
        callig_n_style=int(a.get("callig_n_style") or 8),
        callig_spatial=bool(a.get("callig_spatial", False)),
        callig_spatial_rank=int(a.get("callig_spatial_rank") or 64),
        style_token_n=int(a.get("style_token_n") or 0),
        style_role_init=float(a.get("style_role_init") or 0.02),
        glyph_in_channels=4,
        in_channels=int(a.get("latent_channels") or 4) + 4 * n_aux,
        use_ids_char_embedder=bool(a.get("use_ids_char_embedder", False)),
        ids_file=a.get("ids_file"),
        use_std_dino_char_embedder=bool(a.get("use_std_dino_char_embedder", False)),
        std_dino_table_path=a.get("std_dino_table_path"),
        chars_per_script=int(a.get("chars_per_script") or 7026),
        glyph_inject_mode=a.get("glyph_inject_mode", "adaln"), **arch).to("cuda").eval()
    if a.get("freeze_callig_table"):
        model.y_callig_embedder.freeze_table()
    sd = strip(ck.get("ema") or ck.get("delta"))
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert len(unexp) == 0, f"unexpected {list(unexp)[:4]}"
    return model, a


def _run_one(run_name, ck_path, args, vae, diff, out_rows):
    step = int(os.path.basename(ck_path).split(".")[0])
    results_dir = os.path.dirname(os.path.dirname(os.path.dirname(ck_path)))
    model, a = build_model(ck_path)
    sf = float(a.get("vae_scaling_factor", 0.18215))
    shards = a.get("skel_latent_shards_dir") or "data/skel/std_skel1_latents_fame3_v8"
    cmap = None
    if a.get("callig_id_map") and os.path.exists(a["callig_id_map"]):
        from src.utils.callig_map import load_callig_id_map
        cmap, _ = load_callig_id_map(a["callig_id_map"])
    _yce = getattr(model, "y_char_embedder", None)
    n_char_tbl = (_yce.char_table.shape[0] if _yce is not None else 0)
    print(f"[sweep] {run_name}@{step}: model={a.get('model')} "
          f"num_callig={model.y_callig_embedder.embedding_table.weight.shape[0]} "
          f"num_char={n_char_tbl} "
          f"cmap={'yes' if cmap else 'NO'}", flush=True)
    for set_name, csvp, n in SETS:
        cache = make_eval_cache(csvp, None, None, 256, n, 8,
                                int(a.get("latent_channels") or 4), sf,
                                skel_latent_shards_dir=shards, callig_id_map=cmap)
        n = cache["n"]
        conds = cache["conds"]
        # clamp 防止 CUDA assert (历史 run 的 num_characters 可能小于 eval csv 的
        # glyph_id 体系; 越界样本的 embedding 语义不正确但保持进程存活)
        n_callig = model.y_callig_embedder.embedding_table.weight.shape[0] - 1
        n_char = (n_char_tbl - 1 if n_char_tbl else 10 ** 9)
        n_cid_over = sum(1 for c, _ in conds if c >= n_callig)
        n_g_over = sum(1 for _, g in conds if g >= n_char)
        if n_cid_over or n_g_over:
            print(f"[sweep]   WARNING: cid_over={n_cid_over} (>= {n_callig}), "
                  f"gly_over={n_g_over} (>= {n_char}) -> clamp", flush=True)
        conds = [(min(c, n_callig - 1), min(g, n_char - 1)) for c, g in conds]
        gmax = max(g for _, g in conds)
        print(f"[sweep]   {set_name}: gly_max={gmax} "
              f"missing_skel={cache.get('missing_skel')}", flush=True)
        t0 = time.time()
        self_cond = bool(a.get("eval_self_cond", False)) or ("sym" in run_name or "Sp2" in run_name)
        if self_cond:
            lat = sample_latents_self_cond(
                model, diff, cache["noise"], conds, 0.7,
                args.dit_batch, torch.device("cuda"),
                skel=cache["skels_latent"], seed=0, blend_alpha=0.5)
        else:
            lat = sample_latents(
                model, diff, cache["noise"], conds, 0.7,
                args.dit_batch, torch.device("cuda"),
                skel=cache["skels_latent"], seed=0)
        gts = (cache["gts"].to("cuda") + 1) / 2
        preds = torch.empty_like(gts)
        for i in range(0, n, args.vae_batch):
            j = min(i + args.vae_batch, n)
            _lat = lat[i:j].to("cuda")
            if _lat.shape[1] > 4:
                _lat = _lat[:, :4]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dec = vae.decode(_lat / sf).sample
            preds[i:j] = (dec.float().clamp(-1, 1) + 1) / 2
        pred_np = preds.cpu().numpy().transpose(0, 2, 3, 1)
        gt_np = gts.cpu().numpy().transpose(0, 2, 3, 1)
        sub = "g" if set_name == "seen" else set_name
        sd_dir = os.path.join(results_dir, "eval_samples_ctrl", f"step{step:07d}", sub)
        os.makedirs(sd_dir, exist_ok=True)
        from PIL import Image
        for i in range(n):
            Image.fromarray((pred_np[i] * 255).astype(np.uint8)).save(
                os.path.join(sd_dir, f"g{i}.png"))
            Image.fromarray((gt_np[i] * 255).astype(np.uint8)).save(
                os.path.join(sd_dir, f"gt{i}.png"))
        ssims, mses, ious = [], [], []
        for i in range(n):
            mses.append(_mse(pred_np[i], gt_np[i]))
            ssims.append(_ssim(pred_np[i], gt_np[i]))
            ious.append(_skel_iou(pred_np[i], gt_np[i]))
        out_rows.append({
            "run": run_name, "step": step, "set": set_name, "n": n,
            "ssim_mean": round(float(np.mean(ssims)), 4),
            "ssim_med": round(float(np.median(ssims)), 4),
            "skel_iou_mean": round(float(np.mean(ious)), 4),
            "skel_iou_med": round(float(np.median(ious)), 4),
            "mse_mean": round(float(np.mean(mses)), 5),
            "elapsed_s": round(time.time() - t0, 1)})
        print(f"[sweep] {run_name}@{step} {set_name}: ssim={np.mean(ssims):.4f} "
              f"skel_iou={np.mean(ious):.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-batch", type=int, default=4)
    ap.add_argument("--vae-batch", type=int, default=4)
    ap.add_argument("--out-csv", default="assets/eval_sweep_12ch.csv")
    args = ap.parse_args()

    # SIGSTOP 训练 (让出算力, 显存不释放)
    pids = subprocess.run(["pgrep", "-f", "train[.]py"], capture_output=True, text=True).stdout.split()
    for pid in pids:
        try:
            os.kill(int(pid), 19)
        except Exception:
            pass
    print(f"[sweep] SIGSTOP training pids: {len(pids)}", flush=True)
    time.sleep(2)

    out_rows = []
    try:
        vae = load_eval_vae(torch.device("cuda"), "data/pretrained/sd-vae-ft-ema")
        diff = build_diffusion(50, "flow")
        for run_name, ck_path in RUNS:
            if not os.path.exists(ck_path):
                print(f"[sweep] SKIP missing {ck_path}", flush=True)
                continue
            try:
                _run_one(run_name, ck_path, args, vae, diff, out_rows)
            except Exception as e:
                import traceback
                print(f"[sweep] {run_name} FAILED: {e}", flush=True)
                traceback.print_exc()
            torch.cuda.empty_cache()
    finally:
        for pid in pids:
            try:
                os.kill(int(pid), 18)
            except Exception:
                pass
        print("[sweep] SIGCONT training", flush=True)

    fields = ["run", "step", "set", "n", "ssim_mean", "ssim_med", "skel_iou_mean",
              "skel_iou_med", "mse_mean", "elapsed_s"]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[sweep] written {args.out_csv} ({len(out_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()

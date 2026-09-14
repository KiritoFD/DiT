# -*- coding: utf-8 -*-
"""make_posters_12ch.py — 12ch 以来全部 run 的 poster (标准字行 + per-ckpt gen + GT + 结构图)."""
import os
import sys
import traceback

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.in_mem_eval import render_poster

DIR_PAT = {
    "S2_ref12ch": "v11_pretrain_S2_ref12ch",
    "S2_v8_aux02": "v11_pretrain_S2_v8_aux02",
    "M432_sk3": "v11_pretrain_M432_v8_sk3",
    "M432_adaln4_sym": "v11_pretrain_M432_adaln4_sym",
    "M432_sym_noise400k": "v11_pretrain_M432_adaln4_sym_noise400k",
    "Sp2_base": "v11_pretrain_Sp2_base",
}

for run, pat in DIR_PAT.items():
    rd = f"assets/results/{pat}"
    if not os.path.isdir(os.path.join(rd, "eval_samples_ctrl")):
        print(f"skip {run}: no samples")
        continue
    # 标准字输入 (g 条件): decode skels_latent -> input_g/g{i}.png (幂等)
    try:
        import csv as _csv
        import torch as _torch
        import glob as _glob
        from src.eval.inference import load_eval_vae, make_eval_cache
        ck = sorted(_glob.glob(f"{rd}/*/checkpoints/[0-9]*.pt"),
                    key=lambda p: int(os.path.basename(p).split(".")[0]))[-1]
        ckd = _torch.load(ck, map_location="cpu", weights_only=False)
        a = ckd.get("args", {})
        if not isinstance(a, dict):
            import argparse as _ap
            a = vars(a)
        cmap = None
        if a.get("callig_id_map") and os.path.exists(a["callig_id_map"]):
            from src.utils.callig_map import load_callig_id_map
            cmap, _ = load_callig_id_map(a["callig_id_map"])
        vae = load_eval_vae(_torch.device("cuda"), "data/pretrained/sd-vae-ft-ema")
        sf = float(a.get("vae_scaling_factor", 0.18215))
        shards = a.get("skel_latent_shards_dir") or ""
        for set_name, csvp, n in (("seen", "assets/eval_seen_v10.csv", 10),
                                  ("strict", "assets/eval_fame3_strict_clean_v9.csv", 50)):
            cache = make_eval_cache(csvp, None, None, 256, n, 8,
                                    int(a.get("latent_channels") or 4), sf,
                                    skel_latent_shards_dir=shards, callig_id_map=cmap)
            from src.eval.in_mem_eval import save_input_g
            if cache.get("skels_latent") is not None:
                out = save_input_g(rd, set_name, cache["skels_latent"], vae, sf)
                print(f"{run} {set_name} input_g: {out}", flush=True)
        del vae
        _torch.cuda.empty_cache()
    except Exception:
        import traceback
        print(f"{run} input_g FAILED", flush=True)
        traceback.print_exc()
    for set_name in ("seen", "strict"):
        try:
            p = render_poster(rd, set_name)
            print(f"{run} {set_name}: {p}", flush=True)
        except Exception:
            print(f"{run} {set_name} FAILED", flush=True)
            traceback.print_exc()

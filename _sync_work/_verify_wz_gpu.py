# -*- coding: utf-8 -*-
"""_verify_wz_gpu.py — 单图验证: 修复后推理是否正确 (白底黑字).

对照三组:
  old/    zero_white=False  (修复前的行为 -> 应发黄/发黑)
  new/    zero_white=True   (修复后 -> 应白底黑字)
  gcond/  g 条件 latent 直接 decode (未减白底, 应白底)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--cfg", type=float, default=2.0)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--out", default="_otout3")
args = ap.parse_args()

from src.model import DiT_2Cond_models                      # noqa: E402
from src.eval.inference import (make_eval_cache, load_eval_vae,   # noqa: E402
                                build_diffusion, sample_latents, decode_and_save)
from src.loss import flow_kwargs_from                       # noqa: E402

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a = ck.get("args", {}) or {}
if isinstance(a, argparse.Namespace):
    a = vars(a)


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


dev = torch.device("cuda")
arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
            qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
            rope_theta=float(a.get("rope_theta", 100.0)),
            attn_impl=a.get("attn_impl", "sdpa"))
use_g = bool(a.get("w_glyph_cond", False) or a.get("skel_as_glyph_cond", False))
_aux = [x for x in (a.get("aux_latent_shards_dirs") or "").split(",") if x.strip()]
in_ch = 4 + 4 * len(_aux)
print(f"[arch] in_channels={in_ch} (aux groups={len(_aux)})", flush=True)
model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
    in_channels=in_ch,
    num_calligraphers=int(a.get("num_calligraphers") or 1013),
    num_characters=int(a.get("num_characters") or 35130),
    condition_fusion=a.get("condition_fusion") or "factorized_add",
    callig_embed_dim=int(a.get("callig_embed_dim") or 128),
    char_embed_dim=int(a.get("char_embed_dim") or 384),
    char_proj_mode=a.get("char_proj_mode") or "mlp",
    freeze_char_table=bool(a.get("freeze_char_table", True)),
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, cond_drop_which_glyph_prob=0.5,
    use_checkpoint=False, learn_sigma=False,
    use_glyph_cond=use_g,
    use_char_cond=not bool(a.get("no_char_cond", False)),
    use_std_dino_char_embedder=bool(a.get("use_std_dino_char_embedder", False)),
    std_dino_table_path=a.get("std_dino_table_path"),
    glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
    glyph_drop_prob=float(a.get("glyph_drop_prob", 0.0)),
    glyph_embedder_depth=int(a.get("glyph_embedder_depth", 0)),
    glyph_inject_layers=int(a.get("glyph_inject_layers", 0)),
    callig_style_attn=bool(a.get("callig_style_attn", False)),
    callig_n_style=int(a.get("callig_n_style", 8)),
    style_token_n=int(a.get("style_token_n", 0)),
    style_role_init=float(a.get("style_role_init", 0.02)),
    glyph_inject_mode=a.get("glyph_inject_mode", "adaln"), **arch)
if a.get("freeze_callig_table"):
    model.y_callig_embedder.freeze_table()
sd = _strip(ck.get("ema") or ck.get("model") or ck)
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"[load] missing={len(miss)} unexpected={len(unexp)}", flush=True)
assert len(unexp) == 0, f"unexpected={sorted(unexp)[:8]}"
model = model.to(dev).eval()

csv = a.get("gpu_eval_csv") or a.get("eval_csv") or a.get("data_csv")
cmap = None
_p = a.get("callig_id_map")
if _p and os.path.exists(_p):
    # 用与训练/eval 完全相同的解析 (文件含 num_calligraphers 等元数据键)
    from src.utils.callig_map import load_callig_id_map
    cmap, _meta = load_callig_id_map(_p)
shards = a.get("skel_latent_shards_dir") or None
print(f"[cache] csv={csv} shards={shards} cmap={'yes' if cmap else 'no'}", flush=True)
cache = make_eval_cache(csv, None, None, 256, args.n, 8, 4, 0.18215,
                        skel_latent_shards_dir=shards, callig_id_map=cmap)
print(f"[cache] n={cache['n']} skels_latent={None if cache.get('skels_latent') is None else tuple(cache['skels_latent'].shape)}", flush=True)
if cache.get("skels_latent") is not None:
    sl = cache["skels_latent"].float()
    print(f"[cache] skels_latent mean={sl.mean():.4f} per-ch="
          f"{np.round(sl.mean(dim=(0,2,3)).numpy(),4)}", flush=True)

diff = build_diffusion(str(args.steps), diffusion_type=a.get("diffusion_type", "flow"),
                       flow_kwargs=flow_kwargs_from(a))
lat = sample_latents(model, diff, cache["noise"], cache["conds"],
                     args.cfg, 16, dev, skel=cache["skels_latent"], seed=0)
print(f"[sample] lat shape={tuple(lat.shape)} mean={lat.float().mean():.4f} "
      f"per-ch={np.round(lat.float().mean(dim=(0,2,3)).numpy(),4)}", flush=True)

vae = load_eval_vae(dev, "data/pretrained/sd-vae-ft-ema")
os.makedirs(args.out, exist_ok=True)
for tag, zw in (("old", False), ("new", True)):
    d = os.path.join(args.out, tag)
    os.makedirs(d, exist_ok=True)
    decode_and_save(vae, lat, 0.18215, d, "g", gts=cache["gts"],
                    vae_batch=4, idx_offset=0, zero_white=zw)
    im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(os.path.join(d, "g0.png")))
    print(f"[decode {tag}] zero_white={zw} -> g0.png RGB mean={im.reshape(-1,3).mean(0)}", flush=True)

# g 条件自身 decode (未减白底)
if cache.get("skels_latent") is not None:
    d = os.path.join(args.out, "gcond")
    os.makedirs(d, exist_ok=True)
    decode_and_save(vae, cache["skels_latent"], 0.18215, d, "g",
                    vae_batch=4, idx_offset=0, zero_white=False)
    im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(os.path.join(d, "g0.png")))
    print(f"[decode gcond] -> g0.png RGB mean={im.reshape(-1,3).mean(0)}", flush=True)

print(f"[done] -> {args.out}")

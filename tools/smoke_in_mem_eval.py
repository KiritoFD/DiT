# -*- coding: utf-8 -*-
"""smoke_in_mem_eval.py - end-to-end in-mem eval on 4 samples (concurrent with training)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model import DiT_2Cond_models
from src.eval.in_mem_eval import run_in_mem_eval

CKPT = "assets/results/v11_pretrain_M432_adaln4_sym/20260913-164820-v11-pretrain-M432-adaln4-sym/checkpoints/0042500.pt"
RESULTS = "/tmp/in_mem_eval_smoke"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck.get("args", {})
if not isinstance(a, dict):
    import argparse as _ap
    a = vars(a) if isinstance(a, __import__("argparse").Namespace) else {}

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

sd = ck.get("ema") or ck.get("delta")
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
miss, unexp = model.load_state_dict(sd, strict=False)
assert len(unexp) == 0, f"unexpected keys: {list(unexp)[:5]}"

class A:
    pass
args = A()
for k in ("vae_scaling_factor", "latent_channels", "skel_latent_shards_dir",
          "gpu_eval_img_root", "img_root", "callig_id_map", "eval_cfg", "eval_steps",
          "diffusion_type", "eval_self_cond", "eval_blend_alpha",
          "in_mem_eval_batch", "in_mem_eval_vae_batch", "in_mem_eval_save_samples"):
    setattr(args, k, a.get(k, None) if a.get(k) is not None else getattr(args, k, None))
args.vae_scaling_factor = float(a.get("vae_scaling_factor", 0.18215))
args.latent_channels = int(a.get("latent_channels", 4))
args.eval_cfg = 0.7
args.eval_steps = 50
args.gpu_eval_img_root = a.get("gpu_eval_img_root", "data/imgs/final_imgs_fame_e")
args.diffusion_type = a.get("diffusion_type", "flow")
args.eval_self_cond = True
args.eval_blend_alpha = 0.5
args.in_mem_eval_batch = 4
args.in_mem_eval_vae_batch = 4
args.in_mem_eval_save_samples = True

res = run_in_mem_eval(
    model, args, step=42500, device=torch.device("cuda"), results_dir=RESULTS,
    sets=[("seen", "assets/eval_seen_v10.csv", 4)])
print("RESULT:", res)
print("files:", sorted(os.listdir(os.path.join(RESULTS, "eval_samples_ctrl", "step0425000", "g")))[:4]
      if os.path.isdir(os.path.join(RESULTS, "eval_samples_ctrl", "step0425000", "g")) else "NO PNG DIR")
print(open(os.path.join(RESULTS, "eval_stdskel_summary.csv")).read())
print("SMOKE PASS")

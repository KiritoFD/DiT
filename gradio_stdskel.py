# -*- coding: utf-8 -*-
"""gradio_stdskel.py — std-skel-g DiT (当前最佳 c41x_cos_e) 极简 gradio 前端.

CPU 推理, base conda env (gradio 3.50.2, torch 1.13). xattn 需要的
F.scaled_dot_product_attention 在 torch<2.0 缺失, 这里补一个等价 shim.

用法:
  /opt/conda/bin/python gradio_stdskel.py --device cpu --share
"""
import os
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")
import sys
import csv
import json
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# ---- torch<2.0: 补 scaled_dot_product_attention (xattn 注入需要) ----
if not hasattr(F, "scaled_dot_product_attention"):
    def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        scale = q.shape[-1] ** -0.5
        a = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            a = a + attn_mask
        a = a.softmax(dim=-1)
        if dropout_p and dropout_p > 0:
            a = F.dropout(a, p=dropout_p)
        return a @ v
    F.scaled_dot_product_attention = _sdpa
    torch.nn.functional.scaled_dot_product_attention = _sdpa

from PIL import Image
import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="assets/results/v10b_stdskel_fame3_c41x_cos_e/"
                                  "20260909-193018-v10b-stdskel-fame3-c41x-cos-e/checkpoints/0390000.pt")
ap.add_argument("--bank", default="_sync_work/data/skel/skel_bank_std1_v8.npz",
                help="keys='script|char', latents=(N,4,32,32) 标准字形骨架 latent")
ap.add_argument("--train-csv", default="assets/train_fame3_clean_v8.csv")
ap.add_argument("--callig-map", default="assets/callig_id_map.json")
ap.add_argument("--vae-path", default="data/pretrained/sd-vae-ft-ema")
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=0.7)
ap.add_argument("--port", type=int, default=7863)
ap.add_argument("--share", action="store_true")
ap.add_argument("--smoke", action="store_true", help="只跑一次生成并退出 (不上 gradio)")
args = ap.parse_args()

dev = torch.device(args.device)
print(f"[load] device={dev} ckpt={args.ckpt}", flush=True)

from src.model import DiT_2Cond_models  # noqa: E402
from src.eval.inference import load_eval_vae, sample_latents, build_diffusion  # noqa: E402
from src.utils.callig_map import load_callig_id_map  # noqa: E402

# ---- 模型 (架构取自 ckpt args, 与 tools/eval/eval_stdskel_batch.py 一致) ----
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a = ck.get("args", {})
if not isinstance(a, dict):
    a = vars(a) if hasattr(a, "__dict__") else {}
_arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
             qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
             rope_theta=float(a.get("rope_theta", 100.0)), attn_impl="eager")
model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
    num_calligraphers=int(a.get("num_calligraphers") or 41),
    num_characters=int(a.get("num_characters") or 35130),
    condition_fusion=a.get("condition_fusion", "factorized_add"),
    callig_embed_dim=int(a.get("callig_embed_dim") or 128),
    char_embed_dim=int(a.get("char_embed_dim") or 384),
    char_proj_mode=(a.get("char_proj_mode") or "mlp"),
    freeze_char_table=bool(a.get("freeze_char_table", False)),
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, cond_drop_which_glyph_prob=0.5,
    use_checkpoint=False, learn_sigma=False,
    use_glyph_cond=True, use_char_cond=not bool(a.get("no_char_cond", False)),
    glyph_scale_init=float(a.get("glyph_scale_init") or 0.6), glyph_drop_prob=0.0,
    glyph_embedder_depth=int(a.get("glyph_embedder_depth") or 0),
    glyph_inject_layers=int(a.get("glyph_inject_layers") or 0),
    callig_style_attn=bool(a.get("callig_style_attn", False)),
    callig_n_style=int(a.get("callig_n_style") or 8),
    callig_spatial=bool(a.get("callig_spatial", False)),
    callig_spatial_rank=int(a.get("callig_spatial_rank") or 64),
    style_token_n=int(a.get("style_token_n") or 0),
    style_role_init=float(a.get("style_role_init") or 0.02),
    glyph_in_channels=4,
    in_channels=(int(a.get("latent_channels") or 4)
                 + 4 * len([s for s in str(a.get("aux_latent_shards_dirs") or "").split(",") if s])),
    glyph_inject_mode=a.get("glyph_inject_mode", "adaln"), **_arch)
if a.get("freeze_callig_table"):
    model.y_callig_embedder.freeze_table()
sd = ck.get("ema") or ck.get("model")
if isinstance(sd, dict):
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    _ms, _us = model.load_state_dict(sd, strict=False)
    print(f"[load] weights missing={len(_ms)} unexpected={len(_us)}", flush=True)
model = model.to(dev).eval()

vae = load_eval_vae(dev, args.vae_path)
sf = float(a.get("vae_scaling_factor", 0.18215))
LC = int(a.get("latent_channels") or 4)
LS = 32
_diff_cache = {}

# ---- g 条件库 (标准字形骨架 latent, key="script|char") ----
bank = {}
if os.path.isfile(args.bank):
    z = np.load(args.bank)
    keys = [str(k) for k in z["keys"]]
    lat = z["latents"]
    bank = {k: lat[i] for i, k in enumerate(keys)}
    print(f"[bank] {len(bank)} entries <- {args.bank}", flush=True)
else:
    print(f"[WARN] bank 缺失 {args.bank}; 将无法生成 (需先 build)", flush=True)

# ---- 书家列表 (name -> 连续 idx) ----
cmap, n_callig = load_callig_id_map(args.callig_map)
raw2idx = {}
name_of = {}
with open(args.train_csv, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        raw = int(r["calligrapher_id"])
        raw2idx[raw] = cmap.get(raw, raw)
        name_of[r["calligrapher"]] = raw
CALLIG_NAMES = sorted(name_of.keys())
print(f"[callig] {len(CALLIG_NAMES)} names; num_callig={n_callig}", flush=True)


def _diff(steps):
    if steps not in _diff_cache:
        _diff_cache[steps] = build_diffusion(steps, "flow")
    return _diff_cache[steps]


@torch.no_grad()
def generate(char, callig_name, script, steps, cfg, seed):
    char = (char or "").strip()
    if len(char) != 1:
        return None, "请输入单个汉字"
    key = f"{script}|{char}"
    if key not in bank:
        return None, f"'{char}'({script}) 不在库存 (仅支持训练集出现过的字)"
    g = torch.from_numpy(np.asarray(bank[key], dtype=np.float32))[None]  # (1,4,32,32)
    gen = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(1, LC, LS, LS, generator=gen)
    raw = name_of.get(callig_name)
    if raw is None:
        return None, f"未知书家 {callig_name}"
    conds = [(raw2idx[raw], 0)]
    lat = sample_latents(model, _diff(int(steps)), noise, conds, float(cfg), 1, dev, skel=g)
    lat4 = lat[:, :LC]
    rec = vae.decode(lat4.to(dev) / sf).sample.float().cpu()
    img = ((rec[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype("uint8")), \
        f"OK char={char} script={script} callig={callig_name} steps={steps} cfg={cfg}"


if args.smoke:
    import time as _t
    _t0 = _t.time()
    _img, _msg = generate("阜", CALLIG_NAMES[0], "楷", args.steps, args.cfg, 0)
    print(f"[smoke] {_msg} elapsed={_t.time()-_t0:.1f}s", flush=True)
    if _img is not None:
        os.makedirs("_sync_work", exist_ok=True)
        _img.save("_sync_work/smoke_stdskel.png")
        print("[smoke] saved _sync_work/smoke_stdskel.png", flush=True)
    sys.exit(0)


with gr.Blocks(title="fame 书法生成 (std-skel-g, CPU)") as demo:
    gr.Markdown("## fame 书法生成 — std-skel-g DiT (c41x_cos_e@390k)\n"
                "CPU 推理; g=标准字形骨架 latent; 仅支持训练集出现过的字。")
    with gr.Row():
        char_in = gr.Textbox(label="汉字 (单字)", value="阜")
        callig_in = gr.Dropdown(CALLIG_NAMES, label="书家",
                                value=CALLIG_NAMES[0] if CALLIG_NAMES else None)
        script_in = gr.Radio(["楷", "行", "隶"], label="书体", value="楷")
    with gr.Row():
        steps_in = gr.Slider(10, 100, value=args.steps, step=10, label="采样步数")
        cfg_in = gr.Slider(0.0, 4.0, value=args.cfg, step=0.1, label="CFG")
        seed_in = gr.Number(value=0, label="seed", precision=0)
    btn = gr.Button("生成", variant="primary")
    img_out = gr.Image(label="生成结果", type="pil")
    msg = gr.Textbox(label="状态")
    btn.click(generate, [char_in, callig_in, script_in, steps_in, cfg_in, seed_in],
              [img_out, msg])

print("[load] done, launching gradio ...", flush=True)
demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

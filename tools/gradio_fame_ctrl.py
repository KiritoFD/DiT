# -*- coding: utf-8 -*-
"""
gradio_fame_ctrl.py — fame ControlNet 书法生成前端.

逻辑:
  * 训练过的字 (script,char ∈ fame 训练集) → 训练集骨架库直出
  * 没训过的字 → 提示 ZERO-SHOT, 用标准字体骨架 (楷simkai/行STXINGKA/隶SIMLI)
  * 每次生成: 大图 + 使用骨架 + 专属徽标; 相近 GT (同字训练样本, 最多4张)
  * 会话内所有生成结果累积在底部 gallery

用法: python tools/gradio_fame_ctrl.py   (tmux)  → http://<host>:7861
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
import csv
import glob
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from scipy.ndimage import binary_dilation, generate_binary_structure
try:
    from skimage.morphology import skeletonize
except ImportError:
    from scipy.ndimage import binary_erosion, generate_binary_structure
    def skeletonize(b):
        skel = np.zeros_like(b); img = b.copy()
        st = generate_binary_structure(2, 2)
        while img.any():
            er = binary_erosion(img, structure=st)
            skel |= img & ~er; img = er
        return skel

def dil3(b):
    return binary_dilation(b, structure=generate_binary_structure(2, 2), iterations=3)

SCRIPT_ID = {"楷": 0, "行": 3, "隶": 4, "草": 2, "篆": 1, "六体": 5}
SCRIPT_FONT = {"楷": "/tmp/simkai.ttf", "行": "/tmp/STXINGKA.TTF", "隶": "/tmp/SIMLI.TTF",
               "草": "/tmp/simkai.ttf", "篆": "/tmp/simkai.ttf", "六体": "/tmp/simkai.ttf"}
NUM_CHARACTERS = 7026

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=7861)
p.add_argument("--main-ckpt", default="5script/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/0030000.pt")
p.add_argument("--ctrl-ckpt", default="5script/results/ctrl_fame_v2/20260830-094156-fame-ctrl-skel-v2/checkpoints/0050000.pt")
args = p.parse_args()

import torch
from diffusers.models import AutoencoderKL
from src.model.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import build_diffusion, load_eval_vae

dev = torch.device("cuda")
print("[load] main ...", flush=True)
main = load_main_model("DiT-2Cond-S/2", args.main_ckpt, device=dev,
                       num_calligraphers=1013, num_characters=35130,
                       condition_fusion="factorized_add", callig_embed_dim=128,
                       char_embed_dim=384, char_proj_mode="ln_only",
                       freeze_char_table=True)
main.eval()
print("[load] ctrl ...", flush=True)
ctrl = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True).to(dev)
ck = torch.load(args.ctrl_ckpt, map_location="cpu", weights_only=False)
sd = {k: v for k, v in (ck.get("ema") or ck.get("ctrl")).items() if not k.startswith("main.")}
ctrl.load_state_dict(sd, strict=False)
ctrl.eval()
vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
diff = build_diffusion(50, "flow")
print("[load] done", flush=True)

# ---- 数据: fame 训练表 + 骨架库 ----
rows = list(csv.DictReader(open("5script/train_fame.csv", encoding="utf-8")))
by_pair = {}
for r in rows:
    by_pair.setdefault((r["script"], r["character"]), []).append(r)
b1 = np.load("skel_bank_train.npz")
BANK_TRAIN = {k: b1["latents"][i] for i, k in enumerate(b1["keys"])}
b2 = np.load("skel_bank_std.npz")
BANK_STD = {k: b2["latents"][i] for i, k in enumerate(b2["keys"])}
CALLIGS = sorted({r["calligrapher"] for r in rows})
CALLIG_ID = {}
for r in rows:
    CALLIG_ID.setdefault(r["calligrapher"], int(r["calligrapher_id"]))
CHAR_IDS = {}
for r in rows:
    CHAR_IDS.setdefault((r["script"], r["character"]), int(r["character_id"]))
SCRIPT_OF_PAIR = {(r["script"], r["character"]): r["script"] for r in rows}
_next_cid = [7000]
def char_id_of(script, ch):
    if (script, ch) in CHAR_IDS:
        return CHAR_IDS[(script, ch)], True
    # 未收录: 分配一次 (embedding 为 DINO-fill/随机, 结果仅供参考)
    key = ("*", ch)
    if key not in CHAR_IDS:
        while (_next_cid[0] in set(CHAR_IDS.values())) or _next_cid[0] >= NUM_CHARACTERS:
            _next_cid[0] += 1
        CHAR_IDS[key] = _next_cid[0]
    return CHAR_IDS[key], False

_f_cache = {}
def font_of(script, size):
    key = (script, size)
    if key not in _f_cache:
        _f_cache[key] = ImageFont.truetype(SCRIPT_FONT.get(script, "/tmp/simkai.ttf"), size)
    return _f_cache[key]

def std_skel_latent(script, ch):
    img = Image.new("L", (256, 256), 255)
    d = ImageDraw.Draw(img)
    d.text((128, 128), ch, font=font_of(script, 200), fill=0, anchor="mm")
    a = np.asarray(img)
    if (a < 250).sum() < 10:
        return None
    sk = np.where(dil3(skeletonize(a < 127)), 0, 255).astype("uint8")
    with torch.no_grad():
        x = torch.from_numpy(sk.astype(np.float32) / 255. * 2 - 1)[None, None].repeat(1, 3, 1, 1).to(dev)
        return (vae.encode(x).latent_dist.mode() * 0.18215)[0].half().cpu().numpy(), sk

def decode_lat(lat):
    with torch.no_grad():
        rec = vae.decode(torch.from_numpy(lat.astype(np.float32))[None].to(dev) / 0.18215).sample
    return ((rec[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()

def to_img(arr, size=256):
    return Image.fromarray((arr * 255).astype("uint8")).resize((size, size))

import gradio as gr
SESSION = []   # (char, script, callig, badge, PIL)

def generate(ch, callig, script):
    ch = (ch or "").strip()
    if len(ch) != 1:
        return None, None, None, None, "请输入单个汉字", SESSION_SNAPSHOT()
    known_pair = (script, ch) in by_pair
    known_char = any(c == ch for (s, c) in by_pair)
    trained = known_pair and (script + "|" + ch) in BANK_TRAIN
    callig_id = CALLIG_ID.get(callig)
    if callig_id is None:
        return None, None, None, None, f"书家 {callig} 不在 fame 训练集", SESSION_SNAPSHOT()
    cid, _ = char_id_of(script, ch)
    gid = SCRIPT_ID[script] * NUM_CHARACTERS + cid
    sk_note = ""
    # ---- 骨架条件选择 ----
    if trained:
        sk_lat = BANK_TRAIN[script + "|" + ch]
        badge = f"训练过 · 直出（骨架: 训练集骨架库）"
    else:
        sk = std_skel_latent(script, ch)
        if sk is None:
            return None, None, None, None, f"字体不包含「{ch}」，无法生成标准字形骨架", SESSION_SNAPSHOT()
        sk_lat, sk_img = sk
        badge = "ZERO-SHOT · 标准字形骨架（该字未在 fame 训练，结果特殊标注）"
        sk_note = sk_img
    lat = sample(sk_lat, callig_id, gid)
    out = decode_lat(lat)
    # ---- 相近 GT: 同字同书体的训练样本, 最多 4 ----
    gts = [to_img(np.asarray(Image.open(r["image_path"]).convert("RGB")), 128)
           for r in by_pair.get((script, ch), [])[:4]]
    if not gts:
        gts = [to_img(np.asarray(Image.open(r["image_path"]).convert("RGB")), 128)
               for rr in rows if rr["character"] == ch for r in [rr]][:4]
        if gts:
            gts = gts[:4]
    gt_note = f"相近 GT（{script} {ch}）" if gts else "无 GT（该字未收录）"
    # ---- 面板 ----
    panel = Image.new("RGB", (256 * 2 + 30, 256 + 60), "white")
    d = ImageDraw.Draw(panel)
    panel.paste(to_img(out), (10, 40))
    if sk_note is not None and sk_note is not False and isinstance(sk_note, np.ndarray):
        panel.paste(to_img(sk_note), (256 + 20, 40))
        d.text((256 + 20, 15), "skeleton used", fill="#888888")
    d.text((10, 15), badge, fill="#CC0000" if "ZERO-SHOT" in badge else "#227722")
    d.text((10, 256 + 45), f"{script} {ch} · {callig}", fill="black")
    gt_imgs = [to_img(g, 256) if False else g for g in gts]
    # ---- gallery 项 ----
    gal_img = to_img(out)
    SESSION.append((gal_img, f"{script}·{ch}·{callig} [{badge.split('·')[0]}]"))
    note = f"{badge} | {gt_note}"
    return to_img(out), (sk_note if isinstance(sk_note, np.ndarray) else None), gts, panel, note, SESSION_SNAPSHOT()

def SESSION_SNAPSHOT():
    return [(im, cap) for im, cap in SESSION]

def sample(sk_lat, callig_id, gid):
    yc = torch.tensor([callig_id], device=dev)
    yh = torch.tensor([gid], device=dev)
    sk = torch.from_numpy(sk_lat.astype(np.float32))[None]
    lat = sample_latents(ctrl, diff, 1, [(callig_id, gid)], 0.7, 1, dev,
                         skel=sk, y=(yc, yh), seed=hash((callig_id, gid)) % 100000)
    return lat[0].numpy()

# 猴补: sample_latents 的条件接口是 conds list, y 用 conds 即可; 重写简单版
import src.eval.inference as _inf
import time
def sample(sk_lat, callig_id, gid, _retry=True):
    yc = torch.tensor([callig_id], device=dev)
    yh = torch.tensor([gid], device=dev)
    sk = torch.from_numpy(sk_lat.astype(np.float32))[None]
    outs = []
    torch.manual_seed(int(time.time() * 1000) % 2**31)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = torch.randn(1, 4, 32, 32, device=dev)
        for t in diff.sample_times(50, dev):
            v = ctrl.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk,
                                      g=None)
            if isinstance(v, tuple):
                v = v[0]
            if v.shape[1] == 8:
                v = v[:, :4]
            x = x + (diff.sample_dt(50, dev) if hasattr(diff, "sample_dt") else -1.0 / 50) * v
    return x[0].float().cpu().numpy()

# ---- UI ----
with gr.Blocks(title="fame 书法生成") as demo:
    gr.Markdown("## fame 书法生成 — ControlNet (s21 + GT 骨架训练, 0.8045)\n"
                "训练过的字**直出**；没有的字先提示，用**标准字形骨架** zero-shot 生成并特殊标注。")
    with gr.Row():
        with gr.Column():
            char_in = gr.Textbox(label="汉字（单字）", value="永")
            callig_in = gr.Dropdown(CALLIGS, label="书家", value="赵孟頫" if "赵孟頫" in CALLIGS else CALLIGS[0])
            script_in = gr.Radio(["楷", "行", "隶"], label="书体", value="楷")
            btn = gr.Button("生成", variant="primary")
            note = gr.Textbox(label="状态", interactive=False)
        with gr.Column():
            out_img = gr.Image(label="生成结果", height=320)
            badge_tb = gr.Textbox(label="标记", interactive=False)
    with gr.Row():
        sk_img = gr.Image(label="使用的骨架", height=240)
        gt_gal = gr.Gallery(label="相近 GT", columns=4, height=240)
    gr.Markdown("### 本次会话所有生成")
    gallery = gr.Gallery(label="历史", columns=8, height=220)

    btn.click(generate, [char_in, callig_in, script_in],
              [out_img, sk_img, gt_gal, badge_tb, note, gallery])

demo.queue().launch(server_name="0.0.0.0", server_port=args.port)

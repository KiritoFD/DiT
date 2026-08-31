# -*- coding: utf-8 -*-
"""gradio_fame_local.py — fame 书法生成前端 (本地 GPU, share=True 公网).

逻辑:
  * 骨架 PS 可选 1px / 3px, 分别加载对应 ControlNet ckpt (双 ctrl 常驻)
  * (书体, 字) 在 fame 训练集且有骨架 → 对应 PS 训练集骨架直出
  * 没训过 → 提示 ZERO-SHOT, 1px 优先查标准字形骨架库, 否则本地渲染骨架 (楷simkai / 行STXINGKA / 隶SIMLI)
  * 每次生成: 生成大图 + 使用骨架 + 状态徽标 + 相近 GT (本地 MCCD)
  * 会话所有生成累积在底部 gallery
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
# gradio 6.x 自检用 httpx 回连 localhost, 会读取 Windows 系统代理 (127.0.0.1:7890) 导致 502.
# 强制绕过代理访问本机.
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"
os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
import sys
import csv
import time
import subprocess
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
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
NUM_CHARACTERS = 7026
WIN_FONT = r"C:\Windows\Fonts"
SCRIPT_FONT = {"楷": "simkai.ttf", "行": "STXINGKA.TTF", "隶": "SIMLI.TTF",
               "草": "simkai.ttf", "篆": "simkai.ttf", "六体": "simkai.ttf"}
GT_CACHE = os.path.join(ROOT, "_gt_cache")
REMOTE = "4090"

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7863)
ap.add_argument("--share", action="store_true")
args = ap.parse_args()

import gradio as gr
from src.model.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import build_diffusion, sample_latents, load_eval_vae

print("[load] main/ctrl/vae ...", flush=True)
dev = torch.device("cuda")
main = load_main_model(
    "DiT-2Cond-S/2",
    "5script/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/0030000.pt",
    device=dev, num_calligraphers=1013, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128,
    char_embed_dim=384, char_proj_mode="ln_only", freeze_char_table=True,
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True, attn_impl="eager")
main.eval()

CTRLS = {}
for _ps, _ck in (("1px", "_sync_work/ckpts/ctrl_1px_0050000.pt"),   # 1px GT skel ctrl (ctrl_fame_1pix_v1)
                 ("3px", "_sync_work/ckpts/ctrl_3px_0050000.pt")):  # 3px GT skel ctrl
    _c = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True).to(dev)
    _ckd = torch.load(_ck, map_location="cpu", weights_only=False)
    _sd = {k: v for k, v in (_ckd.get("ema") or _ckd.get("ctrl")).items()
           if not k.startswith("main.")}
    _m, _u = _c.load_state_dict(_sd, strict=False)
    _c.eval()
    CTRLS[_ps] = _c
    print(f"[load] ctrl[{_ps}] {_ck} inj={sum(1 for k in _sd if 'injection' in k)}", flush=True)

vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
diff = build_diffusion(50, "flow")
print("[load] done", flush=True)

MCCD_MAP = {}
try:
    for r in csv.DictReader(open("5script/mccd_image_map.csv", encoding="utf-8")):
        MCCD_MAP.setdefault(r["character"], []).append(r["filepath"])
    print(f"[mccd] chars indexed: {len(MCCD_MAP)}", flush=True)
except Exception as e:
    print("[mccd] map load failed:", e, flush=True)
rows = list(csv.DictReader(open("5script/train_fame.csv", encoding="utf-8")))
by_pair = {}
for r in rows:
    by_pair.setdefault((r["script"], r["character"]), []).append(r)
CALLIGS = sorted({r["calligrapher"] for r in rows})
CALLIG_ID = {}
for r in rows:
    CALLIG_ID.setdefault(r["calligrapher"], int(r["calligrapher_id"]))
CHAR_IDS = {}
for r in rows:
    CHAR_IDS.setdefault((r["script"], r["character"]), int(r["character_id"]))
USED_IDS = set(CHAR_IDS.values())
_next_cid = [7000]

BANK_TRAIN = {}
for _ps, _fn in (("1px", "_sync_work/skel_bank_train_1px.npz"),   # 1px GT 骨架库 (与 1px ctrl 训练条件一致)
                 ("3px", "skel_bank_train.npz")):                 # 3px GT 骨架库 (与 3px ctrl 训练条件一致)
    if not os.path.isfile(_fn):
        print(f"[WARN] 骨架库缺失: {_fn} -> {_ps} 直出不可用, 回退 ZERO-SHOT", flush=True)
        continue
    _b = np.load(_fn)
    _lat = _b["latents"]
    BANK_TRAIN[_ps] = {k: _lat[i] for i, k in enumerate(_b["keys"])}
    del _lat, _b
    print(f"[bank] train[{_ps}]: {len(BANK_TRAIN[_ps])} entries <- {_fn}", flush=True)

BANK_STD1 = {}   # 1px 标准字形骨架库 (与 s27 训练条件同源, zero-shot 优先查表)
_std1_fn = "_sync_work/skel_bank_std1.npz"
if os.path.isfile(_std1_fn):
    _b = np.load(_std1_fn)
    _lat = _b["latents"]
    BANK_STD1 = {k: _lat[i] for i, k in enumerate(_b["keys"])}
    del _lat, _b
    print(f"[bank] std1: {len(BANK_STD1)} entries", flush=True)

_f_cache = {}
def font_of(script, size):
    key = (script, size)
    if key not in _f_cache:
        _f_cache[key] = ImageFont.truetype(os.path.join(WIN_FONT, SCRIPT_FONT.get(script, "simkai.ttf")), size)
    return _f_cache[key]

def char_id_of(script, ch):
    if (script, ch) in CHAR_IDS:
        return CHAR_IDS[(script, ch)], True
    key = ("*", ch)
    if key not in CHAR_IDS:
        while _next_cid[0] in USED_IDS or _next_cid[0] >= NUM_CHARACTERS:
            _next_cid[0] += 1
        CHAR_IDS[key] = _next_cid[0]
    return CHAR_IDS[key], False

def decode_sk(sk_lat):
    """骨架 latent -> 骨架显示图 (VAE decode)."""
    with torch.no_grad():
        rec = vae.decode(torch.from_numpy(sk_lat.astype(np.float32))[None].to(dev) / 0.18215).sample.float().cpu()
    return Image.fromarray((((rec[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy() * 255).astype("uint8"))

def std_skel(script, ch, ps):
    img = Image.new("L", (256, 256), 255)
    d = ImageDraw.Draw(img)
    d.text((128, 128), ch, font=font_of(script, 200), fill=0, anchor="mm")
    a = np.asarray(img)
    if (a < 250).sum() < 10:
        return None, None
    sk = skeletonize(a < 127)
    if ps == "3px":
        sk = dil3(sk)  # 3px: 膨胀 x3, 与 3px ctrl 训练条件一致
    sk_u8 = np.where(sk, 0, 255).astype("uint8")
    sk_img = Image.fromarray(sk_u8, "L")
    with torch.no_grad():
        x = torch.from_numpy(sk_u8.astype(np.float32) / 255. * 2 - 1)[None, None].repeat(1, 3, 1, 1).to(dev)
        lat = (vae.encode(x).latent_dist.mode() * 0.18215)[0].half().cpu().numpy()
    return lat, sk_img

SESSION = []

def generate(ch, callig, script, steps, cfg, seed_n, ps):
    ch = (ch or "").strip()
    if len(ch) != 1:
        return None, None, None, None, "请输入单个汉字", SESSION[:]
    if callig not in CALLIG_ID:
        return None, None, None, None, f"书家「{callig}」不在 fame 训练集", SESSION[:]
    if ps not in CTRLS:
        return None, None, None, None, f"ctrl[{ps}] 未加载", SESSION[:]
    known_pair = (script, ch) in by_pair
    known_char = any(c == ch for (s, c) in by_pair)
    key = script + "|" + ch
    trained = known_pair and ps in BANK_TRAIN and key in BANK_TRAIN[ps]
    callig_id = CALLIG_ID[callig]
    cid, _ = char_id_of(script, ch)
    gid = SCRIPT_ID[script] * NUM_CHARACTERS + cid
    ctrl = CTRLS[ps]

    if trained:
        sk_lat = BANK_TRAIN[ps][key]
        badge = f"训练过 · 直出（骨架={ps} 训练集骨架库）"
        badge_color = "#227722"
        sk_img = decode_sk(sk_lat)
    elif ps == "1px" and key in BANK_STD1:
        sk_lat = BANK_STD1[key]
        badge = ("ZERO-SHOT · 标准字形 1px 骨架库直查（该 (书体,字) 组合未训练）" if known_char
                 else "ZERO-SHOT · 标准字形 1px 骨架库直查（该字未训练）")
        badge_color = "#CC0000"
        sk_img = decode_sk(sk_lat)
    else:
        sk_lat, sk_img = std_skel(script, ch, ps)
        if sk_lat is None:
            return None, None, None, None, f"标准字体不包含「{ch}」，无法构建骨架", SESSION[:]
        if known_char:
            badge = f"ZERO-SHOT · 标准字形 {ps} 骨架（该 (书体,字) 组合未训练）"
        else:
            badge = f"ZERO-SHOT · 标准字形 {ps} 骨架（该字未训练）"
        badge_color = "#CC0000"

    seed = int(seed_n) if str(seed_n).strip() else int(time.time() * 1000) % 2**31
    diff_n = build_diffusion(int(steps), "flow")
    lat = sample_latents(ctrl, diff_n, torch.randn(1, 4, 32, 32), [(callig_id, gid)],
                         float(cfg), 1, dev, skel=torch.from_numpy(sk_lat.astype(np.float32))[None],
                         seed=seed)
    with torch.no_grad():
        rec = vae.decode(lat.to(dev) / 0.18215).sample.float().cpu()
    out_arr = ((rec[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    out_img = Image.fromarray((out_arr * 255).astype("uint8")).resize((320, 320))

    gts = []
    for fp in MCCD_MAP.get(ch, [])[:6]:
        try:
            im = Image.open(fp).convert("RGB")
            im.thumbnail((160, 160))
            gts.append((im, os.path.basename(fp).rsplit("-", 1)[0]))
        except Exception:
            continue
    gt_label = f"相近 GT（本地 MCCD：{ch}，n={len(gts)}）" if gts else "本地 MCCD 无该字样本"
    gts = [(im, cap) for im, cap in gts]

    tag = f"{ps}·" + ("直出" if trained else "ZERO-SHOT")
    SESSION.append((out_img.copy(), f"{script}·{ch}·{callig} [{tag}] ssim_n/a"))
    note = f"{badge} ｜ {gt_label}"
    return out_img, badge, (sk_img if sk_img is not None else None), gts, note, SESSION[:]

with gr.Blocks(title="fame 书法生成") as demo:
    gr.Markdown("## fame 书法生成 — s21 基模 + GT 骨架 ControlNet（**1px / 3px** 双版本可切换）\n"
                "训练过的字**直出**（骨架=对应 PS 的训练骨架库）；没有的字用**标准字形骨架** zero-shot 生成，红色徽标特殊标出。")
    with gr.Row():
        with gr.Column():
            char_in = gr.Textbox(label="汉字（单字）", value="阜")
            callig_in = gr.Dropdown(CALLIGS, label="书家",
                                    value="褚遂良" if "褚遂良" in CALLIGS else CALLIGS[0])
            script_in = gr.Radio(["楷", "行", "隶"], label="书体", value="楷")
            ps_in = gr.Radio(["1px", "3px"], label="骨架 PS（ckpt 版本）", value="1px")
            steps_in = gr.Slider(10, 100, value=50, step=5, label="采样步数 Euler")
            cfg_in = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="CFG（骨架条件 0.5-1.0 最佳）")
            seed_in = gr.Textbox(label="seed（留空随机）", value="")
            btn = gr.Button("生成", variant="primary")
            note = gr.Textbox(label="状态 / 标记", lines=2)
        with gr.Column():
            out_img = gr.Image(label="生成结果")
            badge_tb = gr.Textbox(label="徽标", interactive=False)
    with gr.Row():
        sk_img = gr.Image(label="使用的骨架")
        gt_gal = gr.Gallery(label="相近 GT（本地 MCCD）", columns=6)
    gr.Markdown("### 会话内所有生成结果")
    gallery = gr.Gallery(label="历史", columns=8)

    btn.click(generate, [char_in, callig_in, script_in, steps_in, cfg_in, seed_in, ps_in],
              [out_img, badge_tb, sk_img, gt_gal, note, gallery])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1", server_port=args.port,
                                  share=args.share)

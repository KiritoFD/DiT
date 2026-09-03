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
# ---- v8 链模型 (本地 ckpts_v8/), 可选加载 ----
V8_CKPTS = {
    "v8a-base(无骨架)": "_sync_work/ckpts_v8/A_main_final.pt",
    "v8b-ctrl(skel)": "_sync_work/ckpts_v8/B_ctrl_best.pt",
    "v8c-repa(skel)": "_sync_work/ckpts_v8/v8c_repa_015000.pt",
}

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7863)
ap.add_argument("--share", action="store_true")
ap.add_argument("--model", default="s21",
                help="默认模型: s21=旧双ctrl / v8a / v8b / v8c (可页面切换)")
args = ap.parse_args()

import gradio as gr
from src.model.controlnet import load_main_model, ControlNetDiT
from src.model import DiT_2Cond_models
from src.eval.inference import build_diffusion, sample_latents, load_eval_vae

print("[load] main/ctrl/vae ...", flush=True)
dev = torch.device("cuda")

def _strip_orig(prefix, sd):
    """剥 _orig_mod. 前缀 + 按前缀过滤 (main./ctrl_encoder/injection)."""
    out = {}
    for k, v in sd.items():
        k2 = k.replace("_orig_mod.", "", 1)
        if k2.startswith(prefix):
            out[k2[len(prefix):]] = v
    return out

def _load_full(ckpt_path):
    """加载完整 ControlNetDiT (含 main + ctrl). 返回 (main_model, ctrl_encoder sd, 是否含 main)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck.get("ctrl") or {}
    return sd

# 预加载 v8 链模型 (main + ctrl 都从各自 ckpt 的 ema 提取)
V8 = {}
for _mk, _ck in V8_CKPTS.items():
    _sd = _load_full(_ck)
    _main_sd = _strip_orig("main.", _sd)
    _ctrl_sd = _strip_orig("", {k: v for k, v in _sd.items()
                                if ("ctrl_encoder" in k or "injection" in k)})
    if not _main_sd:  # 纯 base ckpt (A_main_final: ema 即主模型, 无 ctrl)
        _main_sd = {k.replace("_orig_mod.", "", 1): v for k, v in _sd.items()
                    if not any(s in k for s in ("ctrl_encoder", "injection", "optimizer"))}
        _ctrl_sd = {}
    _m = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=1013, num_characters=35130,
        condition_fusion="factorized_add", callig_embed_dim=128,
        char_embed_dim=384, char_proj_mode="mlp", freeze_char_table=True,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.30,
        cond_drop_which_glyph_prob=0.85,
        use_checkpoint=False, learn_sigma=False,
        norm_type="rms", mlp_type="swiglu", qk_norm=True,
        rope=True, rope_theta=100.0, attn_impl="sdpa")
    _m.load_state_dict(_main_sd, strict=False)
    _m.eval()
    _c = ControlNetDiT(_m, cond_in_channels=4, train_ctrl_only=True).to(dev)
    if _ctrl_sd:
        _mm, _uu = _c.load_state_dict(_ctrl_sd, strict=False)
        print(f"[load] v8[{_mk}] ctrl inj={sum(1 for k in _ctrl_sd if 'injection' in k)} "
              f"miss={len(_mm)} unexp={len(_uu)}", flush=True)
    else:
        print(f"[load] v8[{_mk}] base-only (无 ctrl)", flush=True)
    _c.eval()
    V8[_mk] = (_m, _c)

# 旧 s21 双 ctrl (保持原逻辑, 兼容)
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

def generate(ch, callig, script, steps, cfg, seed_n, ps, sketch, model_key="s21"):
    ch = (ch or "").strip()
    if len(ch) != 1:
        return None, None, None, None, "请输入单个汉字", SESSION[:]
    if callig not in CALLIG_ID:
        return None, None, None, None, f"书家「{callig}」不在 fame 训练集", SESSION[:]
    # 模型选择: v8 链 (v8a/v8b/v8c) 或旧 s21 双 ctrl
    if model_key in V8:
        _main_m, ctrl = V8[model_key]
        _ps = "1px"  # v8 链统一 1px skel
    else:
        if ps not in CTRLS:
            return None, None, None, None, f"ctrl[{ps}] 未加载", SESSION[:]
        ctrl = CTRLS[ps]
        _ps = ps
    known_pair = (script, ch) in by_pair
    known_char = any(c == ch for (s, c) in by_pair)
    key = script + "|" + ch
    trained = known_pair and _ps in BANK_TRAIN and key in BANK_TRAIN[_ps]
    callig_id = CALLIG_ID[callig]
    cid, _ = char_id_of(script, ch)
    gid = SCRIPT_ID[script] * NUM_CHARACTERS + cid

    # 手绘骨架优先: 用户画了就用手画的
    sk_lat_user = None
    print(f"[DEBUG] sketch type={type(sketch).__name__} value={repr(sketch)[:200] if sketch is not None else 'None'}", flush=True)
    if sketch is not None:
        try:
            # gradio Sketchpad 可能返回 dict {"background":..., "layers":[...], "composite":...} 或 PIL Image
            if isinstance(sketch, dict):
                # 合成 background + layers
                bg = sketch.get("background")
                layers = sketch.get("layers", [])
                if bg is not None:
                    comp = bg.copy()
                    for layer in layers:
                        if layer is not None:
                            comp = Image.alpha_composite(comp.convert("RGBA"), layer.convert("RGBA")).convert("RGB")
                    sketch_img = comp
                else:
                    sketch_img = None
            elif isinstance(sketch, (Image.Image, np.ndarray)):
                sketch_img = sketch
            else:
                sketch_img = None
            if sketch_img is not None:
                sa = np.asarray(Image.fromarray(sketch_img).convert("L")) if not isinstance(sketch_img, Image.Image) else np.asarray(sketch_img.convert("L"))
                ink_px = int((sa < 200).sum())
                print(f"[DEBUG] sketch ink_px={ink_px} size={sa.shape}", flush=True)
                if ink_px > 20:
                    sk = skeletonize(sa < 200)
                    sk3 = np.where(dil3(sk), 0, 255).astype("uint8")
                    with torch.no_grad():
                        x = torch.from_numpy(sk3.astype(np.float32)/255.*2-1)[None,None].repeat(1,3,1,1).to(dev)
                        sk_lat_user = (vae.encode(x).latent_dist.mode()*0.18215)[0].half().cpu().numpy()
                    print(f"[DEBUG] sk_lat_user OK, will use hand-drawn", flush=True)
                else:
                    print(f"[DEBUG] sketch too empty ({ink_px}px), skipping hand-drawn", flush=True)
        except Exception as e:
            print(f"[DEBUG] sketch processing ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if sk_lat_user is not None:
        sk_lat = sk_lat_user
        badge = f"手绘骨架 · ControlNet 条件（{_ps}）"
        badge_color = "#FF6600"
        sk_img = decode_sk(sk_lat)
    elif trained:
        sk_lat = BANK_TRAIN[_ps][key]
        badge = f"训练过 · 直出（骨架={_ps} 训练集骨架库）"
        badge_color = "#227722"
        sk_img = decode_sk(sk_lat)
    elif _ps == "1px" and key in BANK_STD1:
        sk_lat = BANK_STD1[key]
        badge = ("ZERO-SHOT · 标准字形 1px 骨架库直查（该 (书体,字) 组合未训练）" if known_char
                 else "ZERO-SHOT · 标准字形 1px 骨架库直查（该字未训练）")
        badge_color = "#CC0000"
        sk_img = decode_sk(sk_lat)
    else:
        sk_lat, sk_img = std_skel(script, ch, _ps)
        if sk_lat is None:
            return None, None, None, None, f"标准字体不包含「{ch}」，无法构建骨架", SESSION[:]
        if known_char:
            badge = f"ZERO-SHOT · 标准字形 {_ps} 骨架（该 (书体,字) 组合未训练）"
        else:
            badge = f"ZERO-SHOT · 标准字形 {_ps} 骨架（该字未训练）"
        badge_color = "#CC0000"

    seed = int(seed_n) if str(seed_n).strip() else int(time.time() * 1000) % 2**31
    diff_n = build_diffusion(int(steps), "flow")
    _skel_arg = None
    if model_key in V8:
        # v8c/v8b 有 ctrl → 传 skel; v8a base-only → 不传 (走 base 通道)
        if model_key != "v8a-base(无骨架)" and sk_lat is not None:
            _skel_arg = torch.from_numpy(sk_lat.astype(np.float32))[None]
    else:
        _skel_arg = torch.from_numpy(sk_lat.astype(np.float32))[None]
    lat = sample_latents(ctrl, diff_n, torch.randn(1, 4, 32, 32), [(callig_id, gid)],
                         float(cfg), 1, dev, skel=_skel_arg,
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

    tag = f"{_ps}·" + ("直出" if trained else "ZERO-SHOT")
    SESSION.append((out_img.copy(), f"{script}·{ch}·{callig} [{tag}] ssim_n/a"))
    note = f"{badge} ｜ {gt_label}"
    return out_img, badge, (sk_img if sk_img is not None else None), gts, note, SESSION[:]

with gr.Blocks(title="fame 书法生成") as demo:
    gr.Markdown("## fame 书法生成 — 多模型可选（s21 GT 骨架 CtrlNet / **v8 链 v8a·v8b·v8c**）\n"
                "模型下拉切换：v8a（无骨架 base）、v8b（skel-ctrl）、v8c（REPA）为 v8 资产同协议产物；"
                "s21 为旧双 ctrl（1px/3px）。训练过的字**直出**；没有的字用**标准字形骨架** zero-shot 生成，红色徽标特殊标出。")
    with gr.Row():
        with gr.Column():
            model_in = gr.Dropdown(["s21", "v8a-base(无骨架)", "v8b-ctrl(skel)", "v8c-repa(skel)"],
                                   label="模型", value=args.model if args.model in
                                   ["s21", "v8a-base(无骨架)", "v8b-ctrl(skel)", "v8c-repa(skel)"] else "s21")
            char_in = gr.Textbox(label="汉字（单字）", value="阜")
            callig_in = gr.Dropdown(CALLIGS, label="书家",
                                    value="褚遂良" if "褚遂良" in CALLIGS else CALLIGS[0])
            script_in = gr.Radio(["楷", "行", "隶"], label="书体", value="楷")
            ps_in = gr.Radio(["1px", "3px"], label="骨架 PS（仅 s21 生效，v8 链固定 1px）", value="1px")
            sketch_in = gr.Sketchpad(type="pil", label="手绘骨架（白底黑笔，优先级最高）",
                                     height=256, brush=gr.Brush(colors=["#000000"], default_size=4))
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

    btn.click(generate,
              [char_in, callig_in, script_in, steps_in, cfg_in, seed_in, ps_in, sketch_in, model_in],
              [out_img, badge_tb, sk_img, gt_gal, note, gallery])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1", server_port=args.port,
                                  share=args.share)

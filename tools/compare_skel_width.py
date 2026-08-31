# -*- coding: utf-8 -*-
"""compare_skel_width.py — 1px vs 3px skel ControlNet 同 seed 对比推理.

复用 gradio_fame_local.py 的 pipeline (s21 main + ControlNetDiT + VAE + flow):
  * 每个样例: GT | 1px ctrl 生成 | 3px ctrl 生成 并排输出
  * 条件域严格匹配训练分布: 1px ctrl 喂 1px 骨架 latent, 3px ctrl 喂 3px 骨架 latent
  * 同 seed 同 cfg, 唯一变量 = 骨架线宽 (训练数据)

用法: python tools/compare_skel_width.py [--samples-dir _sync_work/samples] [--out ...]
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
import csv
import argparse
import glob
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--samples-dir", default="_sync_work/samples")
ap.add_argument("--out", default="_sync_work/compare")
ap.add_argument("--cfg", type=float, default=0.7)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--seed", type=int, default=20260831)
ap.add_argument("--main-ckpt",
                default="5script/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/0030000.pt")
ap.add_argument("--ctrl-1px", default="_sync_work/ckpts/ctrl_1px_0050000.pt")
ap.add_argument("--ctrl-3px", default="_sync_work/ckpts/ctrl_3px_0050000.pt")
args = ap.parse_args()

from src.model.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import build_diffusion, sample_latents, load_eval_vae

dev = torch.device("cuda")
SCRIPT_ID = {"楷": 0, "行": 3, "隶": 4, "草": 2, "篆": 1, "六体": 5, "其他": 0}
NUM_CHARACTERS = 7026

# ---- 与 gradio_fame_local.py 一致的条件映射 ----
rows = list(csv.DictReader(open("5script/train_fame.csv", encoding="utf-8")))
CALLIG_ID = {}
CHAR_IDS = {}
for r in rows:
    CALLIG_ID.setdefault(r["calligrapher"], int(r["calligrapher_id"]))
    CHAR_IDS.setdefault((r["script"], r["character"]), int(r["character_id"]))
USED_IDS = set(CHAR_IDS.values())
_next_cid = [7000]


def char_id_of(script, ch):
    if (script, ch) in CHAR_IDS:
        return CHAR_IDS[(script, ch)]
    key = ("*", ch)
    if key not in CHAR_IDS:
        while _next_cid[0] in USED_IDS or _next_cid[0] >= NUM_CHARACTERS:
            _next_cid[0] += 1
        CHAR_IDS[key] = _next_cid[0]
    return CHAR_IDS[key]


def load_ctrl(ckpt_path):
    ctrl = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True).to(dev)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in (ck.get("ema") or ck.get("ctrl")).items()
          if not k.startswith("main.")}
    ctrl.load_state_dict(sd, strict=False)
    ctrl.eval()
    n_inj = sum(1 for k in sd if "injection" in k)
    print(f"[ctrl] {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}: "
          f"{len(sd)} params, injection keys={n_inj}", flush=True)
    return ctrl


@torch.no_grad()
def encode_skel(png_path):
    """skel PNG (白底黑线) -> latent (4,32,32), 与训练 encode 一致"""
    img = Image.open(png_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(dev)
    lat = vae.encode(x).latent_dist.mode() * 0.18215
    return lat[0].float().cpu().numpy()


@torch.no_grad()
def gen(ctrl, sk_lat, callig_id, gid, seed):
    lat = sample_latents(ctrl, diff, torch.randn(1, 4, 32, 32),
                         [(callig_id, gid)], args.cfg, 1, dev,
                         skel=torch.from_numpy(sk_lat.astype(np.float32))[None],
                         seed=seed)
    rec = vae.decode(lat.to(dev) / 0.18215).sample.float().cpu()
    arr = ((rec[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype("uint8")).resize((256, 256))


# ---- 加载 ----
print("[load] main ...", flush=True)
main = load_main_model(
    "DiT-2Cond-S/2", args.main_ckpt, device=dev,
    num_calligraphers=1013, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128,
    char_embed_dim=384, char_proj_mode="ln_only", freeze_char_table=True,
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True, attn_impl="eager")
main.eval()
print("[load] vae/diff ...", flush=True)
vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
diff = build_diffusion(args.steps, "flow")
print("[load] ctrl 1px/3px ...", flush=True)
ctrl1 = load_ctrl(args.ctrl_1px)
ctrl3 = load_ctrl(args.ctrl_3px)

# ---- 遍历样例 ----
os.makedirs(args.out, exist_ok=True)
meta = [l.strip().split("\t") for l in
        open(os.path.join(args.samples_dir, "meta.txt"), encoding="utf-8") if l.strip()]
print(f"[run] {len(meta)} samples, seed={args.seed}, cfg={args.cfg}", flush=True)

for tag, script, ch, img_id, callig in meta:
    cp = f"U+{ord(ch):04X}"
    pre = f"{tag}_{script}_{cp}_{img_id}"
    base = os.path.join(args.samples_dir, pre)
    if not os.path.isfile(base + "_skel1.png"):
        print(f"  skip {pre} (no skel1)"); continue
    callig_id = CALLIG_ID.get(callig, 0)
    gid = SCRIPT_ID[script] * NUM_CHARACTERS + char_id_of(script, ch)

    lat1 = encode_skel(base + "_skel1.png")
    lat3 = encode_skel(base + "_skel3.png")
    img_gt = Image.open(base + "_gt.png").convert("RGB").resize((256, 256))
    img1 = gen(ctrl1, lat1, callig_id, gid, args.seed)
    img3 = gen(ctrl3, lat3, callig_id, gid, args.seed)
    # 交叉对照: 1px ctrl 喂 3px 骨架 (验证条件域不匹配的退化) —— 不需要, 保持干净对比

    canvas = Image.new("RGB", (256 * 3 + 16, 256), "white")
    canvas.paste(img_gt, (0, 0))
    canvas.paste(img1, (256 + 8, 0))
    canvas.paste(img3, (512 + 16, 0))
    out_path = os.path.join(args.out, f"cmp_{pre}.png")
    canvas.save(out_path)
    print(f"  saved {out_path}  ({tag} {script}{ch} callig={callig})", flush=True)

print("done.")

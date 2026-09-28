# -*- coding: utf-8 -*-
"""DG-Font sampling on top10 eval protocol.

DG-Font inference = GuidingNet(x_ref, sty=True) -> style vector, then
G.decode(cnt_encoder(x_content), s_ref, skip1, skip2). One reference per item
(refs[slot][char][0], same protocol file as the other baselines).
Output {slot}__{char}.png at 256.

Run from baseline/DG-Font:
  python ../adapter/dgfont/sample_top10.py \
      --model_dir log/GAN_top10_256_XXXX/models/ (auto-picks latest) \
      --eval_tag strict84 --save_dir ../results/dgfont/strict84
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.getcwd())  # DG-Font repo

from models.generator import Generator  # noqa: E402
from models.guidingNet import GuidingNet  # noqa: E402

DIT = "/root/Workspace/xy/DiT"
CONTENT = os.path.join(DIT, "baseline/data/content_font/deng")
EVAL = os.path.join(DIT, "baseline/data/eval")
DG_TRAIN = os.path.join(DIT, "baseline/data/dgfont/train")


def load_latest(log_dir):
    cands = sorted(glob.glob(os.path.join(log_dir, "model_*.ckpt")),
                   key=lambda p: int(p.rsplit("_", 1)[1][:-5]))
    assert cands, f"no model_*.ckpt under {log_dir}"
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs/GAN_top10_256",
                    help="DG-Font log dir containing model_*.ckpt")
    ap.add_argument("--ckpt", default=None, help="explicit model_*.ckpt override")
    ap.add_argument("--eval_tag", default="strict84", choices=["strict84", "seen20"])
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--sty_dim", type=int, default=128)
    ap.add_argument("--output_k", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--random_init", action="store_true",
                    help="skip ckpt loading (inference-path probe only)")
    ap.add_argument("--limit", type=int, default=0, help="only first N items")
    args = ap.parse_args()

    if args.random_init:
        ckpt_path = None
    else:
        ckpt_path = args.ckpt or load_latest(args.log_dir)
    print(f"using checkpoint {ckpt_path}")
    G = Generator(args.img_size, args.sty_dim, use_sn=False).cuda().eval()
    C = GuidingNet(args.img_size, {"cont": args.sty_dim, "disc": args.output_k}).cuda().eval()
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        g_sd, c_sd = sd["G_EMA_state_dict"], sd["C_EMA_state_dict"]
        if any(k.startswith("_orig_mod.") for k in g_sd):
            g_sd = {k.replace("_orig_mod.", ""): v for k, v in g_sd.items()}
        if any(k.startswith("_orig_mod.") for k in c_sd):
            c_sd = {k.replace("_orig_mod.", ""): v for k, v in c_sd.items()}
        G.load_state_dict(g_sd)
        C.load_state_dict(c_sd)

    tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    with open(os.path.join(EVAL, "refs.json"), encoding="utf-8") as f:
        refs = json.load(f)
    with open(os.path.join(DG_TRAIN, "domain_map.json"), encoding="utf-8") as f:
        domain_map = json.load(f)          # id_XX -> slot_name
    slot2id = {v: k for k, v in domain_map.items()}
    items = sorted((slot, ch) for slot, d in refs.items() for ch in d)
    if args.limit:
        items = items[:args.limit]

    out_dir = args.save_dir
    os.makedirs(out_dir, exist_ok=True)

    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            x_src, x_ref = [], []
            for slot, ch in chunk:
                x_src.append(tf(Image.open(os.path.join(CONTENT, f"{ch}.png")).convert("RGB")))
                ref_ch = refs[slot][ch][0]
                x_ref.append(tf(Image.open(
                    os.path.join(DG_TRAIN, slot2id[slot], f"{ref_ch}.png")).convert("RGB")))
            x_src = torch.stack(x_src).cuda()
            x_ref = torch.stack(x_ref).cuda()

            c_src, skip1, skip2 = G.cnt_encoder(x_src)
            s_ref = C(x_ref, sty=True)
            out, _ = G.decode(c_src, s_ref, skip1, skip2)   # [-1,1] via tanh

            for (slot, ch), im in zip(chunk, out.cpu()):
                arr = ((im.clamp(-1, 1) + 1) / 2).numpy()
                if arr.shape[0] == 3:
                    arr = arr.transpose(1, 2, 0)
                arr = (arr * 255).round().astype(np.uint8).squeeze()
                Image.fromarray(arr).save(os.path.join(out_dir, f"{slot}__{ch}.png"))
            print(f"[{i + len(chunk)}/{len(items)}] saved", flush=True)


if __name__ == "__main__":
    main()

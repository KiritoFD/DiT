# -*- coding: utf-8 -*-
"""diag_skelnet_dit_output.py — 直接看 SkelNet-DiT(纯生成, 无 warp) 的输出骨架白不白。

回答: "完全扔掉 warp, 直接干 DiT, 没道理白化" —— 实测它到底白不白。
输出三列对照: [输入标准骨架 | SkelNet-DiT 生成骨架 | GT 目标骨架], 并算各自 ink。

用法:
  python tools/diag_skelnet_dit_output.py --ckpt assets/skelnet_dit_B_w3.pt.step024000 --n 8
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch as th
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TIME_SCALE = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="assets/skelnet_dit_B_w3.pt.step024000")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cell", type=int, default=165)
    ap.add_argument("--out", default="/tmp/skelnet_dit_out.png")
    ap.add_argument("--tol", type=int, default=3)
    a = ap.parse_args()

    sd = th.load(a.ckpt, map_location="cpu", weights_only=False)
    A = sd.get("args", {})
    A = dict(A) if not isinstance(A, argparse.Namespace) else vars(A)
    n_slots = int(sd.get("n_slots", 0))
    print(f"[ckpt] step={sd.get('step')} n_slots={n_slots}")
    print(f"[args] depth={A.get('depth')} hidden={A.get('hidden')} heads={A.get('heads')} "
          f"inject={A.get('inject_layers')} pred={A.get('pred')} "
          f"sample_steps={A.get('sample_steps')}")
    print(f"[args] tgt={A.get('tgt_shards')} cond={A.get('cond_shards')} val={A.get('val_csv')}")

    from src.model.dit import DiT_2Cond
    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4, out_channels=4,
        depth=int(A.get("depth", 6)), hidden_size=int(A.get("hidden", 256)),
        num_heads=int(A.get("heads", 4)), num_calligraphers=max(n_slots, 1),
        num_characters=1, use_char_cond=False, use_glyph_cond=True,
        glyph_in_channels=4, glyph_inject_layers=int(A.get("inject_layers", 2)),
        glyph_inject_mode="adaln", glyph_scale_init=0.6, glyph_drop_prob=0.0,
        glyph_embedder_depth=2, condition_fusion="factorized_cat",
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=float(A.get("style_drop", 0.1)),
        cond_drop_one_prob=0.0, learn_sigma=False,
    ).cuda().eval()
    model.load_state_dict({k: v.cuda() for k, v in sd["ema"].items()}, strict=False)
    print("[model] EMA 权重已载入")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("data/pretrained/pretrained_models/sd-vae-ft-ema"
                                        ).cuda().eval()

    # 数据 (与训练脚本同构)
    from src.utils.latent_dataset import MCCDLatentDataset
    import json
    csmap = None
    if os.path.exists(A.get("callig_map", "")):
        csmap = json.load(open(A["callig_map"], encoding="utf-8"))
    ds = MCCDLatentDataset(
        csv_file=A.get("val_csv"), latent_shards_dir=A.get("tgt_shards"),
        img_root="", image_size=256, is_train=False, preload=True,
        load_image=False, skel_latent_shards_dir=A.get("cond_shards"),
        callig_id_map=None, callig_script_map=csmap)
    print(f"[data] val {len(ds)}")

    steps = int(A.get("sample_steps", 20))
    pred_mode = A.get("pred", "v")

    def dec(lat):
        with th.no_grad():
            d = vae.decode(lat.cuda() / 0.18215).sample.mean(1)
        return (d < 0).cpu().numpy()

    rows = list(csv.DictReader(open(A.get("val_csv"), encoding="utf-8")))
    n = min(a.n, len(ds))
    C = a.cell
    lab = 24
    cv = Image.new("RGB", (C * 3 + 24, 30 + n * (C + lab)), (245, 245, 245))
    dr = ImageDraw.Draw(cv)
    dr.text((6, 5), "左=输入标准骨架  中=SkelNet-DiT生成  右=GT目标骨架", fill=(0, 0, 0))
    inks_pred, inks_std, inks_gt = [], [], []
    for i in range(n):
        b = ds[i]
        g = b["skel_latent"].float()[None].cuda()          # 输入标准骨架
        y = th.tensor([int(b["y_callig"])], device="cuda")
        x0 = b["latent"].float()[None].cuda()              # GT 目标 (仅形状/GT参考)
        z = th.randn_like(x0)
        eps = z.clone()
        ts = th.linspace(1.0, 0.0, steps + 1, device="cuda")
        with th.no_grad():
            for k in range(steps):
                out = model(z, th.full((1,), float(ts[k]) * TIME_SCALE, device="cuda"),
                            y_callig=y, y_char=th.zeros_like(y), g=g)
                if isinstance(out, tuple):
                    out = out[0]
                if pred_mode == "x0":
                    z = (1 - ts[k + 1]) * out + ts[k + 1] * eps
                else:
                    z = z + (ts[k + 1] - ts[k]) * out
        p_pred = dec(z)[0]
        p_std = dec(g)[0]
        p_gt = dec(x0)[0]
        inks_pred.append(float(p_pred.mean()))
        inks_std.append(float(p_std.mean()))
        inks_gt.append(float(p_gt.mean()))
        yy = 30 + i * (C + lab)
        for j, pm in enumerate((p_std, p_pred, p_gt)):
            x = 6 + j * (C + 6)
            im = Image.fromarray(np.where(pm, 0, 255).astype("uint8"), "L").convert("RGB")
            cv.paste(im.resize((C, C)), (x, yy))
            dr.text((x, yy + C + 3), f"ink={float(pm.mean()):.4f}", fill=(80, 80, 80))
    cv.save(a.out)
    print("->", a.out)
    ip, isd, ig = (np.array(x) for x in (inks_pred, inks_std, inks_gt))
    print(f"\n  ink: std输入 {isd.mean():.4f} | SkelNet-DiT生成 {ip.mean():.4f} | "
          f"GT目标 {ig.mean():.4f}")
    print(f"  生成/GT 墨量比 = {ip.mean()/max(ig.mean(),1e-9):.3f}   "
          f"(<0.5 即为白化)")


if __name__ == "__main__":
    main()

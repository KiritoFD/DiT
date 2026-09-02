# -*- coding: utf-8 -*-
"""
make_ctrl_posters.py — ControlNet vs Base 对照 poster 生成 (远程 GPU).

每个样本画 [生成 | GT] 对, 崩溃样本 (ssim < 0.4) 红圈 + 红边标出;
头部写两组统计量 (median/mean SSIM, median MSE, 失败数, 分书体中位).
样本 = 已知 6 个最差 + 均匀抽 18 个 → 共 24, 两臂同噪声同 seed.
输出: /tmp/poster_ctrl.png, /tmp/poster_base.png
"""
import os
import sys
import csv
import math

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from src.model.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import (
    build_diffusion, sample_latents, load_eval_vae, make_eval_cache, _ssim, _mse,
)

N_TOTAL = 24
COLS = 6
CELL = 128          # 单图边长 (poster 里 out|GT 并排)
THRESH = 0.40       # 崩溃判定
CFGS = [0.7]
STEPS, BATCH = 50, 8
FONT = "/tmp/simhei.ttf"

WORST = ["稷", "祭", "累", "牒", "虢", "街"]  # 已知崩溃字 (保证入镜)


def pick_indices(rows):
    worst = [i for i, r in enumerate(rows) if r["character"] in WORST][:6]
    rest = [i for i in range(len(rows)) if i not in worst]
    stride = max(1, len(rest) // (N_TOTAL - len(worst)))
    picks = worst + rest[::stride][: N_TOTAL - len(worst)]
    return picks[:N_TOTAL]


def main():
    dev = torch.device("cuda")
    main_model = load_main_model(
        "DiT-2Cond-S/2",
        "5script/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/0030000.pt",
        device=dev, num_calligraphers=1013, num_characters=35130,
        condition_fusion="factorized_add", callig_embed_dim=128,
        char_embed_dim=384, char_proj_mode="ln_only", freeze_char_table=True)
    main_model.eval()
    ctrl = ControlNetDiT(main_model, cond_in_channels=4, train_ctrl_only=True).to(dev)
    ck = torch.load(
        "5script/results/s20_ctrl_skel_flow_v2/20260829-161522-s20-ctrl-skel-flow-v2/checkpoints/0050000.pt",
        map_location="cpu", weights_only=False)
    sd = {k: v for k, v in (ck.get("ema") or ck.get("ctrl")).items()
          if not k.startswith("main.")}
    ctrl.load_state_dict(sd, strict=False)
    ctrl.eval()
    n_inj = sum(1 for k in sd if k.startswith("injections"))
    print(f"injections loaded: {n_inj}")
    vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
    diffusion = build_diffusion(STEPS, "flow")
    cache = make_eval_cache("5script/eval_fame_strict.csv", "final_imgs_256",
                            None, 256, 100, 8, 4, 0.18215,
                            skel_latent_shards_dir="final_skel_latents_fame")
    rows = list(csv.DictReader(open("5script/eval_fame_strict.csv",
                                    encoding="utf-8")))[:cache["n"]]
    picks = pick_indices(rows)
    print("picks:", [(rows[i]["script"], rows[i]["character"]) for i in picks])

    def arm(with_skel, CFG):
        lat = sample_latents(ctrl, diffusion, cache["noise"], cache["conds"], CFG,
                             BATCH, dev,
                             skel=cache["skels_latent"] if with_skel else None, seed=0)
        outs = []
        with torch.no_grad():
            for i in range(0, cache["n"], 8):
                rec = vae.decode(lat[i:i + 8].to(dev) / 0.18215).sample.float().cpu()
                outs.append(((rec.clamp(-1, 1) + 1) / 2))
        return torch.cat(outs)

    f_big = ImageFont.truetype(FONT, 22)
    f_mid = ImageFont.truetype(FONT, 15)
    f_small = ImageFont.truetype(FONT, 12)

    def stats(vals):
        a = np.array(vals)
        return float(np.median(a)), float(np.mean(a))

    def build_poster(out_t, tag, CFG):
        per = []
        for k in picks:
            p = out_t[k].permute(1, 2, 0).numpy()
            g = ((cache["gts"][k].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
            per.append({"k": k, "p": p, "g": g, "s": _ssim(p, g), "m": _mse(p, g),
                        "row": rows[k]})
        all_s = [d["s"] for d in per]
        med, mean = stats(all_s)
        med_m = float(np.median([d["m"] for d in per]))
        fails = [d for d in per if d["s"] < THRESH]
        per_script = {}
        for d in per:
            per_script.setdefault(d["row"]["script"], []).append(d["s"])
        script_line = "  ".join(
            f"{s}中位 {np.median(v):.3f}" for s, v in sorted(per_script.items()))

        n_rows = math.ceil(len(per) / COLS)
        CW, CH = CELL * 2 + 6, CELL + 26
        W = COLS * CW + 20
        HEAD = 118
        H = HEAD + n_rows * CH + 10
        canvas = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(canvas)
        arm_name = "ControlNet (GT 骨架条件)" if tag.startswith("ctrl") else "Base (仅 书家+字 ID)"
        d.text((16, 10), f"s21-fame {arm_name} cfg={CFG} — eval_fame_strict 抽样 24 (含全部 6 个最差字)",
               font=f_big, fill="black")
        d.text((16, 44),
               f"SSIM 中位 {med:.3f} / 均值 {mean:.3f}   MSE中位 {med_m:.3f}   "
               f"失败(ssim<{THRESH}): {len(fails)}/{len(per)}", font=f_mid, fill="black")
        d.text((16, 70), f"分书体: {script_line}", font=f_mid, fill="#333333")
        d.text((16, 94),
               "每格左=模型输出 右=GT;  红圈=崩溃样本 (结构不可读/彩色伪影); ckpt=0050000, cfg1.0, Euler50, 同噪声",
               font=f_small, fill="#666666")
        for j, dd in enumerate(per):
            r_, c_ = divmod(j, COLS)
            x0, y0 = 10 + c_ * CW, HEAD + r_ * CH
            po = Image.fromarray((dd["p"] * 255).astype("uint8")).resize((CELL, CELL))
            pg = Image.fromarray((dd["g"] * 255).astype("uint8")).resize((CELL, CELL))
            canvas.paste(po, (x0, y0))
            canvas.paste(pg, (x0 + CELL + 6, y0))
            d.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1],
                        outline="#CCCCCC")
            d.rectangle([x0 + CELL + 6, y0, x0 + CELL * 2 + 5, y0 + CELL - 1],
                        outline="#CCCCCC")
            bad = dd["s"] < THRESH
            cap_c = "#CC0000" if bad else "black"
            d.text((x0, y0 + CELL + 2),
                   f"{dd['row']['script']} {dd['row']['character']} ssim={dd['s']:.2f}",
                   font=f_small, fill=cap_c)
            if bad:
                cx, cy = x0 + CELL // 2, y0 + CELL // 2
                rr = CELL * 0.62
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                          outline="#FF0000", width=4)
                for bx in (x0, x0 + CELL + 6):
                    d.rectangle([bx, y0, bx + CELL - 1, y0 + CELL - 1],
                                outline="#FF0000", width=3)
        out = f"/tmp/poster_{tag}.png"
        canvas.save(out)
        print("saved", out, f"(median {med:.3f}, fails {len(fails)})")

    for CFG in CFGS:
        imgs_ctrl = arm(True, CFG)
        imgs_base = arm(False, CFG)
        build_poster(imgs_ctrl, f"ctrl_cfg{CFG:.2f}", CFG)
        build_poster(imgs_base, f"base_cfg{CFG:.2f}", CFG)


if __name__ == "__main__":
    main()

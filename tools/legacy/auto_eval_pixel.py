#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_eval_pixel.py — Pixel-DiT 的独立 CPU 评估进程。

训练进程 (src/train_pixel.py) 只存 ckpt 不 eval; 本进程轮询新 ckpt,
在 CPU 上用 EMA 权重直接像素采样 eval100 (DDIM, cfg 4.0, 50 步), 与 GT 图算
MSE/SSIM, 写 eval_auto_<step>.json (供 dashboard/早停读取)。

用法:
  /opt/conda/bin/python tools/auto_eval_pixel.py --results-dir 5script/results/px_s_scratch
"""
import os
import sys
import time
import json
import glob
import argparse
import datetime
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from models import DiT_2Cond
from diffusion import create_diffusion

DT = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 像素 eval 缓存: eval100 csv -> (conds, gts)
# ---------------------------------------------------------------------------
def build_eval_cache(args):
    import csv
    import numpy as np
    from PIL import Image
    samples = []
    if not args.eval_csv or not os.path.exists(args.eval_csv):
        log(f"[cache] eval csv missing: {args.eval_csv}")
        return None
    with open(args.eval_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append(row)
    samples = samples[: args.eval_n]
    conds = []  # (callig_id, char_id)
    gts = torch.zeros(len(samples), 3, 256, 256)
    gt_np = np.zeros((len(samples), 256, 256, 3), dtype=np.float32)
    for i, s in enumerate(samples):
        import re
        m = re.search(r"(\d+)\.png", s["image_path"])
        img_id = int(m.group(1))
        with Image.open(os.path.join(args.img_root, f"{img_id}.png")) as im:
            gt = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        gt_np[i] = gt
        gts[i] = torch.from_numpy(gt.transpose(2, 0, 1)).float() * 2.0 - 1.0
        conds.append((int(s["calligrapher_id"]), int(s.get("glyph_id", s["character_id"]))))
    log(f"[cache] eval100 loaded ({len(samples)} samples)")
    return {"conds": conds, "gts": gts, "gt_np": gt_np}


# ---------------------------------------------------------------------------
# 模型构建 (与 train_pixel.py 的 DiT_2Cond( S/8 ) 一致)
# ---------------------------------------------------------------------------
def build_model(ckpt, device="cpu"):
    args = ckpt.get("args")
    if args is not None and hasattr(args, "image_size"):
        image_size = int(getattr(args, "image_size", 256))
        patch_size = int(getattr(args, "patch_size", 8))
        in_channels = int(getattr(args, "in_channels", 3))
        num_calligraphers = int(getattr(args, "num_calligraphers", 1011))
        num_characters = int(getattr(args, "num_characters", 35130))
        condition_fusion = getattr(args, "condition_fusion", "factorized_add")
        callig_embed_dim = int(getattr(args, "callig_embed_dim", 128))
        char_embed_dim = int(getattr(args, "char_embed_dim", 256))
        cond_drop_all = float(getattr(args, "cond_drop_all", 0.05))
        cond_drop_one = float(getattr(args, "cond_drop_one", 0.25))
    else:
        image_size, patch_size, in_channels = 256, 8, 3
        num_calligraphers, num_characters = 1011, 35130
        condition_fusion, callig_embed_dim, char_embed_dim = "factorized_add", 128, 256
        cond_drop_all, cond_drop_one = 0.05, 0.25
    model = DiT_2Cond(
        input_size=image_size, patch_size=patch_size, in_channels=in_channels,
        hidden_size=384, depth=12, num_heads=6,
        num_calligraphers=num_calligraphers, num_characters=num_characters,
        condition_fusion=condition_fusion,
        callig_embed_dim=callig_embed_dim, char_embed_dim=char_embed_dim,
        cond_drop_all_prob=cond_drop_all, cond_drop_one_prob=cond_drop_one,
        use_checkpoint=False, learn_sigma=True,
    )
    sd = ckpt.get("ema") or ckpt.get("delta")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"[model] loaded from ckpt ema/delta (missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device).eval()


def _ssim_simple(p, g):
    """p,g: (H,W,3) [0,1] numpy —— 简化 SSIM (全局均值/方差/协方差)。"""
    import numpy as np
    gp, gg = p.mean(), g.mean()
    vp, vg = p.var(), g.var()
    cov = np.mean((p - gp) * (g - gg))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * gp * gg + c1) * (2 * cov + c2))
                 / ((gp ** 2 + gg ** 2 + c1) * (vp + vg + c2) + 1e-8))


# ---------------------------------------------------------------------------
# 单 ckpt 评测 (像素域直接采样)
# ---------------------------------------------------------------------------
def eval_one(model, ckpt_dir, step, cache, cfg, ckpt_path, device="cpu"):
    steps = int(cfg["steps"])
    sample_cfg = float(cfg["cfg"])
    seed = int(cfg["seed"])
    metric_batch = int(cfg["batch"])

    ddim = create_diffusion(str(steps))
    conds = cache["conds"]
    gts = cache["gts"]
    gt_np = cache["gt_np"]
    n = len(conds)
    mse_sum = ssim_sum = 0.0
    torch.manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, metric_batch):
            j = min(i + metric_batch, n)
            z = torch.randn(j - i, 3, 256, 256, device=device)
            mk = dict(
                y_callig=torch.tensor([c[0] for c in conds[i:j]], device=device),
                y_char=torch.tensor([c[1] for c in conds[i:j]], device=device),
                cfg_scale=sample_cfg,
            )
            samples = ddim.ddim_sample_loop(model.forward_with_cfg, z.shape, z,
                                            clip_denoised=False, model_kwargs=mk,
                                            device=device)
            dec = samples.clamp(-1, 1).float()
            gt = gts[i:j]
            mse_sum += F.mse_loss(dec, gt).item() * (j - i)
            for k in range(dec.shape[0]):
                p = (dec[k:k+1].squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
                g = gt_np[i + k]
                ssim_sum += _ssim_simple(p, g)

    mse = mse_sum / n
    ssim = ssim_sum / n
    result = {"step": step, "mse": round(mse, 5), "ssim": round(ssim, 4),
              "ts": datetime.datetime.now().isoformat()}
    with open(os.path.join(ckpt_dir, f"eval_auto_{step:07d}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)
    log(f"[eval] step {step}: MSE={mse:.5f} SSIM={ssim:.4f} ({time.time()-t0:.0f}s)")
    # 保存演示 sample (每 ckpt 一张 GT + 一张 pred 拼接)
    try:
        import numpy as np
        from PIL import Image
        _merge = []
        for k in range(min(4, n)):
            _merge.append(gt_np[k])
            _merge.append((dec[k].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0)
        canvas = np.concatenate(_merge, axis=1)
        Image.fromarray((canvas * 255).clip(0, 255).astype("uint8")).save(
            os.path.join(ckpt_dir, f"eval_px_{step:07d}.png"))
    except Exception as e:
        log(f"[eval] visual save failed: {e!r}")
    return result


# ---------------------------------------------------------------------------
# 轮询主循环
# ---------------------------------------------------------------------------
def read_active_ckpt_dir(results_dir):
    marker = os.path.join(results_dir, "_active_ckpt_dir.txt")
    if not os.path.exists(marker):
        return None
    with open(marker, encoding="utf-8") as f:
        return f.read().strip() or None


def load_state(ckpt_dir):
    sp = os.path.join(ckpt_dir, "pixel_eval_state.json")
    try:
        with open(sp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(ckpt_dir, state):
    with open(os.path.join(ckpt_dir, "pixel_eval_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="5script/results/px_s_scratch")
    ap.add_argument("--eval-csv", default="5script/eval100_top6.csv")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--interval", type=int, default=20, help="轮询间隔秒")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    cache = build_eval_cache(args)
    if cache is None:
        log("[fatal] eval csv missing; exit")
        sys.exit(1)

    model = None
    state = {}
    last_ckpt_dir = None
    done_set = set()

    while True:
        ckpt_dir = read_active_ckpt_dir(args.results_dir)
        if ckpt_dir and os.path.isdir(ckpt_dir):
            if ckpt_dir != last_ckpt_dir:
                state = load_state(ckpt_dir)
                done_set = set(state.get("done_steps", []))
                last_ckpt_dir = ckpt_dir
                log(f"[poll] ckpt_dir -> {ckpt_dir}")
            ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
            for cp in ckpts:
                try:
                    step = int(os.path.basename(cp).replace(".pt", ""))
                except ValueError:
                    continue
                if step in done_set:
                    continue
                log(f"[poll] new ckpt step {step}: {cp}")
                ck = torch.load(cp, map_location=args.device, weights_only=False)
                if model is None:
                    model = build_model(ck, device=args.device)
                else:
                    sd = ck.get("ema") or ck.get("delta")
                    missing, unexpected = model.load_state_dict(sd, strict=False)
                    log(f"[model] updated weights (missing={len(missing)})")
                if args.device == "cuda":
                    model = model.cuda()
                eval_one(model, ckpt_dir, step, cache,
                         {"steps": args.steps, "cfg": args.cfg, "seed": args.seed,
                          "batch": args.batch}, cp, device=args.device)
                done_set.add(step)
                state["done_steps"] = sorted(done_set)
                save_state(ckpt_dir, state)
                del ck
        if args.once:
            log("[done] --once; exit")
            sys.exit(0)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
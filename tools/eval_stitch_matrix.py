# -*- coding: utf-8 -*-
"""eval_stitch_matrix.py — v30 联合训练前的全面分段+组合评估 (忠实推理路径).

推理 100% 复刻 in_mem_eval: _get_cache(真实 conds=callig pair_id+glyph_id)
+ sample_latents(heun50, forward_with_cfg cfg=1.0) + fp32 VAE decode.

评估矩阵 (全部 84 字, strict84_aligned, 与 GT shards 零重叠):
  stage1 ckpts: v33 {20000, 25000, 30000}      -> 骨架质量 (墨比/dice@3px vs std & GT)
  stage2 ckpts: v32 {50000, 60000, 70000, 80000} -> 3 种 g 条件:
      std(输入照抄) / stage1@30k predskel / GT(oracle)
  拼接: 每 (s1, s2) 组合 -> strict ssim / follow dice@3px
  poster: 最优组合 + 最终组合 (30000x80000)
运行 ETA ~25 分钟 (4 条件 x 84 字 x Heun100NFE + 7 stage1/s2 组合加载)
"""
import os, sys, json, glob, csv, re, time
import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
TIME_SCALE = 1000.0
OUT = "assets/results/v30_stitch_matrix"


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def dice3(a, b):
    from scipy.ndimage import binary_dilation, generate_binary_structure
    st = generate_binary_structure(2, 2)
    ad = binary_dilation(a, structure=st, iterations=3)
    bd = binary_dilation(b, structure=st, iterations=3)
    inter = (a & bd).sum() + (b & ad).sum()
    return 2 * inter / max(a.sum() + b.sum(), 1)


def frag_ratio(mask):
    from scipy.ndimage import label
    _, n = label(mask)
    return float(n) / max(mask.sum() / 400, 1)


def _ssim(pred, gt):
    from scipy.ndimage import gaussian_filter
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x, y = pred.astype(np.float64), gt.astype(np.float64)
    mu_x, mu_y = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    sx = gaussian_filter(x * x, 1.5) - mu_x ** 2
    sy = gaussian_filter(y * y, 1.5) - mu_y ** 2
    sxy = gaussian_filter(x * y, 1.5) - mu_x * mu_y
    n = 121
    vx, vy, cxy = sx * n / (n - 1), sy * n / (n - 1), sxy * n / (n - 1)
    return float(np.mean(((2 * mu_x * mu_y + C1) * (2 * cxy + C2)) /
                         ((mu_x ** 2 + mu_y ** 2 + C1) * (vx + vy + C2))))


@th.no_grad()
def _decode_all(vae, lat, dev):
    lat = lat.to(dev) if not lat.is_cuda else lat
    imgs = []
    for s in range(0, lat.shape[0], 28):
        with th.autocast("cuda", enabled=False):
            d = vae.decode(lat[s:s + 28].float() / 0.18215).sample.float().cpu()
        imgs.append(((d.clamp(-1, 1) + 1) / 2).mean(1).numpy())
    return np.concatenate(imgs)


@th.no_grad()
def main():
    dev = th.device("cuda")
    from src.eval import model_io
    from src.eval.in_mem_eval import _get_vae, _get_cache
    from src.eval.inference import sample_latents, build_diffusion
    from types import SimpleNamespace
    from PIL import Image, ImageDraw
    th.backends.cuda.matmul.allow_tf32 = True

    rcfg = json.load(open(glob.glob(
        "assets/results/v32_stage2_img/20261001-062933*/resolved_config.json")[0], encoding="utf-8"))
    ev = SimpleNamespace(**{k: v for k, v in rcfg.items() if not k.startswith("_")})
    ev.eval_blend_alpha, ev.eval_cfg, ev.eval_steps = 0.0, 1.0, 50
    ev.eval_self_cond, ev.img_root = False, None
    CSV = "assets/eval_v13_strict84_aligned.csv"
    SHARDS_GT = "data/top10_style23/gt_skel_eval_strict84"
    vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").float()
    cache = _get_cache(CSV, 84, None, SHARDS_GT, ev)
    conds, noise = cache["conds"], cache["noise"]
    skels_gt = cache["skels_latent"].float().to(dev)
    n = cache["n"]
    diff = build_diffusion(50, "flow")
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    ids84 = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in rows]

    std_masks, gt_imgs, std_l = [], [], []
    for r in rows:
        p = r["std_path"] if os.path.isabs(r["std_path"]) else os.path.join("/root/Workspace/xy/DiT", r["std_path"])
        img = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
        std_masks.append(img < 0.5)
        x = th.from_numpy(1.0 - 2.0 * img)[None, None].repeat(1, 3, 1, 1).to(dev)
        with th.autocast("cuda", dtype=th.bfloat16):
            std_l.append((vae.encode(x).latent_dist.mode() * 0.18215).float()[0].cpu())
        gp = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join("/root/Workspace/xy/DiT", r["image_path"])
        gt_imgs.append(np.asarray(Image.open(gp).convert("L").resize((256, 256)), np.float32) / 255.0)
    std84 = th.stack(std_l).to(dev)
    std_masks = np.stack(std_masks)
    gt_imgs = np.stack(gt_imgs)
    print(f"[data] n={n}", flush=True)

    s1_paths = {st: sorted(glob.glob(f"assets/results/v33_stage1_xs/*/checkpoints/{st:07d}.pt"))[-1]
                for st in (20000, 25000, 30000)}
    s2_paths = {st: sorted(glob.glob(f"assets/results/v32_stage2_img/*/checkpoints/{st:07d}.pt"))[-1]
                for st in (50000, 60000, 70000, 80000)}

    def stage1_gen(path):
        s1 = model_io.load_model_from_ckpt(path, device=dev, use_ema=True)[0].eval().float()
        pred = sample_latents(s1, diff, noise, conds, 1.0, 28, dev, skel=skels_gt, seed=0).to(dev)
        imgs = _decode_all(vae, pred, dev)
        d_std = float(np.mean([dice3(imgs[i] < 0.5, std_masks[i]) for i in range(n)]))
        inks = float(np.mean([(imgs[i] < 0.5).mean() for i in range(n)]))
        d_gt = []
        for i in range(n):
            gp = f"data/top10_style23/gt_skel_png/{ids84[i]:06d}.png"
            if os.path.exists(gp):
                gtm = np.asarray(Image.open(gp).convert("L"), np.float32) / 255.0 < 0.5
                d_gt.append(dice3(imgs[i] < 0.5, gtm))
        del s1
        th.cuda.empty_cache()
        return pred, {"ink": round(inks, 4), "dice3_vs_std": round(d_std, 4),
                      "dice3_vs_gt": round(float(np.mean(d_gt)), 4)}

    def stage2_gen(path, g):
        s2 = model_io.load_model_from_ckpt(path, device=dev, use_ema=True)[0].eval().float()
        img = sample_latents(s2, diff, noise, conds, 1.0, 28, dev, skel=g, seed=0).to(dev)
        imgs = _decode_all(vae, img, dev)
        del s2
        th.cuda.empty_cache()
        return imgs

    results = {"stage1": {}, "stage2_conds": {}, "stitch": []}
    t00 = time.time()

    # ---- stage1 质量 (3 ckpt) ----
    preds = {}
    for st in sorted(s1_paths):
        pred, m = stage1_gen(s1_paths[st])
        preds[st] = pred
        results["stage1"][st] = m
        print(f"[s1 {st}] {m}", flush=True)
    std_vs_gt = float(np.mean([
        dice3(std_masks[i],
              (lambda p: np.asarray(Image.open(p).convert("L"), np.float32) / 255.0 < 0.5)(
                  f"data/top10_style23/gt_skel_png/{ids84[i]:06d}.png"))
        for i in range(n) if os.path.exists(f"data/top10_style23/gt_skel_png/{ids84[i]:06d}.png")]))
    results["std_copy_vs_gt_dice3"] = round(std_vs_gt, 4)
    print(f"[baseline] std 照抄 vs GT: dice@3px={std_vs_gt:.4f}", flush=True)

    # ---- stage2 条件响应 (4 ckpt x 3 条件) ----
    g_conds = {"std_copy": std84, "s1_30000": preds[30000], "gt_oracle": skels_gt}
    for s2_step in sorted(s2_paths):
        row = {}
        for cname, g in g_conds.items():
            imgs = stage2_gen(s2_paths[s2_step], g)
            row[cname] = {"ssim": round(float(np.mean([_ssim(imgs[i], gt_imgs[i])
                                                       for i in range(n)])), 4),
                          "frag": round(float(np.mean([frag_ratio(imgs[i] < 0.5)
                                                       for i in range(n)])), 2),
                          "follow_dice3": round(float(np.mean([dice3(imgs[i] < 0.5, std_masks[i])
                                                               for i in range(n)])), 4)}
            del imgs
            th.cuda.empty_cache()
        results["stage2_conds"][s2_step] = row
        print(f"[s2 {s2_step}] " + "  ".join(
            f"{k}: ssim={v['ssim']} frag={v['frag']} follow={v['follow_dice3']}"
            for k, v in row.items()), flush=True)

    # ---- 拼接矩阵 (3 s1 x 4 s2) ----
    for s1_step in sorted(s1_paths):
        pred_s1 = sample_latents(
            model_io.load_model_from_ckpt(s1_paths[s1_step], device=dev, use_ema=True)[0].eval().float(),
            diff, noise, conds, 1.0, 28, dev, skel=skels_gt, seed=0).to(dev)
        for s2_step in sorted(s2_paths):
            imgs = stage2_gen(s2_paths[s2_step], pred_s1)
            ss = [_ssim(imgs[i], gt_imgs[i]) for i in range(n)]
            fr = [frag_ratio(imgs[i] < 0.5) for i in range(n)]
            fd = [dice3(imgs[i] < 0.5, std_masks[i]) for i in range(n)]
            row = {"s1_step": s1_step, "s2_step": s2_step,
                   "ssim": round(float(np.mean(ss)), 4), "ssim_med": round(float(np.median(ss)), 4),
                   "frag": round(float(np.mean(fr)), 2),
                   "follow_dice3": round(float(np.mean(fd)), 4)}
            results["stitch"].append(row)
            print(f"[stitch {s1_step}x{s2_step}] ssim={row['ssim']} frag={row['frag']} "
                  f"follow={row['follow_dice3']}", flush=True)
            del imgs
            th.cuda.empty_cache()
        del pred_s1
        th.cuda.empty_cache()

    json.dump(results, open(os.path.join(OUT, "eval_matrix.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[saved] {OUT}/eval_matrix.json | 总耗时 {time.time()-t00:.0f}s", flush=True)

    # ---- poster ----
    best = max(results["stitch"], key=lambda x: x["ssim"])
    fin = [r for r in results["stitch"] if r["s1_step"] == 30000 and r["s2_step"] == 80000][0]
    cell = 128
    show = list(range(8))

    def bar(txt):
        im = Image.new("RGB", (10 * cell + 10, 26), (12, 12, 26))
        ImageDraw.Draw(im).text((6, 6), txt, fill=(255, 220, 120))
        return np.asarray(im, np.float32) / 255.0

    def line_of(masks):
        line = [np.asarray(Image.fromarray((m * 255).clip(0, 255).astype("uint8"))
                           .convert("RGB").resize((cell, cell)), np.float32) / 255.0
                for m in masks]
        return np.concatenate(line, axis=1)

    parts = [bar(f"v30 stitch eval | best={best['s1_step']}x{best['s2_step']} ssim={best['ssim']} "
                 f"frag={best['frag']} | final 30000x80000 ssim={fin['ssim']}")]
    s1_best_imgs = _decode_all(vae, preds[30000], dev)
    g_best = stage2_gen(s2_paths[best["s2_step"]], preds[30000])
    g_gt = stage2_gen(s2_paths[80000], skels_gt)
    g_fin = stage2_gen(s2_paths[80000], preds[30000])
    parts.append(bar("best 30000x" + str(best["s2_step"]) +
                     ": row1=std | row2=stage1骨架输出 | row3=拼接生成 | row4=GT真迹"))
    parts.append(line_of([std_masks[i] for i in show]))
    parts.append(line_of([s1_best_imgs[i] for i in show]))
    parts.append(line_of([g_best[i] for i in show]))
    parts.append(line_of([gt_imgs[i] for i in show]))
    parts.append(bar("final 30000x80000: row1=s2+GT oracle | row2=拼接(对照) | row3=GT真迹"))
    parts.append(line_of([g_gt[i] for i in show]))
    parts.append(line_of([g_fin[i] for i in show]))
    parts.append(line_of([gt_imgs[i] for i in show]))
    canvas = np.concatenate(parts, axis=0)
    Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8)).save(
        os.path.join(OUT, "poster.png"))
    print(f"[poster] {OUT}/poster.png", flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""eval_stage_stitch.py — v30 系列扎实评估: stage1/stage2 分段 + 拼接, 84 字全量.

A. stage1 (v33) 骨架质量: 生成骨架 vs (a) std 输入骨架 (b) GT 骨架
   dice@3px / IoU / 墨比 / 连通碎片 —— 回答"stage1 是否好于照抄 std"
B. stage2 (v32) 条件响应: g ∈ {std, stage1生成, GT} × 84 字 -> ssim/frag/lpips
   —— 回答"接口断裂有多深、断在哪个条件分布上"
C. 拼接端到端 = B 的 stage1生成 列
"""
import os, sys, json, glob, csv, re, time
import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
TIME_SCALE = 1000.0


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def dice3(a, b):
    """3px 容差 dice (骨架级有效判据, 06 号文档 §0)."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    st = generate_binary_structure(2, 2)
    ad = binary_dilation(a, structure=st, iterations=3)
    bd = binary_dilation(b, structure=st, iterations=3)
    inter = (a & bd).sum() + (b & ad).sum()
    return 2 * inter / max(a.sum() + b.sum(), 1)


def frag_ratio(mask):
    """连通碎片度: 连通域数 / (总墨像素/平均笔画面积) 的简化版 —— 连通域计数."""
    from scipy.ndimage import label
    _, n = label(mask)
    return float(n) / max(mask.sum() / 400, 1)


def main():
    dev = th.device("cuda")
    from src.eval import model_io
    from src.eval.in_mem_eval import _get_vae
    from src.utils.callig_script_map import map_callig_script
    from PIL import Image
    from scipy.ndimage import skeletonize

    GEN_CK = sorted(glob.glob("assets/results/v33_stage1_xs/*/checkpoints/0*.pt"))[-1]
    BAK_CK = sorted(glob.glob("assets/results/v32_stage2_img/*/checkpoints/0*.pt"))[-1]
    print(f"stage1={GEN_CK}\nstage2={BAK_CK}", flush=True)
    s1, _ = model_io.load_model_from_ckpt(GEN_CK, device=dev, use_ema=True)
    s2, _ = model_io.load_model_from_ckpt(BAK_CK, device=dev, use_ema=True)
    vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema")
    csmap = json.load(open("assets/callig_script_id_map_top10.json", encoding="utf-8"))

    rows = list(csv.DictReader(open("assets/eval_v13_strict84_aligned.csv", encoding="utf-8")))
    print(f"eval84 rows={len(rows)}", flush=True)

    # std 骨架 PNG -> latent (batch encode)
    std_l, gt_l = [], []
    for r in rows:
        p = r["std_path"] if os.path.isabs(r["std_path"]) else os.path.join("/root/Workspace/xy/DiT", r["std_path"])
        img = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
        x = th.from_numpy(1.0 - 2.0 * img)[None, None].repeat(1, 3, 1, 1).to(dev)
        with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
            std_l.append((vae.encode(x).latent_dist.mode() * 0.18215).float()[0].cpu())
    std84 = th.stack(std_l).to(dev)
    ys = th.tensor([int(map_callig_script(int(r["calligrapher_id"]), int(r["script_id"]), csmap))
                    for r in rows], device=dev)

    # GT 骨架 latent (oracle, 从 gt_skel_eval_strict84 shards)
    idx = {}
    for sp in sorted(glob.glob("data/top10_style23/gt_skel_eval_strict84/shard_*.npz")):
        with np.load(sp) as z:
            for j, iid in enumerate(z["img_ids"]):
                idx[int(iid)] = z["latents"][j].astype(np.float32)
    ids84 = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in rows]
    gt84 = th.stack([th.from_numpy(idx[i]) for i in ids84]).to(dev)

    def s1_sample(g, steps=25):
        b = g.shape[0]
        z = th.randn_like(g)
        ts = np.linspace(1.0, 0.0, steps + 1)
        yc = th.zeros_like(ys[:b])
        with th.no_grad():
            for k in range(steps):
                t = th.full((b,), float(ts[k]) * TIME_SCALE, device=dev)
                v = s1(z, t, y_callig=ys[:b], y_char=yc, g=g)
                if isinstance(v, tuple):
                    v = v[0]
                z = z + (float(ts[k + 1]) - float(ts[k])) * v.float()
        return z

    def decode_ink(lat):
        """latent -> (墨mask 256², 墨比)."""
        with th.no_grad():
            dec = vae.decode(lat / 0.18215).sample.float().cpu()
        g = ((dec.clamp(-1, 1) + 1) / 2).mean(1).numpy()
        return g < 0.5, g

    print("\n===== A. stage1 骨架质量 (84 字) =====", flush=True)
    t0 = time.time()
    pred_lat = []
    for s in range(0, len(rows), 16):
        pred_lat.append(s1_sample(std84[s:s + 16]))
    pred84 = th.cat(pred_lat)
    print(f"  stage1 采样 {time.time()-t0:.0f}s", flush=True)
    am_metrics = {"dice3_vs_std": [], "dice3_vs_gt": [], "ink_ratio": [], "frag": []}
    gt_png_dir = "data/top10_style23/gt_skel_png"
    for i, r in enumerate(rows):
        am, _ = decode_ink(pred84[i:i + 1])
        am = am[0]
        std_png = r["std_path"] if os.path.isabs(r["std_path"]) else os.path.join("/root/Workspace/xy/DiT", r["std_path"])
        std_m = np.asarray(Image.open(std_png).convert("L"), np.float32) < 128
        gp = os.path.join(gt_png_dir, f"{ids84[i]:06d}.png")
        gt_m = (np.asarray(Image.open(gp).convert("L"), np.float32) < 128
                if os.path.exists(gp) else None)
        am_metrics["dice3_vs_std"].append(dice3(am, std_m))
        if gt_m is not None:
            am_metrics["dice3_vs_gt"].append(dice3(am, gt_m))
        am_metrics["ink_ratio"].append(float(am.mean() / max(std_m.mean(), 1e-6)))
        am_metrics["frag"].append(frag_ratio(am))
    print(f"  stage1 vs std 照抄: dice@3px = {np.mean(am_metrics['dice3_vs_std']):.4f}", flush=True)
    if am_metrics["dice3_vs_gt"]:
        print(f"  stage1 vs GT 骨架:  dice@3px = {np.mean(am_metrics['dice3_vs_gt']):.4f}", flush=True)
    print(f"  墨比(生成/std): {np.mean(am_metrics['ink_ratio']):.3f} (1=正常, >>1 过墨)", flush=True)
    print(f"  生成骨架 frag: {np.mean(am_metrics['frag']):.1f} (连通域计数, 低=好)", flush=True)
    # std 自身 vs GT (基线: 照抄的距离)
    stdvgt = []
    for i, r in enumerate(rows):
        gp = os.path.join(gt_png_dir, f"{ids84[i]:06d}.png")
        if os.path.exists(gp):
            std_png = r["std_path"] if os.path.isabs(r["std_path"]) else os.path.join("/root/Workspace/xy/DiT", r["std_path"])
            std_m = np.asarray(Image.open(std_png).convert("L"), np.float32) < 128
            gt_m = np.asarray(Image.open(gp).convert("L"), np.float32) < 128
            stdvgt.append(dice3(std_m, gt_m))
    print(f"  [基线] std 照抄 vs GT: dice@3px = {np.mean(stdvgt):.4f}", flush=True)

    print("\n===== B/C. stage2 条件响应 + 拼接 (84 字) =====", flush=True)
    conds = {"std": std84, "stage1_pred": pred84, "gt_oracle": gt84}
    out = {}
    for name, g in conds.items():
        m_ssim, m_frag, m_lpips = [], [], []
        for s in range(0, len(rows), 8):
            b = min(8, len(rows) - s)
            x = th.randn(b, 4, 32, 32, device=dev)
            ts = np.linspace(1.0, 0.0, 51)
            zz = x
            with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
                for k in range(50):
                    t = th.full((b,), float(ts[k]) * TIME_SCALE, device=dev)
                    v = s2(zz, t, y_callig=ys[s:s + b], y_char=th.zeros(b, dtype=th.long, device=dev),
                           g=g[s:s + b])
                    if isinstance(v, tuple):
                        v = v[0]
                    zz = zz + (float(ts[k + 1]) - float(ts[k])) * v.float()
            with th.no_grad():
                dec = vae.decode(zz / 0.18215).sample.float().cpu()
            imgs = ((dec.clamp(-1, 1) + 1) / 2).mean(1).numpy()
            for j in range(b):
                r = rows[s + j]
                gp = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join("/root/Workspace/xy/DiT", r["image_path"])
                gt_img = np.asarray(Image.open(gp).convert("L").resize((256, 256)), np.float32) / 255.0
                pred = imgs[j]
                m_ssim.append(_ssim(pred, gt_img))
                m_frag.append(frag_ratio(pred < 0.5))
                m_lpips.append(0.0)
            del dec, zz
            th.cuda.empty_cache()
        out[name] = (np.mean(m_ssim), np.mean(m_frag))
        print(f"  g={name:>12}: img ssim={out[name][0]:.4f} frag={out[name][1]:.2f}", flush=True)

    print("\n===== 汇总 =====", flush=True)
    print(json.dumps({"stage1": {k: float(np.mean(v)) for k, v in am_metrics.items()},
                      "std_baseline_vs_gt": float(np.mean(stdvgt)),
                      "stage2_conds": {k: {"ssim": round(v[0], 4), "frag": round(v[1], 2)}
                                        for k, v in out.items()}},
                     ensure_ascii=False, indent=1))
    json.dump({"stage1": {k: [float(x) for x in v] for k, v in am_metrics.items()},
               "std_baseline_vs_gt": float(np.mean(stdvgt)),
               "stage2_conds": {k: {"ssim": float(v[0]), "frag": float(v[1])}
                                for k, v in out.items()}},
              open("assets/results/v30_stage_stitch_eval.json", "w"), indent=1)


def _ssim(pred, gt, win=11):
    """与 in_mem_eval 相同口径的高斯窗 SSIM (灰度)."""
    from scipy.ndimage import gaussian_filter
    C1, C2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    x = pred.astype(np.float64)
    y = gt.astype(np.float64)
    mu_x = gaussian_filter(x, 1.5)
    mu_y = gaussian_filter(y, 1.5)
    sx = gaussian_filter(x * x, 1.5) - mu_x ** 2
    sy = gaussian_filter(y * y, 1.5) - mu_y ** 2
    sxy = gaussian_filter(x * y, 1.5) - mu_x * mu_y
    n = win ** 2
    unbiased = max(n - 1, 1)
    vx = sx * n / unbiased
    vy = sy * n / unbiased
    cxy = sxy * n / unbiased
    num = (2 * mu_x * mu_y + C1) * (2 * cxy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (vx + vy + C2)
    return float(np.mean(num / den))


if __name__ == "__main__":
    main()

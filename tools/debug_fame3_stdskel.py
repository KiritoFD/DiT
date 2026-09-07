#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_fame3_stdskel.py — v10b-stdskel-fame3 的 std-g 通路诊断 (GPU).

针对当前实验 (训练 g = std 骨架 latent, 无 GT 实例信息) 做数值诊断, 回答:
  1. g 来源一致性 : 读到的 g 是否真是 std_skel1 (非 GT 泄漏), 覆盖几何
  2. glyph_scale   : 训练后 scale 学到多少 (init 0.4), 是否漂离/趋零 (g 通路生命信号)
  3. g 注入作用    : 有/无 g 的整网输出相对差 (应 >~2%)
  4. 逐层 g 衰减   : 单层 token-add 穿过 12 block 后被稀释到多少 (每层有/无 g 相对差)
  5. follow IoU3   : 生成图 vs 输入 std 骨架的 IoU (fame3 真正的"写对"指标;
                     report 里的 skel_iou 是 vs GT 图, std-g 场景无意义)

用法 (远程 4090):
  python tools/debug_fame3_stdskel.py --ckpt 52500 --n 32
  python tools/debug_fame3_stdskel.py --ckpt /abs/path/to/ckpt.pt --n 32
"""
import os, sys, json, argparse, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.chdir(_root)

import numpy as np
import torch


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def _locate_ckpt(spec):
    if os.path.isfile(spec):
        return spec
    import glob
    cands = sorted(glob.glob("5script/results/v10b_stdskel_fame3/*/checkpoints/*.pt"),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not cands:
        raise FileNotFoundError("找不到 v10b_stdskel_fame3 的 checkpoints")
    if spec.isdigit():
        hit = next((c for c in cands if int(os.path.basename(c).split(".")[0]) == int(spec)), None)
        if hit:
            return hit
    return cands[-1]


def _skeletonize(binary):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(binary)
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        sk = np.zeros_like(binary); im = binary.copy()
        st = generate_binary_structure(2, 2)
        while im.any():
            er = binary_erosion(im, structure=st); sk |= im & ~er; im = er
        return sk


def _follow_iou(gen_img_01, in_img_01):
    """gen/输入图 (灰度 0~1, 白底黑字) → 骨架 IoU (1px 严格 + 3px 容差)."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    def sk(mask):
        return _skeletonize(mask)
    def d3(m):
        return binary_dilation(m, structure=generate_binary_structure(2, 2), iterations=3)
    s1 = sk(gen_img_01 < 0.5); s2 = sk(in_img_01 < 0.5)
    iou1 = (s1 & s2).sum() / max((s1 | s2).sum(), 1)
    iou3 = (d3(s1) & d3(s2)).sum() / max((d3(s1) | d3(s2)).sum(), 1)
    return float(iou1), float(iou3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="step 数字或 ckpt 绝对路径")
    ap.add_argument("--n", type=int, default=32, help="follow IoU 采样样本数")
    ap.add_argument("--n-batch", type=int, default=4, help="梯度/注入作用诊断 batch 数")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="5script/results/v10b_stdskel_fame3/debug_stdskel.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    ck_path = _locate_ckpt(args.ckpt)
    step = int(os.path.basename(ck_path).split(".")[0])
    print(f"[ckpt] {ck_path} (step {step})", flush=True)

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    if isinstance(a, argparse.Namespace):
        a = vars(a)

    from src.model import DiT_2Cond_models
    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)), attn_impl="sdpa")
    use_g = bool(a.get("skel_as_glyph_cond", False) or a.get("w_glyph_cond", False))
    model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
        input_size=int(a.get("image_size", 256)) // 8,
        num_calligraphers=int(a.get("num_calligraphers", 1013)),
        num_characters=int(a.get("num_characters", 35130)),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim", 128)),
        char_embed_dim=int(a.get("char_embed_dim", 384)),
        char_proj_mode=a.get("char_proj_mode", "mlp"),
        freeze_char_table=bool(a.get("freeze_char_table", True)),
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.3,
        cond_drop_which_glyph_prob=0.85, use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=use_g,
        use_char_cond=not bool(a.get("no_char_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
        glyph_drop_prob=float(a.get("glyph_drop_prob", 0.0)),
        glyph_inject_layers=int(a.get("glyph_inject_layers", 0)), **arch).to(dev)
    sd = _strip(ck.get("ema") or ck.get("model") or ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load] miss={len(miss)} unexp={len(unexp)}", flush=True)

    res = {"ckpt": ck_path, "step": step,
           "skel_latent_shards_dir": a.get("skel_latent_shards_dir"),
           "glyph_drop_prob": float(a.get("glyph_drop_prob", 0)),
           "glyph_inject_layers": int(a.get("glyph_inject_layers", 0))}

    # ── 1. glyph_scale 生命信号 ──
    gs = getattr(model, "glyph_scale", None)
    if gs is not None:
        print(f"[glyph_scale] = {gs.item():.4f} (init {a.get('glyph_scale_init')}) "
              f"requires_grad={gs.requires_grad}", flush=True)
        res["glyph_scale"] = round(float(gs.item()), 5)
        res["glyph_scale_init"] = float(a.get("glyph_scale_init", 0.4))

    # ── 数据 ──
    from src.utils.latent_dataset import MCCDLatentDataset
    from torch.utils.data import DataLoader, Subset
    csv_path = a.get("eval_csv") or a.get("data_csv")
    ds = MCCDLatentDataset(
        csv_file=csv_path, latent_shards_dir=a.get("latent_shards_dir"),
        img_root=a.get("img_root"), image_size=int(a.get("image_size", 256)),
        preload=True, load_image=True, use_glyph_cond=False,
        skel_latent_shards_dir=(a.get("skel_latent_shards_dir") if use_g else None))
    n_use = min(args.n_batch * args.batch, len(ds))
    dl = DataLoader(Subset(ds, list(range(n_use))), batch_size=args.batch,
                    shuffle=False, num_workers=2, drop_last=True)
    print(f"[data] csv={csv_path} use={n_use}", flush=True)

    # ── 2. g 来源一致性: 覆盖 + 统计 ──
    batch0 = next(iter(dl))
    g0 = batch0["skel_latent"].float() if use_g else None
    if g0 is not None:
        hit = int((g0.view(g0.shape[0], -1).abs().sum(1) != 0).sum())
        print(f"[g来源] shard={a.get('skel_latent_shards_dir')} 覆盖 {hit}/{g0.shape[0]} "
              f"(std 骨架应全非零)", flush=True)
        res["g_coverage"] = f"{hit}/{g0.shape[0]}"
        res["g_mean_abs"] = round(float(g0.abs().mean().item()), 5)
        res["g_frac_nz"] = round(float((g0 != 0).float().mean().item()), 5)

    # ── 3. g 注入作用 + 4. 逐层衰减 (eval, 干净测量) ──
    model.eval()
    with torch.no_grad():
        b = next(iter(dl))
        xb = b["latent"][:8].to(dev)
        tb = torch.rand(8, device=dev)
        ycb = b["y_callig"][:8].to(dev)
        yhb = b["y_char"][:8].to(dev)
        gb = b["skel_latent"][:8].to(dev).float() if use_g else None

        feats_g, feats_nog = {}, {}
        hooks = []
        for i, blk in enumerate(model.blocks):
            hooks.append(blk.register_forward_hook(
                (lambda ii: lambda m, inp, out: feats_g.__setitem__(ii, out))(i)))
        o_g = model(xb, tb, ycb, yhb, g=gb).float()
        for h in hooks:
            h.remove()

        hooks = []
        for i, blk in enumerate(model.blocks):
            hooks.append(blk.register_forward_hook(
                (lambda ii: lambda m, inp, out: feats_nog.__setitem__(ii, out))(i)))
        o_nog = model(xb, tb, ycb, yhb, g=None).float()
        for h in hooks:
            h.remove()

        rel_all = (o_nog - o_g).norm() / o_g.norm().clamp_min(1e-8)
        print(f"\n[g注入作用] 有/无 g 全输出相对差 = {rel_all:.5f} "
              f"({'条件生效' if rel_all > 0.02 else '<<< 条件几乎无影响'})", flush=True)
        res["g_out_diff"] = round(float(rel_all), 5)

        print("\n[逐层 g 衰减] (相对差 = ||block_out(无g) - block_out(g)|| / ||block_out(g)||)")
        decays = []
        for i in range(len(model.blocks)):
            fg = feats_g[i].float(); fn = feats_nog[i].float()
            r = (fn - fg).norm() / fg.norm().clamp_min(1e-8)
            decays.append(float(r))
            print(f"  block{i:02d}: {r:.5f}", flush=True)
        res["per_block_g_rel"] = [round(d, 5) for d in decays]

    # ── 5. follow IoU3 (vs 输入 std 骨架) ──
    from src.eval.inference import load_eval_vae, sample_latents, build_diffusion
    vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
    diff = build_diffusion(int(a.get("eval_steps", 50)), "flow")
    n_follow = min(args.n, len(ds))
    sub = DataLoader(Subset(ds, list(range(n_follow))), batch_size=args.batch,
                     shuffle=False, num_workers=2, drop_last=False)
    cfg = float(a.get("eval_cfg", a.get("gpu_eval_cfg", 0.7)))
    iou1s, iou3s = [], []
    print(f"\n[follow IoU] vs 输入 std 骨架 (n={n_follow}, cfg={cfg})", flush=True)
    for bi, batch in enumerate(sub):
        skels = batch["skel_latent"].float()          # std 骨架 latent
        yc = batch["y_callig"]
        # 输入 std 骨架可视化 (decode → 灰度)
        with torch.no_grad():
            in_imgs = vae.decode(skels.to(dev) / 0.18215).sample
        in_imgs = ((in_imgs.clamp(-1, 1) + 1) / 2).mean(1).cpu().numpy()
        for k in range(skels.shape[0]):
            noise = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(0))
            conds = [(int(yc[k].item()), 0)]
            lat = sample_latents(model, diff, noise, conds, cfg, 1, dev,
                                 skel=skels[k].unsqueeze(0), seed=0)
            with torch.no_grad():
                gi = vae.decode(lat.to(dev) / 0.18215).sample[0]
            gen = ((gi.clamp(-1, 1) + 1) / 2).mean(0).cpu().numpy()
            i1, i3 = _follow_iou(gen, in_imgs[k])
            iou1s.append(i1); iou3s.append(i3)
        print(f"  batch{bi}: {len(iou1s)} 样本累计", flush=True)

    if iou3s:
        res["follow_n"] = len(iou3s)
        res["follow_iou1_mean"] = round(float(np.mean(iou1s)), 4)
        res["follow_iou3_mean"] = round(float(np.mean(iou3s)), 4)
        res["follow_iou3_median"] = round(float(np.median(iou3s)), 4)
        print(f"\n[follow 结果] IoU1 mean={res['follow_iou1_mean']:.4f} "
              f"IoU3 mean={res['follow_iou3_mean']:.4f} median={res['follow_iou3_median']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
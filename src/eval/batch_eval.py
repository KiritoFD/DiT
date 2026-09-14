#!/usr/bin/env python3
"""eval_stdskel_batch.py — GPU 批量评测 std-skel 系列全部 ckpt (seen + strict).

对 results_dir 下所有 ckpt, 在给定评测集上: GPU heun 采样 -> VAE decode ->
逐样本 MSE/SSIM/LPIPS. 输出 (写在 results_dir 原位置, 追加幂等):
  eval_stdskel_batch.csv    逐样本原始数据 (exp,step,set,idx,img_id,char,mse,ssim,lpips)
  eval_stdskel_summary.csv  每 ckpt x 集汇总 (mean + P10/Q1/med/Q3/P90)

用法:
  python tools/eval/eval_stdskel_batch.py --results-dir assets/results/v10b_stdskel_fame3 \
      --sets seen:assets/eval_seen_v10.csv:10 strict:assets/eval_fame3_strict_clean_v9.csv:237
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)
sys.stdout.reconfigure(encoding="utf-8")


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--sets", nargs="+",
                    default=["seen:assets/eval_seen_v10.csv:10",
                             "strict:assets/eval_fame3_strict_clean_v9.csv:237"])
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dit-batch", type=int, default=50)
    ap.add_argument("--vae-batch", type=int, default=25)
    ap.add_argument("--include-steps", type=str, default="",
                    help="逗号分隔 step 列表 (空=全部); 如 180000,200000,250000")
    ap.add_argument("--disable-callig-style", action="store_true",
                    help="强制 callig_style_attn=False 建模 (消融: 风格注入对指标的贡献)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="in-mem eval 设备 (cuda=GPU, cpu=CPU)")
    ap.add_argument("--ckpt-override", default="",
                    help="只评测指定 ckpt 路径 (batch 扫描/单点复评)")
    ap.add_argument("--skel-shards", default="",
                    help="覆盖 ckpt args 里的 skel_latent_shards_dir (g 条件); "
                         "用于修复 shard 覆盖不到 eval id 时 g=零 的静默失效")
    ap.add_argument("--save-samples", action="store_true",
                    help="落盘 g{i}/gt{i}.png 到 eval_samples_ctrl/ (供 poster)")
    ap.add_argument("--self-cond", action="store_true", default=False,
                    help="启用两遍自条件采样 (Self-Conditioning)")
    ap.add_argument("--blend-alpha", type=float, default=0.0,
                    help="自条件骨架混合系数 (0=纯预测骨架, 0.5=各半)")
    ap.add_argument("--force", action="store_true", default=False,
                    help="强制重新评测 (忽略已有 summary 记录)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.set_grad_enabled(False)

    cks = sorted(glob.glob(f"{args.results_dir}/*/checkpoints/[0-9]*.pt"),
                 key=lambda p: int(os.path.basename(p).split(".")[0]))
    if args.ckpt_override:
        cks = [args.ckpt_override]
    if not cks:
        print("[batch] no ckpts"); return 1
    if args.include_steps:
        keep = {int(s) for s in args.include_steps.split(",") if s.strip()}
        cks = [p for p in cks
               if int(os.path.basename(p).split(".")[0]) in keep]
    exp = os.path.basename(args.results_dir.rstrip("/"))
    print(f"[batch] exp={exp} ckpts={len(cks)}")

    # ── 模型/VAE 只建一次 (架构取自首个 ckpt args) ──────────────────────
    from src.model import DiT_2Cond_models
    from src.eval.inference import (make_eval_cache, load_eval_vae, sample_latents,
                                    sample_latents_self_cond,
                                    _mse, _ssim, build_diffusion)

    ck0 = torch.load(cks[0], map_location="cpu", weights_only=False)
    a = ck0.get("args", {})
    if not isinstance(a, dict):
        import argparse as _ap
        a = vars(a) if isinstance(a, _ap.Namespace) else {}
    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)),
                attn_impl=a.get("attn_impl", "sdpa"))
    # IDS 组件码本: 若 ckpt 用了 IDS, 需要复现 char_id->char 映射
    _ids_c2c = None
    if a.get("use_ids_char_embedder"):
        _ic = a.get("ids_char_map_csv")
        if _ic and os.path.isfile(_ic):
            from src.model.ids_embedder import build_char_id_map_from_csv
            _ids_c2c = build_char_id_map_from_csv(_ic)
    model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
        num_calligraphers=int(a.get("num_calligraphers") or 1013),
        num_characters=int(a.get("num_characters") or 35130),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim") or 128),
        char_embed_dim=int(a.get("char_embed_dim") or 384),
        char_proj_mode=(a.get("char_proj_mode") or "mlp"),
        freeze_char_table=bool(a.get("freeze_char_table", False)),
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        cond_drop_which_glyph_prob=0.5, use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=True, use_char_cond=not bool(a.get("no_char_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init") or 0.4),
        glyph_drop_prob=0.0,
        glyph_embedder_depth=int(a.get("glyph_embedder_depth") or 0),
        glyph_inject_layers=int(a.get("glyph_inject_layers") or 0),
        callig_style_attn=bool(a.get("callig_style_attn", False)) and not args.disable_callig_style,
        callig_n_style=int(a.get("callig_n_style") or 8),
        callig_spatial=bool(a.get("callig_spatial", False)),
        callig_spatial_rank=int(a.get("callig_spatial_rank") or 64),
        style_token_n=int(a.get("style_token_n") or 0),
        style_role_init=float(a.get("style_role_init") or 0.02),
        glyph_in_channels=4,
        in_channels=(int(a.get("latent_channels") or 4)
                     + 4 * len([s for s in str(a.get("aux_latent_shards_dirs") or "").split(",") if s])),
        image_channels=int(a.get("latent_channels") or 4),
        use_ids_char_embedder=bool(a.get("use_ids_char_embedder", False)),
        ids_file=a.get("ids_file"),
        char_id_to_char=_ids_c2c,
        use_std_dino_char_embedder=bool(a.get("use_std_dino_char_embedder", False)),
        std_dino_table_path=a.get("std_dino_table_path"),
        chars_per_script=int(a.get("chars_per_script") or 7026),
        glyph_inject_mode=a.get("glyph_inject_mode", "adaln"), **arch).to(dev).eval()
    # 冻结书家表: 复现 null_embed 独立参数结构 (与 ckpt state_dict 对齐)
    if a.get("freeze_callig_table"):
        model.y_callig_embedder.freeze_table()
    vae = load_eval_vae(dev, "data/pretrained/sd-vae-ft-ema")
    diff = build_diffusion(args.steps, "flow")
    shards = args.skel_shards or a.get("skel_latent_shards_dir") or "data/skel/std_skel1_latents_fame3_v8"
    img_root = a.get("gpu_eval_img_root") or a.get("img_root")
    sf = float(a.get("vae_scaling_factor", 0.18215))
    # 书家词表收紧: y_callig 与训练数据层同一张映射表
    cmap = None
    if a.get("callig_id_map"):
        from src.utils.callig_map import load_callig_id_map
        _cm_path = a["callig_id_map"]
        if not os.path.isabs(_cm_path) and not os.path.exists(_cm_path):
            _cm_path = os.path.join(_root, _cm_path)
        cmap, _ = load_callig_id_map(_cm_path)

    try:
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    except Exception as e:
        print(f"[batch] lpips 不可用 ({e}), 跳过")
        lpips_fn = None

    # ── 评测集缓存 (跨 ckpt 复用: 同 noise/conds/g, 保证可比) ────────────
    sets = {}
    for spec in args.sets:
        name, csvp, n = spec.split(":")
        n = int(n)
        rows = list(csv.DictReader(open(csvp, encoding="utf-8")))[:n]
        cache = make_eval_cache(csvp, img_root, None, 256, n, 8, 4, sf,
                                skel_latent_shards_dir=shards, callig_id_map=cmap)
        meta = []
        for r in rows:
            m = re.search(r"(\d+)\.png", r["image_path"])
            meta.append((int(m.group(1)) if m else -1, r.get("character", ""),
                         r.get("script", "")))
        sets[name] = {"cache": cache, "meta": meta, "n": len(rows), "csv": csvp}
        print(f"[batch] set={name} n={len(rows)} csv={csvp}")

    # ── 幂等: 读已有 summary, 跳过已完成的 (step,set) ────────────────────
    sum_path = f"{args.results_dir}/eval_stdskel_summary.csv"
    raw_path = f"{args.results_dir}/eval_stdskel_batch.csv"
    done = set()
    if os.path.exists(sum_path):
        for r in csv.DictReader(open(sum_path, encoding="utf-8")):
            done.add((int(r["step"]), r["set"]))
    new_sum = not os.path.exists(sum_path)
    new_raw = not os.path.exists(raw_path)
    f_sum = open(sum_path, "a", newline="", encoding="utf-8")
    f_raw = open(raw_path, "a", newline="", encoding="utf-8")
    w_sum = csv.writer(f_sum)
    w_raw = csv.writer(f_raw)
    if new_sum:
        w_sum.writerow(["exp", "step", "set", "n", "ssim_mean", "ssim_p10", "ssim_q1",
                        "ssim_med", "ssim_q3", "ssim_p90", "mse_mean", "lpips_mean"])
    if new_raw:
        w_raw.writerow(["exp", "step", "set", "idx", "img_id", "char", "script",
                        "mse", "ssim", "lpips"])

    for ck in cks:
        step = int(os.path.basename(ck).split(".")[0])
        todo = list(sets.keys()) if getattr(args, "force", False) else [s for s in sets if (step, s) not in done]
        if not todo:
            continue
        d = torch.load(ck, map_location="cpu", weights_only=False)
        sd = _strip(d.get("ema") or d.get("model") or d)
        if args.disable_callig_style:
            # 消融模式: 丢弃 ckpt 里的 callig_style 系键 (模块已被禁用), 其余照常
            sd = {k: v for k, v in sd.items()
                  if "callig_style" not in k and "callig_basis" not in k}
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert len(unexp) == 0, f"step{step} unexpected={len(unexp)[:3] if unexp else 0}"

        for name in todo:
            S = sets[name]
            cache, n = S["cache"], S["n"]
            t0 = time.time()
            if getattr(args, "self_cond", False):
                lat = sample_latents_self_cond(
                    model, diff, cache["noise"], cache["conds"],
                    args.cfg, args.dit_batch, dev,
                    skel=cache["skels_latent"], seed=0,
                    blend_alpha=args.blend_alpha)
            else:
                lat = sample_latents(
                    model, diff, cache["noise"], cache["conds"],
                    args.cfg, args.dit_batch, dev,
                    skel=cache["skels_latent"], seed=0)
            t_s = time.time() - t0
            # decode in memory (bf16 autocast: 共 GPU 训练时省一半激活显存)
            gts = (cache["gts"].to(dev) + 1) / 2
            preds = torch.empty_like(gts)
            for i in range(0, n, args.vae_batch):
                j = min(i + args.vae_batch, n)
                _lat = lat[i:j].to(dev)
                if _lat.shape[1] > 4:      # aux 目标通道: 只解码图像 4 通道
                    _lat = _lat[:, :4]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dec = vae.decode(_lat / sf).sample
                preds[i:j] = (dec.clamp(-1, 1) + 1) / 2
            pred_np = preds.cpu().numpy().transpose(0, 2, 3, 1)   # -> (n,H,W,C) 与 _ssim/_lpips 对齐
            gt_np = gts.cpu().numpy().transpose(0, 2, 3, 1)
            if args.save_samples:
                # 落盘样本供 poster: seen->g/, strict->strict50/
                from PIL import Image as _Img
                _sub = "g" if name in ("seen", "g") else ("strict50" if name == "strict" else name)
                _sd = os.path.join(os.path.dirname(os.path.dirname(ck)), "eval_samples_ctrl",
                                   f"step{step:07d}", _sub)
                os.makedirs(_sd, exist_ok=True)
                for i in range(n):
                    _Img.fromarray((pred_np[i] * 255).astype(np.uint8)).save(
                        os.path.join(_sd, f"g{i}.png"))
                    _Img.fromarray((gt_np[i] * 255).astype(np.uint8)).save(
                        os.path.join(_sd, f"gt{i}.png"))
                print(f"[batch] step{step} {name}: saved {n} samples -> {_sub}/", flush=True)
            ssims, mses, lp_list = [], [], []
            for i in range(n):
                mses.append(_mse(pred_np[i], gt_np[i]))
                ssims.append(_ssim(pred_np[i], gt_np[i]))
                if lpips_fn is not None:
                    p = torch.from_numpy(pred_np[i].transpose(2, 0, 1)[None] * 2 - 1).to(dev)
                    g = torch.from_numpy(gt_np[i].transpose(2, 0, 1)[None] * 2 - 1).to(dev)
                    lp_list.append(float(lpips_fn(p, g).mean().item()))
            ssim = np.array(ssims)
            q10, q25, q50, q75, q90 = np.percentile(ssim, [10, 25, 50, 75, 90])
            iid_s = [m0 for m0, _, _ in S["meta"]]
            ch_s = [c for _, c, _ in S["meta"]]
            scr_s = [s0 for _, _, s0 in S["meta"]]
            for i in range(n):
                w_raw.writerow([exp, step, name, i, iid_s[i], ch_s[i], scr_s[i],
                                round(mses[i], 6), round(ssims[i], 6),
                                round(lp_list[i], 6) if lp_list else ""])
            w_sum.writerow([exp, step, name, n, round(float(ssim.mean()), 6),
                            round(float(q10), 6), round(float(q25), 6),
                            round(float(q50), 6), round(float(q75), 6),
                            round(float(q90), 6), round(float(np.mean(mses)), 6),
                            round(float(np.mean(lp_list)), 6) if lp_list else ""])
            f_sum.flush(); f_raw.flush()
            print(f"[batch] step{step} {name}: ssim={ssim.mean():.4f} "
                  f"P10={q10:.4f} med={q50:.4f} Q3={q75:.4f} ({t_s:.0f}s)", flush=True)
    f_sum.close(); f_raw.close()
    if dev.type == "cuda":
        print(f"[batch] peak GPU mem = {torch.cuda.max_memory_allocated() / 1024**3:.2f}G "
              f"(device={dev}, dit_batch={args.dit_batch}, vae_batch={args.vae_batch})", flush=True)
    print("BATCH_EVAL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

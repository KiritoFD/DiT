#!/usr/bin/env python3
"""skel_follow_gpu.py — GPU 版新字骨架遵循度测试 (系统性地跑多模型/多ckpt)。

与 handwrite_follow_test.py 同协议 (同 10 新字同 seed 同条件), 但:
  * GPU 采样 (cu121, 快 20-30x)
  * 支持任意模型: v10b (无char) / v10a (有char) / v8e (两阶段)
  * 支持 --ckpt 指定 (多训练点画曲线)
  * 输出 flat json + PNG 逐字落盘

用法:
  python tools/eval/skel_follow_gpu.py --model v10b --ckpt 82500 --tag v10b_82500 --n 30
  python tools/eval/skel_follow_gpu.py --model v8e --tag v8e_22500
"""
import os, sys, csv, time, argparse, json, glob, random
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from PIL import Image


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="assets/results/skel_follow_gpu")
    ap.add_argument("--model", choices=["v10b", "v10a", "v8e"], default="v10b")
    ap.add_argument("--ckpt", default="", help="v10b/v10a: step 或路径; v8e: 忽略(固定22500)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cfg", type=float, default=0.7, help="callig 轴 CFG (与 eval 协议一致)")
    ap.add_argument("--cfg-g", type=float, default=0.0,
                    help="g 轴 CFG: 0=关(历史协议, g 两分支全给); >0 时 uncond 分支 g=0, "
                         "eps = eps(g=0) + cfg_g * (eps(g) - eps(g=0))。依赖训练期 glyph_drop")
    args = ap.parse_args()

    dev = torch.device("cuda")
    arch = dict(norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
                rope_theta=100.0, attn_impl="sdpa")

    from src.model import DiT_2Cond_models
    from src.utils import get_glyph_lookup_v2
    from src.eval.inference import load_eval_vae, sample_latents, build_diffusion

    # ---- 模型加载 ----
    if args.model == "v10b":
        cks = sorted(glob.glob("assets/results/v10b_skel_only_pretrain/*/checkpoints/*.pt"),
                     key=lambda p: int(os.path.basename(p).split(".")[0]))
        ck = args.ckpt if os.path.isfile(args.ckpt) else \
            next((c for c in cks if int(os.path.basename(c).split(".")[0]) == int(args.ckpt)),
                 cks[-1])
        step = int(os.path.basename(ck).split(".")[0])
        print(f"[v10b] ckpt={ck} step={step}", flush=True)
        d = torch.load(ck, map_location="cpu", weights_only=False)
        a = vars(d["args"]) if isinstance(d.get("args"), argparse.Namespace) else d["args"]
        model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
            num_calligraphers=int(a.get("num_calligraphers", 1013)),
            num_characters=int(a.get("num_characters", 35130)),
            condition_fusion="factorized_add", callig_embed_dim=128,
            char_embed_dim=384, char_proj_mode="mlp", freeze_char_table=False,
            cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
            cond_drop_which_glyph_prob=0.5, use_checkpoint=False, learn_sigma=False,
            use_glyph_cond=True, use_char_cond=False,
            glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
            glyph_embedder_depth=int(a.get("glyph_embedder_depth", 0)),
            glyph_inject_layers=int(a.get("glyph_inject_layers", 0)),
            callig_style_attn=bool(a.get("callig_style_attn", False)),
            callig_n_style=int(a.get("callig_n_style", 8)),
            glyph_inject_mode=a.get("glyph_inject_mode", "adaln"), **arch).to(dev)
        model.load_state_dict(_strip(d.get("ema") or d.get("model") or d), strict=False)
        model.eval()
    elif args.model == "v10a":
        cks = sorted(glob.glob("assets/results/v10a_skel_cond_pretrain/*/checkpoints/*.pt"),
                     key=lambda p: int(os.path.basename(p).split(".")[0]))
        ck = args.ckpt if os.path.isfile(args.ckpt) else \
            next((c for c in cks if int(os.path.basename(c).split(".")[0]) == int(args.ckpt)),
                 cks[-1])
        step = int(os.path.basename(ck).split(".")[0])
        print(f"[v10a] ckpt={ck} step={step}", flush=True)
        d = torch.load(ck, map_location="cpu", weights_only=False)
        a = vars(d["args"]) if isinstance(d.get("args"), argparse.Namespace) else d["args"]
        model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
            num_calligraphers=int(a.get("num_calligraphers", 1013)),
            num_characters=int(a.get("num_characters", 35130)),
            condition_fusion="factorized_add", callig_embed_dim=128,
            char_embed_dim=384, char_proj_mode="mlp", freeze_char_table=False,
            cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
            cond_drop_which_glyph_prob=0.5, use_checkpoint=False, learn_sigma=False,
            use_glyph_cond=True, use_char_cond=True,
            glyph_scale_init=float(a.get("glyph_scale_init", 0.4)), **arch).to(dev)
        model.load_state_dict(_strip(d.get("ema") or d.get("model") or d), strict=False)
        model.eval()
    else:  # v8e
        from src.model.controlnet import load_main_model, ControlNetDiT
        main = load_main_model(
            ckpt_path="assets/results/v8_3stage/A_main_final.pt", device=dev,
            num_calligraphers=1013, num_characters=35130,
            condition_fusion="factorized_add", callig_embed_dim=128,
            char_embed_dim=384, char_proj_mode="mlp", freeze_char_table=True,
            learn_sigma=False, **arch).eval()
        model = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True,
                              injection="modulate", null_cond="gaussian", **arch).to(dev)
        v8e = sorted(glob.glob("assets/results/v8_3stage/v8e/*/checkpoints/0022500.pt"))[-1]
        d = torch.load(v8e, map_location="cpu", weights_only=False)
        model.load_state_dict(_strip(d.get("ema") or d.get("ctrl")), strict=False)
        model.eval()
        step = 22500
        print(f"[v8e] ckpt={v8e}", flush=True)

    # ---- g 轴 CFG (--cfg-g > 0) ----
    # 历史 forward_with_cfg 两分支都给真实 g (g 是正条件), CFG 只推 callig 轴;
    # 这里在采样端外包一层: uncond 臂 g=0, eps = eps(g=0) + cfg_g*(eps(g)-eps(g=0))。
    # 依赖训练期 glyph_drop (>0) 造出的 uncond-g 分支在域内。
    if args.cfg_g > 0:
        _orig_fwd = model.forward_with_cfg

        def _fwd_gcfg(x, t, cfg_scale=0.0, **kw):
            a = _orig_fwd(x, t, cfg_scale=cfg_scale, **kw)
            g = kw.get("g")
            if g is None:
                return a
            kw0 = dict(kw)
            kw0["g"] = torch.zeros_like(g)
            b = _orig_fwd(x, t, cfg_scale=cfg_scale, **kw0)
            return b + args.cfg_g * (a - b)

        model.forward_with_cfg = _fwd_gcfg
        print(f"[cfg-g] g 轴 CFG={args.cfg_g} (callig 轴 {args.cfg} 保持不变)", flush=True)


    # ---- 条件/数据 ----
    # 对等协议: 三个模型统一 callig=1 + char=null + skel=字库骨架。
    #   * v10a/v8e 有 char 因子: y_char 必须传 num_classes (null 占位符),
    #     而非随机 id=0 —— id=0 是训练外查表污染向量, CFG 会当 positive 强化,
    #     把有 char 模型人为拖垮 (v10a 0.541 低分的测量伪影)。
    #   * v10b 无 char 因子: y_char 被 forward 忽略, 传啥都一样。
    fame_chars = {r["character"] for r in csv.DictReader(open("assets/train_fame_clean_v8.csv", encoding="utf-8"))}
    cand = []
    for p in sorted(glob.glob("src/utils/std_glyph_latent_v2/kai_gb/U+*.npy")):
        try:
            cp = int(os.path.basename(p)[2:-4], 16)
        except ValueError:
            continue
        ch = chr(cp)
        if ch not in fame_chars and 0x4E00 <= cp <= 0x9FFF:
            cand.append(ch)
    print(f"字库新字候选 {len(cand)}", flush=True)
    random.seed(7)
    picks = random.sample(cand, args.n)
    print("测试字:", " ".join(picks), flush=True)

    if getattr(model, "use_char_cond", False) or getattr(
            getattr(model, "main", None), "use_char_cond", False):
        # 有 char 因子: 显式 null-char (num_classes 占位符)
        char_embedder = getattr(model, "y_char_embedder", None) or model.main.y_char_embedder
        y_char_null = char_embedder.num_classes
        conds = [(1, y_char_null)]
        print(f"[char] 有 char 因子 → y_char={y_char_null} (null), CFG 只推 callig 轴", flush=True)
    else:
        conds = [(1, 0)]   # v10b: 值被忽略
        print("[char] 无 char 因子 (v10b), y_char 忽略", flush=True)

    vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
    diff = build_diffusion(50, "flow")
    y_callig = torch.tensor([1])   # 固定书家 (与 CPU 版一致)

    if args.tag:
        args.out = os.path.join(args.out, args.tag)
    else:
        args.out = os.path.join(args.out, f"{args.model}_step{step:07d}")
    os.makedirs(args.out, exist_ok=True)
    print(f"out={args.out}", flush=True)

    try:
        from skimage.morphology import skeletonize
        def skel_of(img_f32):
            return skeletonize(img_f32 < 0.5)
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        def skel_of(img_f32):
            mask = img_f32 < 0.5
            sk = np.zeros_like(mask); im = mask.copy()
            st = generate_binary_structure(2, 2)
            while im.any():
                er = binary_erosion(im, structure=st)
                sk |= im & ~er; im = er
            return sk
    from scipy.ndimage import binary_dilation, generate_binary_structure
    def dilate3(m):
        return binary_dilation(m, structure=generate_binary_structure(2, 2), iterations=3)

    lk = get_glyph_lookup_v2()
    rows = []
    for ch in picks:
        gv = lk.get(0, ch, random=False)   # 楷体
        if gv is None:
            print(f"  {ch}: 字库缺失, 跳过", flush=True)
            continue
        g = gv.float().unsqueeze(0).to(dev)
        with torch.no_grad():
            dec_in = vae.decode(g / 0.18215).sample[0]
        gen = ((dec_in.clamp(-1, 1) + 1) / 2)
        in_mask = gen.mean(0).cpu().numpy() < 0.5

        noise = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(7))
        t0 = time.time()
        # sample_latents: skel 参数自动路由 (use_glyph_cond → 'g'; ControlNet → 'cond')
        lat = sample_latents(model, diff, noise, conds, args.cfg, 1, dev,
                             skel=g, seed=0)
        t_s = time.time() - t0
        with torch.no_grad():
            img = vae.decode(lat.to(dev) / 0.18215).sample[0]
        out_f = ((img.clamp(-1, 1) + 1) / 2).mean(0).cpu().numpy()
        out_mask = out_f < 0.5
        out_skel = skel_of(out_f)
        iou = (out_skel & in_mask).sum() / max((out_skel | in_mask).sum(), 1)
        d3 = lambda m: dilate3(m)
        iou3 = (d3(out_skel) & d3(in_mask)).sum() / max((d3(out_skel) | d3(in_mask)).sum(), 1)
        rows.append({"char": ch, "iou": round(float(iou), 4), "iou3": round(float(iou3), 4),
                     "in_px": int(in_mask.sum()), "gen_dark": int(out_mask.sum()), "t": round(t_s, 1)})
        Image.fromarray((out_f * 255).astype("uint8")).save(os.path.join(args.out, f"{hex(ord(ch))}_gen.png"))
        Image.fromarray((gen.mean(0).cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(args.out, f"{hex(ord(ch))}_inputskel.png"))
        print(f"  {ch}: IoU={iou:.3f} IoU3={iou3:.3f} ({t_s:.1f}s)", flush=True)

    if rows:
        ious3 = [r["iou3"] for r in rows]
        ious1 = [r["iou"] for r in rows]
        summary = {"model": args.model, "step": step, "n": len(rows),
                   "cfg": args.cfg, "cfg_g": args.cfg_g,
                   "iou_mean": round(float(np.mean(ious1)), 4),
                   "iou3_mean": round(float(np.mean(ious3)), 4),
                   "iou3_median": round(float(np.median(ious3)), 4),
                   "rows": rows}
        with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"\n[summary] {args.model} step{step}: IoU3 mean={summary['iou3_mean']:.3f} "
              f"median={summary['iou3_median']:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
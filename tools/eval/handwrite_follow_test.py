# -*- coding: utf-8 -*-
"""handwrite_follow_test.py — 新字手绘遵循度测试 (CPU 小测试).

流程: 标准字库选 10 个 fame 训练集之外的新字 (楷体) → g=字库骨架 latent →
v10b 生成 → 生成图骨架化 → 与输入骨架 (VAE decode g) 算 IoU (原始/3px 容差).
输出图与输入骨架 PNG 落盘供目检.
"""
import os, sys, csv, time, argparse, json
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)
sys.stdout.reconfigure(encoding="utf-8")


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def main():
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(32)
    dev = torch.device("cpu")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="assets/results/v10b_handwrite_test")
    ap.add_argument("--model", choices=["v10b", "v8e"], default="v10b",
                    help="v10b=无char裸模型; v8e=两阶段最优 (ControlNetDiT on v8a, char=null行)")
    ap.add_argument("--ckpt", default="",
                    help="指定 ckpt 路径或 step 数字 (默认: 最新); 用于按训练步画遵循度曲线")
    ap.add_argument("--tag", default="",
                    help="输出子目录名 (默认 = step)")
    args = ap.parse_args()

    from src.model import DiT_2Cond_models
    from src.utils import get_glyph_lookup_v2
    from src.eval.inference import load_eval_vae
    from src.eval.cpu_sampler import heun_sample_cpu

    # 最新 v10b ckpt
    arch = dict(norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
                rope_theta=100.0, attn_impl="eager")  # CPU: eager (xformers sdpa 仅 cuda)
    if args.model == "v10b":
        import glob as _g
        cks = sorted(_g.glob("assets/results/v10b_skel_only_pretrain/*/checkpoints/0*.pt"),
                     key=lambda p: int(os.path.basename(p).split(".")[0]))
        if args.ckpt:
            if os.path.isfile(args.ckpt):
                ck = args.ckpt
            else:  # step 数字
                match = [c for c in cks if os.path.basename(c).startswith(str(args.ckpt))]
                ck = match[0] if match else cks[-1]
                if not match:
                    print(f"[warn] --ckpt {args.ckpt} 无匹配, 用最新", flush=True)
        else:
            ck = cks[-1]
        step = int(os.path.basename(ck).split(".")[0])
        print(f"ckpt = {ck} (step {step})", flush=True)
        if args.tag:
            args.out = os.path.join(args.out, args.tag)
        else:
            args.out = os.path.join(args.out, f"step{step:07d}")
        d = torch.load(ck, map_location="cpu", weights_only=False)
        a = vars(d["args"]) if isinstance(d.get("args"), __import__("argparse").Namespace) else d["args"]
        model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
            num_calligraphers=int(a.get("num_calligraphers", 1013)),
            num_characters=int(a.get("num_characters", 35130)),
            condition_fusion="factorized_add", callig_embed_dim=128,
            char_embed_dim=384, char_proj_mode="mlp", freeze_char_table=False,
            cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
            cond_drop_which_glyph_prob=0.5, use_checkpoint=False, learn_sigma=False,
            use_glyph_cond=True, use_char_cond=False,
            glyph_scale_init=float(a.get("glyph_scale_init", 0.4)), **arch)
        sd = _strip(d.get("ema") or d.get("model") or d)
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert len(unexp) == 0
        model.eval()
        cond_key = "g"
    else:
        # 两阶段最优 v8e: ControlNetDiT(v8a base) + ctrl ckpt; char 条件传 null 行
        from src.model.legacy.controlnet import load_main_model, ControlNetDiT
        main = load_main_model(
            ckpt_path="assets/results/v8_3stage/A_main_final.pt", device=dev,
            num_calligraphers=1013, num_characters=35130,
            condition_fusion="factorized_add", callig_embed_dim=128,
            char_embed_dim=384, char_proj_mode="mlp", freeze_char_table=True,
            learn_sigma=False, **arch)
        main.eval()
        model = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True,
                              injection="modulate", null_cond="gaussian", **arch)
        v8e = sorted(__import__("glob").glob(
            "assets/results/v8_3stage/v8e/*/checkpoints/0022500.pt"))[-1]
        step = 22500
        print(f"ckpt = {v8e} (v8e ctrl @22500, 两阶段)", flush=True)
        d = torch.load(v8e, map_location="cpu", weights_only=False)
        sd = _strip(d.get("ema") or d.get("ctrl"))
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert len(unexp) == 0
        model.eval()
        cond_key = "cond"

    # fame 训练字集合 (排除用)
    fame_chars = set()
    for r in csv.DictReader(open("assets/train_fame_clean_v8.csv", encoding="utf-8")):
        fame_chars.add(r["character"])
    print(f"fame 训练字数 {len(fame_chars)}", flush=True)

    # 字库可用新字 (楷体)
    import glob as _g
    lk = get_glyph_lookup_v2()
    cand = []
    for p in sorted(_g.glob("src/utils/std_glyph_latent_v2/kai_gb/U+*.npy")):
        try:
            cp = int(os.path.basename(p)[2:-4], 16)
        except ValueError:
            continue
        ch = chr(cp)
        if ch not in fame_chars and 0x4E00 <= cp <= 0x9FFF:   # 排除 fame 已见字, CJK 统一表意区
            cand.append(ch)
    print(f"字库新字候选 {len(cand)}", flush=True)
    import random
    random.seed(7)
    picks = random.sample(cand, args.n)
    print("测试字:", " ".join(picks), flush=True)

    vae = load_eval_vae(dev, "data/pretrained/sd-vae-ft-ema")
    from src.loss import create_diffusion_or_flow
    flow = create_diffusion_or_flow("50", diffusion_type="flow", t_sampler="logit_normal",
                                    sampler="heun", shift=1.0)
    y_callig = torch.tensor([1])   # 固定一个书家 id (风格)
    null_char_id = None
    if args.model == "v8e":
        null_char_id = int(model.main.y_char_embedder.num_classes)  # 训练过的 null 行

    try:
        from skimage.morphology import skeletonize
        def skel_of(img_f32):
            mask = img_f32 < 0.5
            return skeletonize(mask)
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

    def dilate3(m):
        from scipy.ndimage import binary_dilation, generate_binary_structure
        return binary_dilation(m, structure=generate_binary_structure(2, 2), iterations=3)

    os.makedirs(args.out, exist_ok=True)
    print(f"model={args.model} out={args.out}", flush=True)
    rows = []
    for ch in picks:
        gv = lk.get(0, ch, random=False)     # 楷体
        if gv is None:
            print(f"  {ch}: 字库缺失, 跳过", flush=True)
            continue
        g = gv.float()[None]
        with torch.no_grad():
            dec_in = vae.decode(g / 0.18215).sample.float().cpu()[0]     # 输入骨架 (模型所见)
            gen = ((dec_in.clamp(-1, 1) + 1) / 2)
        in_mask = (gen.mean(0).numpy() < 0.5)
        noise = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(7))
        t0 = time.time()
        _yh = torch.tensor([null_char_id]) if null_char_id is not None else torch.tensor([0])
        lat = heun_sample_cpu(model, noise, [(1, _yh.item())], 0.7, 1, skel=g, seed=0,
                              steps=50, shift=1.0, cond_key=cond_key)
        t_s = time.time() - t0
        with torch.no_grad():
            img = vae.decode(lat / 0.18215).sample.float().cpu()[0]
        out_f = ((img.clamp(-1, 1) + 1) / 2).mean(0).numpy()
        out_mask = out_f < 0.5
        out_skel = skel_of(out_f)
        inter = (out_skel & in_mask).sum()
        union = (out_skel | in_mask).sum()
        iou = inter / max(union, 1)
        iou3 = (dilate3(out_skel) & dilate3(in_mask)).sum() / max((dilate3(out_skel) | dilate3(in_mask)).sum(), 1)
        rows.append({"char": ch, "iou": round(float(iou), 4), "iou3": round(float(iou3), 4),
                     "in_px": int(in_mask.sum()), "gen_dark": int(out_mask.sum()), "t": round(t_s, 1)})
        Image.fromarray((out_f * 255).astype("uint8")).save(
            os.path.join(args.out, f"{hex(ord(ch))}_gen.png"))
        Image.fromarray((gen.mean(0).numpy() * 255).astype("uint8")).save(
            os.path.join(args.out, f"{hex(ord(ch))}_inputskel.png"))
        print(f"  {ch}: IoU={iou:.3f} IoU3={iou3:.3f} ({t_s:.0f}s)", flush=True)

    if rows:
        ious = [r["iou"] for r in rows]
        iou3s = [r["iou3"] for r in rows]
        print(f"\n=== 新字手绘遵循度汇总 (v10b step {step}, {len(rows)} 个训练集外新字) ===", flush=True)
        print(f"原始骨架 IoU: mean={np.mean(ious):.3f} median={np.median(ious):.3f}", flush=True)
        print(f"3px 容差 IoU: mean={np.mean(iou3s):.3f} median={np.median(iou3s):.3f}", flush=True)
        json.dump({"step": step, "rows": rows}, open(os.path.join(args.out, "summary.json"), "w"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

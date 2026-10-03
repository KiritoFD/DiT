import os, sys, glob, argparse
import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (e.g. exp/v37_skelnet_sp/.../0005000.pt)")
    ap.add_argument("--cache", default="data/top10_style23/eval_real200_cache.pt")
    ap.add_argument("--out-poster", default="", help="Output poster png path")
    ap.add_argument("--use-ema", action="store_true", help="Evaluate ema weights instead of online model")
    return ap.parse_args()

def main():
    a = parse_args()
    dev = th.device("cuda")

    from src.model.dit import DiT_2Cond_Sp_2
    model = DiT_2Cond_Sp_2(
        callig_embed_dim=128,
        glyph_vec_cond=True,
        glyph_vec_dim=128,
        condition_fusion="factorized_cat",
        cond_fusion_norm="split",
        glyph_inject_layers=4,
        glyph_embedder_depth=2,
        num_calligraphers=23,
        use_glyph_cond=True,
        use_char_cond=False,
        learn_sigma=False,
        glyph_scale_init=0.6,
        norm_type="rms",
        mlp_type="swiglu",
        qk_norm=1,
        rope=1,
        rope_theta=100.0
    ).to(dev).eval()

    d = th.load(a.ckpt, map_location="cpu", weights_only=False)
    weight_key = "ema" if a.use_ema else "model"
    if weight_key not in d:
        weight_key = "gen" if not a.use_ema else "gen_ema"
    sd = d.get(weight_key, d)
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    # Load cache
    cache = th.load(a.cache, map_location="cpu", weights_only=False)
    noise = cache["noise"].to(dev)
    conds = cache["conds"]
    std_lats = cache["std_lats"].to(dev)
    gt_pngs = cache["gt_pngs"].to(dev)
    std_pngs = cache["std_pngs"].to(dev)
    n_eval = len(conds)

    from src.eval import inference
    from src.eval.in_mem_eval import _get_vae
    from src.eval.metrics import ssim_torch, frag_ratio

    vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
    diff_eval = inference.build_diffusion(25, "flow")

    step_info = d.get("step", os.path.basename(a.ckpt).split(".")[0])
    print(f"\n=================================================================")
    print(f"【SkelNet-Sp 纯血真迹 200 样本评测 @ Checkpoint: {os.path.basename(a.ckpt)} ({weight_key})】")
    with th.no_grad():
        g_pred = inference.sample_latents(
            model, diff_eval, noise, conds,
            cfg_scale=1.0, batch=50, device=dev, skel=std_lats
        )
        dec_list = []
        for s in range(0, n_eval, 28):
            _dec = (vae.decode(g_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
            dec_list.append(_dec)
        dec = th.cat(dec_list, dim=0)

        # 1. 骨架 SSIM
        ssim_vals = ssim_torch(dec, gt_pngs).cpu().numpy()
        mean_ssim = float(np.mean(ssim_vals))
        med_ssim = float(np.median(ssim_vals))

        # 2. 墨迹 SSIM
        dec_gray = dec.mean(dim=1)
        gt_gray = gt_pngs.mean(dim=1)
        dec_mask = (dec_gray < 0.6).float().unsqueeze(1).repeat(1, 3, 1, 1)
        gt_mask = (gt_gray < 0.6).float().unsqueeze(1).repeat(1, 3, 1, 1)
        ink_ssim_vals = ssim_torch(dec_mask, gt_mask).cpu().numpy()
        mean_ink = float(np.mean(ink_ssim_vals))

        # 3. 骨架连通块破碎度
        p_np = dec_gray.cpu().numpy()
        g_np = gt_gray.cpu().numpy()
        frags = [frag_ratio(p_np[i:i+1], g_np[i:i+1]) for i in range(len(p_np))]
        mean_frag = float(np.mean(frags))
        med_frag = float(np.median(frags))

    print(f"  骨架 SSIM (均值)   : {mean_ssim:.4f} (中位: {med_ssim:.4f})")
    print(f"  墨迹 SSIM (Ink)    : {mean_ink:.4f}")
    print(f"  破碎度 (frag_ratio): {mean_frag:.3f} (中位: {med_frag:.3f}, 1.0=完全连贯无破碎)")
    print(f"=================================================================\n")

    # 4. 渲染海报
    out_poster = a.out_poster
    if not out_poster:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(a.ckpt)), "posters")
        os.makedirs(out_dir, exist_ok=True)
        out_poster = os.path.join(out_dir, f"eval_step_{step_info}_{weight_key}.png")

    p_cols = 20
    p_canvas = Image.new("RGB", (256 * p_cols, 256 * 3))
    sub_indices = np.linspace(0, n_eval - 1, p_cols, dtype=int)
    for col_idx, col in enumerate(sub_indices):
        p_std = Image.fromarray((std_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        p_gen = Image.fromarray((dec[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        p_gt = Image.fromarray((gt_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        p_canvas.paste(p_std, (col_idx * 256, 0))
        p_canvas.paste(p_gen, (col_idx * 256, 256))
        p_canvas.paste(p_gt, (col_idx * 256, 512))
    p_canvas.save(out_poster)
    print(f"海报已保存到: {out_poster}")

if __name__ == "__main__":
    main()

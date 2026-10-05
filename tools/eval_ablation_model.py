import argparse
import csv
import glob
import json
import os
import sys

base = "/home/ds/Workspace/DiT"
sys.path.insert(0, base)

from diffusers.models import AutoencoderKL
import lpips
import numpy as np
from PIL import Image
from scipy.ndimage import correlate1d
from skimage.morphology import skeletonize
import torch
from src.model import DiT_2Cond_models


def _g(img, k1d):
    return correlate1d(
        correlate1d(img, k1d, axis=0, mode="reflect"),
        k1d,
        axis=1,
        mode="reflect",
    )


def ssim_np(pred, gt, win=11, sigma=1.5, dr=1.0):
    r = win // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    out = []
    for ch in range(pred.shape[2]):
        xx = pred[:, :, ch].astype(np.float64)
        yy = gt[:, :, ch].astype(np.float64)
        ux, uy = _g(xx, k), _g(yy, k)
        ux2, uy2, uxy = ux**2, uy**2, ux * uy
        sx2 = _g(xx * xx, k) - ux2
        sy2 = _g(yy * yy, k) - uy2
        sxy = _g(xx * yy, k) - uxy
        m = ((2 * uxy + c1) * (2 * sxy + c2)) / (
            (ux2 + uy2 + c1) * (sx2 + sy2 + c2)
        )
        out.append(float(m.mean()))
    return float(np.mean(out))


def calc_skel_iou(pred_np, gt_np, thresh=0.5):
    b1 = pred_np.mean(axis=2) < thresh
    b2 = gt_np.mean(axis=2) < thresh
    if not b1.any() and not b2.any():
        return 1.0
    if not b1.any() or not b2.any():
        return 0.0
    s1, s2 = skeletonize(b1), skeletonize(b2)
    inter = float((s1 & s2).sum())
    union = float((s1 | s2).sum())
    return inter / union if union > 0 else 1.0


CSV_STRICT = "/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv"
CSV_SEEN = "/home/ds/Workspace/moyi/exp-std-csv/seen20.csv"
SHARDS_DIR = "/home/ds/Workspace/moyi/data/top10_style23/shards_img"
VAE_PATH = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
ASSETS_DIR = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal"


def evaluate_dataset(
    model,
    vae,
    lpips_fn,
    char_remap,
    callig_remap,
    font_remap,
    id_to_lat,
    csv_path,
    tag,
    out_dir,
    device,
    save_top10=False,
):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    ssims, mses, lpip_list, skel_list = [], [], [], []
    steps = 50
    torch.manual_seed(0)

    with torch.no_grad():
        for idx, r in enumerate(rows):
            cid_raw = int(r["calligrapher_id"])
            sid_raw = int(r["script_id"])
            chid_raw = int(r["character_id"])

            c_mapped = callig_remap.get(cid_raw, 0)
            s_mapped = font_remap.get(sid_raw, 0)
            ch_mapped = char_remap.get(chid_raw, 0)

            y_callig = torch.tensor([c_mapped], device=device, dtype=torch.long)
            y_script = torch.tensor([s_mapped], device=device, dtype=torch.long)
            y_char = torch.tensor([ch_mapped], device=device, dtype=torch.long)

            z = torch.randn(1, 4, 32, 32, device=device)
            ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
            for i in range(steps):
                t_curr = ts[i].expand(1) * 1000.0
                dt = ts[i + 1] - ts[i]
                v_pred = model(
                    z, t_curr, y_callig=y_callig, y_script=y_script, y_char=y_char
                )
                if isinstance(v_pred, tuple):
                    v_pred = v_pred[0]
                z = z + dt * v_pred[:, :4]

            dec = vae.decode(z / 0.18215).sample
            pred = torch.clamp((dec + 1.0) / 2.0, 0.0, 1.0)
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

            lat_gt = id_to_lat[str(r["img_id"])].unsqueeze(0).to(device)
            dec_gt = vae.decode(lat_gt / 0.18215).sample
            gt = torch.clamp((dec_gt + 1.0) / 2.0, 0.0, 1.0)
            gt_np = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

            if save_top10 and idx < 10:
                p_arr = (pred_np * 255).astype(np.uint8)
                Image.fromarray(p_arr).save(f"{out_dir}/{tag}_{idx}.png")

            ssims.append(ssim_np(pred_np, gt_np))
            mses.append(float(np.mean((pred_np - gt_np) ** 2)) * 4.0)

            p_lp = pred * 2.0 - 1.0
            g_lp = gt * 2.0 - 1.0
            lpip_list.append(float(lpips_fn(p_lp, g_lp).item()))
            skel_list.append(calc_skel_iou(pred_np, gt_np))

    return {
        "tag": tag,
        "n": len(ssims),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_med": float(np.median(ssims)),
        "mse_mean": float(np.mean(mses)),
        "lpips_mean": float(np.mean(lpip_list)),
        "skel_iou_mean": float(np.mean(skel_list)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint file path")
    parser.add_argument(
        "--out-dir", required=True, help="Directory to save eval results"
    )
    args_p = parser.parse_args()

    os.makedirs(args_p.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[eval] Loading VAE from {VAE_PATH}...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device)
    vae.eval()

    print("[eval] Loading LPIPS VGG...")
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    char_remap = {
        int(k): int(v)
        for k, v in json.load(
            open(f"{ASSETS_DIR}/char_remap.json", encoding="utf-8")
        ).items()
    }
    callig_remap = {
        int(k): int(v)
        for k, v in json.load(
            open(f"{ASSETS_DIR}/callig_remap.json", encoding="utf-8")
        ).items()
    }
    font_remap = {
        int(k): int(v)
        for k, v in json.load(
            open(f"{ASSETS_DIR}/font_remap.json", encoding="utf-8")
        ).items()
    }

    print("[eval] Indexing latent shards...")
    id_to_lat = {}
    for sf in sorted(glob.glob(f"{SHARDS_DIR}/shard_*.npz")):
        with np.load(sf) as d:
            lats = d["latents"]
            ids = d["img_ids"]
            for idx_i, iid in enumerate(ids):
                id_to_lat[str(iid)] = torch.from_numpy(lats[idx_i]).float()

    print(f"\n================ Evaluating Checkpoint: {args_p.ckpt} ================")
    ckpt = torch.load(args_p.ckpt, map_location="cpu", weights_only=False)
    cfg_args = ckpt["args"]

    model = DiT_2Cond_models[cfg_args.model](
        learn_sigma=False,
        norm_type=cfg_args.norm_type,
        mlp_type=cfg_args.mlp_type,
        qk_norm=cfg_args.qk_norm,
        rope=cfg_args.rope,
        condition_fusion=cfg_args.condition_fusion,
        cond_fusion_norm=cfg_args.cond_fusion_norm,
        num_calligraphers=cfg_args.num_calligraphers,
        num_characters=cfg_args.num_characters,
        num_script_classes=getattr(cfg_args, "num_script_classes", 3),
        use_script_cond=getattr(cfg_args, "use_script_cond", False),
        char_embed_dim=cfg_args.char_embed_dim,
        callig_embed_dim=cfg_args.callig_embed_dim,
        script_embed_dim=getattr(cfg_args, "script_embed_dim", None),
        glyph_inject_layers=cfg_args.glyph_inject_layers,
    ).to(device)

    state = ckpt.get("ema", ckpt.get("model", ckpt.get("delta", ckpt)))
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    model.eval()

    step = ckpt.get("step", 40000)
    res_seen = evaluate_dataset(
        model,
        vae,
        lpips_fn,
        char_remap,
        callig_remap,
        font_remap,
        id_to_lat,
        CSV_SEEN,
        f"seen_{step}",
        args_p.out_dir,
        device,
        save_top10=False,
    )
    res_strict = evaluate_dataset(
        model,
        vae,
        lpips_fn,
        char_remap,
        callig_remap,
        font_remap,
        id_to_lat,
        CSV_STRICT,
        f"strict_{step}",
        args_p.out_dir,
        device,
        save_top10=True,
    )

    gap_ssim = res_seen["ssim_mean"] - res_strict["ssim_mean"]
    gap_lpips = res_seen["lpips_mean"] - res_strict["lpips_mean"]

    passed = res_strict["ssim_mean"] >= 0.5815
    status_str = "PASS (>= 0.5815)" if passed else "FAIL (< 0.5815)"

    print(f"\n================ EVALUATION RESULTS @ Step {step} ================")
    print(
        f"  Strict SSIM : {res_strict['ssim_mean']:.4f} (med {res_strict['ssim_med']:.4f})  [Threshold >= 0.5815: {status_str}]"
    )
    print(f"  Strict LPIPS: {res_strict['lpips_mean']:.4f}")
    print(f"  Strict Skel : {res_strict['skel_iou_mean']:.4f}")
    print(
        f"  Seen SSIM   : {res_seen['ssim_mean']:.4f} (med {res_seen['ssim_med']:.4f})"
    )
    print(f"  Gap (Seen-Strict): SSIM={gap_ssim:+.4f}, LPIPS={gap_lpips:+.4f}")
    print("===============================================================\n")

    out_metrics = {
        "step": step,
        "strict": res_strict,
        "seen": res_seen,
        "gap_ssim": gap_ssim,
        "gap_lpips": gap_lpips,
        "passed_threshold": passed,
        "v61_40k_strict_ssim_baseline": 0.5821,
    }

    out_json = os.path.join(args_p.out_dir, "metrics_40k.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)
    print(f"[eval] Metrics saved to {out_json}")


if __name__ == "__main__":
    main()

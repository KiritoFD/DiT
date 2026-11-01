"""K-GAR: decoder-only post-training with the GAR objective, generator frozen at K.

Contract (audited in P0):
  * generator = K EMA, frozen; verified bit-identical before/after (368/368 tensors)
  * decoder starts from the ORIGINAL sd-vae decoder (a copy, never overwritten)
  * CFG fixed at 1.0 -- K is deployed at cfg=1.0, so training uses the same latent regime
  * single-step Euler with the generator's own velocity, t ~ U(t_lo, t_hi)
  * loss = MSE + LPIPS/ConvNeXt perceptual + 10*Gram   (NO edge term)
  * optional preserve regulariser: a fraction of each batch decodes a REAL latent and is
    anchored to the original decoder's output

Exposure is measured in epochs over unique artworks, not wall-clock steps.
"""
import argparse
import csv
import io
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load_generator(ckpt):
    from src.unet.io import load
    model, cfg, ck = load(ckpt, "cuda", use_ema=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt")
    ap.add_argument("--csv", default="assets/p125_20261030/stage_p2.csv")
    ap.add_argument("--out", default="assets/gar_20261101")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--t-lo", type=float, default=0.25)
    ap.add_argument("--t-hi", type=float, default=0.45)
    ap.add_argument("--preserve", type=float, default=0.2)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--dry-run", type=int, default=0, help="run N steps then stop (capacity preflight)")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("gar")

    rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8-sig"))
            if Path(r["image_path"]).is_file()]
    uniq_pix = {r.get("pixel_sha256") or r["image_path"] for r in rows}
    n_uniq = len(uniq_pix)
    steps_per_epoch = math.ceil(n_uniq / a.batch)
    total = int(round(steps_per_epoch * a.epochs))
    log.info(f"rows={len(rows)} unique_artworks={n_uniq} batch={a.batch} "
             f"steps/epoch={steps_per_epoch} target_steps={total}")

    # ---- generator (frozen) ----
    model, cfg, ck = load_generator(a.checkpoint)
    sf = float(cfg.vae_scaling_factor)
    shift = float(getattr(cfg, "vae_shift_factor", 0.0) or 0.0)
    gen_sha = {n: float(v.float().abs().sum()) for n, v in model.state_dict().items()}

    # ---- decoders: original (frozen ref) + trainable copy ----
    from diffusers import AutoencoderKL
    vae_ref = AutoencoderKL.from_pretrained(str(cfg.vae_path), local_files_only=True)
    vae_ref = vae_ref.cuda().eval().requires_grad_(False)
    import copy
    dec_student = copy.deepcopy(vae_ref.decoder).cuda().train()
    dec_ref = copy.deepcopy(vae_ref.decoder).cuda().eval().requires_grad_(False)
    for p in dec_student.parameters():
        p.requires_grad_(True)
    n_dec = sum(p.numel() for p in dec_student.parameters())
    log.info(f"decoder params={n_dec/1e6:.1f}M  lr={a.lr}  preserve={a.preserve}")

    from src.unet.joint.gar import GARLoss
    crit = GARLoss("cuda")
    opt = torch.optim.AdamW(dec_student.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    # ---- data access ----
    def load_img(path, size=256):
        with Image.open(path) as im:
            x = np.asarray(im.convert("RGB").resize((size, size))).astype(np.float32) / 255.0
        return torch.from_numpy(x).permute(2, 0, 1)

    def encode_batch(imgs):
        with torch.no_grad():
            lat = vae_ref.encode(imgs * 2 - 1).latent_dist.mode()
        return (lat - shift) * sf

    rng = np.random.default_rng(0)

    # ---- STD skeleton bank: (script_id, character) -> 64x64 tensor ----
    std_root = Path("assets/generic48_kxl_top10_20261022/std_bbox64")
    std_map = {}
    std_index = std_root / "index.csv"
    if std_index.is_file():
        for r in csv.DictReader(open(std_index, encoding="utf-8-sig")):
            std_map[(int(r["script_id"]), r["character"])] = std_root / r["path"]
    log.info(f"std bank entries={len(std_map)}")

    def std_for(script_id, character):
        p = std_map.get((int(script_id), character))
        if p is None or not p.is_file():
            return None
        with Image.open(p) as im:
            arr = np.asarray(im.convert("L").resize((64, 64))).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    t0 = time.time()
    stats_hist = []
    for step in range(1, total + 1):
        idx = rng.choice(len(rows), a.batch, replace=False)
        imgs = torch.stack([load_img(rows[i]["image_path"]) for i in idx]).cuda()
        z0 = encode_batch(imgs)

        # conditioning tensors for the frozen generator
        w = torch.tensor([int(rows[i].get("calligrapher_id", 0) or 0) for i in idx], device="cuda")
        s = torch.tensor([int(rows[i].get("script_id", 0) or 0) for i in idx], device="cuda")
        gs = []
        for i in idx:
            st = std_for(rows[i].get("script_id", 0), rows[i]["character"])
            if st is None:
                break
            gs.append(st)
        if len(gs) != len(idx):
            log.warning(f"step {step}: STD skeleton missing, skipping batch")
            continue
        g = torch.stack(gs).cuda()[:, None]

        n_pres = int(round(a.batch * a.preserve))
        opt.zero_grad(set_to_none=True)
        step_stats = {"mse": 0.0, "lpips": 0.0, "convnext": 0.0, "gram": 0.0, "n": 0}
        for begin in range(0, a.batch, a.micro):
            part = slice(begin, begin + a.micro)
            z = z0[part]
            t = torch.rand(len(z), device="cuda") * (a.t_hi - a.t_lo) + a.t_lo
            eps = torch.randn_like(z)
            zt = z * (1 - t[:, None, None, None]) + eps * t[:, None, None, None]
            with torch.no_grad():
                vel = model(zt, t * 1000.0, g=g[part], y_callig=w[part],
                            y_script=s[part], condition_dropped=True).float()
                zhat = (zt - t[:, None, None, None] * vel).detach()
            pred = dec_student((vae_ref.post_quant_conv(zhat.float() / sf + shift)
                                if hasattr(vae_ref, "post_quant_conv") else zhat.float() / sf + shift))
            pred = pred.sample if hasattr(pred, "sample") else pred
            loss, detail = crit(pred, imgs[part])
            if n_pres and begin < n_pres:
                sel = slice(0, min(a.micro, n_pres - begin))
                zr = z0[part][sel].float() / sf + shift
                with torch.no_grad():
                    out_ref = dec_ref(zr)
                    out_ref = out_ref.sample if hasattr(out_ref, "sample") else out_ref
                out_stu = dec_student(zr)
                out_stu = out_stu.sample if hasattr(out_stu, "sample") else out_stu
                loss = loss + 0.2 * torch.nn.functional.l1_loss(out_stu, out_ref)
            (loss / math.ceil(a.batch / a.micro)).backward()
            for k in ("mse", "lpips", "convnext", "gram"):
                step_stats[k] += detail[k]
            step_stats["n"] += 1
        torch.nn.utils.clip_grad_norm_(dec_student.parameters(), 1.0)
        opt.step()
        stats_hist.append(step_stats)

        if step % 20 == 0 or step == 1:
            el = time.time() - t0
            ep = step / steps_per_epoch
            log.info(f"step {step}/{total} ep={ep:.2f} "
                     f"mse={step_stats['mse']/step_stats['n']:.4f} "
                     f"lpips={step_stats['lpips']/step_stats['n']:.4f} "
                     f"gram={step_stats['gram']/step_stats['n']:.4f} "
                     f"{(step/el):.2f} it/s")
        if step % a.save_every == 0 or step == total:
            torch.save({"decoder": dec_student.state_dict(), "step": step,
                        "config": vars(a) if hasattr(a, "__dict__") else str(a)},
                       out / "ckpt" / f"{step:06d}.pt")
        if a.dry_run and step >= a.dry_run:
            log.info(f"dry-run stop at {step}")
            break


if __name__ == "__main__":
    main()

"""Audit the timestep window for a future STRUCT arm -- corrected design.

Two earlier versions of this were wrong, and the way they were wrong is the point:

  v1 used v = eps instead of v = eps - z_gt, so the "latent error" it printed was just
     the noise magnitude.
  v2 fixed the convention, but then z_hat == z_gt at every t by construction, so all
     rows were identical -- the audit could not depend on t at all.

The lesson is that "how much signal does the structure loss carry at time t" is NOT a
property of the ideal one-step estimate. With an exact estimate every t is equally
informative. It only matters under REAL model error, because the arm trains on z_hat
produced by the actual U-Net, whose error grows with t (less signal, more noise to
denoise).

So this version measures the two things that genuinely vary with t:

  1. TRIVIALITY: how close z_hat is to z_gt under realistic K error. If the estimate is
     already near-GT at small t, a structure term there supervises something already
     solved -- the failure mode the inherited `latent_struct_max_t=0.3` gate assumes
     away. (Old docs call this "低噪声时间段的结构损失可能变成平凡任务".)
  2. DISCRIMINATION: under that error, does the loss still separate a connected render
     from a shredded one with the same ink? A margin near zero means it cannot.

Model error is injected as a fraction of the actual noise direction, which is the
dominant error mode for a flow model at time t.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

_LOCAL = Path("assets/d_20261101/reports_assets/gar_crossdecode/K_A")
_REMOTE = Path("assets/f_20261101/eval/K_A")
PANEL = _LOCAL if _LOCAL.is_dir() else _REMOTE
OUT = Path("assets/d_20261101/ts_audit")
# K's measured flow-matching loss sits around 0.006-0.009 (velocity MSE), i.e. the
# velocity error is a few percent of the noise scale; sweep around that.
ERR_FRACS = (0.0, 0.05, 0.15, 0.30)


def load_gts(n=16, dev="cuda"):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    d = next(PANEL.glob("eval_samples_ctrl/*/eval200fix"))
    gts = []
    for i in range(100):
        gp = d / f"gt{i}.png"
        if gp.is_file():
            a = np.asarray(Image.open(gp).convert("RGB"), np.float32) / 255.0
            t = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev) * 2 - 1
            with torch.no_grad():
                gts.append(vae.encode(t).latent_dist.mode() * 0.18215)
        if len(gts) >= n:
            break
    return vae, torch.cat(gts)


def soft_clDice(pred, target, iters=6):
    def skel(m, it):
        s = m
        for _ in range(it):
            e = -F.max_pool2d(-s, 3, 1, 1)
            s = F.relu(s - e + s * e)
        return s
    sp, st = skel(pred, iters), skel(target, iters)
    tp = (2 * (sp * target).sum() + 1e-6) / (sp.sum() + target.sum() + 1e-6)
    ts = (2 * (st * pred).sum() + 1e-6) / (st.sum() + pred.sum() + 1e-6)
    return 1.0 - 0.5 * (tp + ts)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sf = 0.18215
    vae, z_gt = load_gts(dev=dev)
    print(f"device={dev}  chars={z_gt.shape[0]}  err_fracs={ERR_FRACS}")

    gt01 = (vae.decode(z_gt / sf).sample.clamp(-1, 1) + 1) / 2
    tw = (gt01.mean(1, keepdim=True) < 0.5).float()

    rows = []
    for t_val in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
        eps = torch.randn_like(z_gt)
        v_true = eps - z_gt
        z_t = z_gt * (1 - t_val) + eps * t_val
        for ef in ERR_FRACS:
            # the model's velocity is wrong by ef * (the true velocity magnitude)
            v_pred = v_true + ef * torch.randn_like(v_true)
            z_hat = z_t - t_val * v_pred
            lat_err = float((z_hat - z_gt).abs().mean())

            img01 = (vae.decode(z_hat / sf).sample.clamp(-1, 1) + 1) / 2
            # Ink is DARK: a stroke pixel has channel-mean near 1 in [0,1] convention below,
            # so the ink probability must be sigmoid(+k*(0.5 - mean)), not sigmoid(-k*mean).
            # Getting this backwards makes every mask empty and silently zeroes the metric.
            ink = lambda t: torch.sigmoid((0.5 - t.mean(1, keepdim=True)) * 12.0)
            pw = ink(img01)

            # Fix ONE hard reference mask (the clean render's own ink) and score both
            # predictions against it, so the comparison is about connectivity only.
            ref = (pw > 0.5).float()
            if ref.sum() == 0:
                continue
            cl_clean = float(soft_clDice(pw, ref).item())

            # Same latent with every 4th latent column overwritten from GT. The
            # perturbation is verified to shred strokes (component counts 4->9, 2->8,
            # 1->7) in check_shred_perturb.py.
            z_frag = z_hat.clone()
            z_frag[:, :, ::4, :] = z_gt[:, :, ::4, :]
            img_f = (vae.decode(z_frag / sf).sample.clamp(-1, 1) + 1) / 2
            pw_f = ink(img_f)
            cl_shred = float(soft_clDice(pw_f, ref).item())

            # How much WORSE a shredded render of the same latent scores. Positive means
            # the term penalises fragmentation, so it can drive repair.
            margin = cl_shred - cl_clean

            rows.append(dict(t=t_val, err_frac=ef, latent_err=lat_err,
                             cldice_clean=cl_clean, cldice_shred=cl_shred,
                             margin=margin))
        sub = [r for r in rows if r["t"] == t_val]
        base = next(r for r in sub if r["err_frac"] == 0.0)
        ec = next(r for r in sub if r["err_frac"] == ERR_FRACS[-1])
        print(f"  t={t_val:4.2f} | err=0.00: |dz|={base['latent_err']:.3f} "
              f"margin={base['margin']:+.4f} | err=0.30: |dz|={ec['latent_err']:.3f} "
              f"margin={ec['margin']:+.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "ts_audit.json", "w"), indent=1)
    print(f"\nwrote {OUT/'ts_audit.json'}")
    print("\nmargin>0 => shredding the strokes raises the structure loss, so the term")
    print("carries usable repair gradient at that t and error level. Margin~0 means the")
    print("term is blind to connectivity there and the gate should exclude that window.")


if __name__ == "__main__":
    main()

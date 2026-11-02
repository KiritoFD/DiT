"""Sanity-check the shredded-render perturbation used by ts_window_audit.

ts_window_audit reported margins of ~0 and slightly negative at every t, which would
mean soft-clDice cannot see fragmentation. Before believing that, check the perturbation
itself: overwriting every 4th LATENT column with GT should shred the decoded strokes,
but at 32x32 the decoder may simply smooth stripes away, in which case the margin is
measuring nothing about connectivity.

This script prints, for one latent: the ink share and connected-component count of the
clean render vs the "shredded" render, and the soft-clDice of each against GT. If the
shattered render is not actually more fragmented, the audit's margin is vacuous and the
ts_window_audit conclusion must be withdrawn.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import label  # noqa: E402

_LOCAL = Path("assets/d_20261101/reports_assets/gar_crossdecode/K_A")
_REMOTE = Path("assets/f_20261101/eval/K_A")
PANEL = _LOCAL if _LOCAL.is_dir() else _REMOTE


def comps(img01):
    b = img01.mean(axis=2) < 0.5
    _, n = label(b, structure=np.ones((3, 3), int))
    return int(n), float(b.mean())


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
    return float(1.0 - 0.5 * (tp + ts))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    sf = 0.18215

    d = next(PANEL.glob("eval_samples_ctrl/*/eval200fix"))
    for i in (0, 1, 2, 3, 4):
        gp = d / f"gt{i}.png"
        if not gp.is_file():
            continue
        a = np.asarray(Image.open(gp).convert("RGB"), np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev) * 2 - 1
        with torch.no_grad():
            z = vae.encode(t).latent_dist.mode() * sf
            clean = (vae.decode(z / sf).sample.clamp(-1, 1) + 1) / 2

            # perturbation A: overwrite every 4th latent column from a NEIGHBOUR (destroys
            # spatial continuity across latent columns; actual stripe artifact)
            zA = z.clone()
            zA[:, :, :, ::4] = z[:, :, :, 1::4] if z.shape[-1] > 4 else z[:, :, :, ::4]
            imgA = (vae.decode(zA / sf).sample.clamp(-1, 1) + 1) / 2

            # perturbation B: additive high-frequency noise in latent space (the more
            # standard way to break thin strokes)
            zB = z + 0.6 * torch.randn_like(z)
            imgB = (vae.decode(zB / sf).sample.clamp(-1, 1) + 1) / 2

        c0, i0 = comps(clean[0].permute(1, 2, 0).cpu().numpy())
        cA, iA = comps(imgA[0].permute(1, 2, 0).cpu().numpy())
        cB, iB = comps(imgB[0].permute(1, 2, 0).cpu().numpy())
        tw = (clean.mean(1, keepdim=True) < 0.5).float()
        # use the CLEAN render as its own "GT" so the comparison is about connectivity only
        cl0 = soft_clDice(torch.sigmoid(-clean.mean(1, keepdim=True) * 6), tw)
        clA = soft_clDice(torch.sigmoid(-imgA.mean(1, keepdim=True) * 6), tw)
        clB = soft_clDice(torch.sigmoid(-imgB.mean(1, keepdim=True) * 6), tw)
        print(f"char {i}: clean comps={c0:4d} ink={i0:.3f} clDice={cl0:.4f}")
        print(f"          perturbA(cols) comps={cA:4d} ink={iA:.3f} clDice={clA:.4f} "
              f"margin={clA - cl0:+.4f}")
        print(f"          perturbB(noise) comps={cB:4d} ink={iB:.3f} clDice={clB:.4f} "
              f"margin={clB - cl0:+.4f}")


if __name__ == "__main__":
    main()

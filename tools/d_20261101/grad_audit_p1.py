"""P1 gradient audit: can an image-domain loss reach the U-Net usefully?

The proposed P1 objective is L = L_FM + lambda * L_image(D(z_hat), GT), with the VAE frozen
and the U-Net unfrozen, so the image loss back-propagates THROUGH the decoder.

Three things must be true before spending hours on it, and each is cheap to measure:

  1. REACH: the image loss must actually produce a gradient at the U-Net's parameters.
     (An accidental detach, or a decoder in eval/no_grad, silently yields zero.)
  2. SCALE: its gradient norm relative to L_FM's. The review's starting scan is ~5-20%.
     Too small and it does nothing; too large and it overrides the flow-matching target
     and blurs strokes.
  3. ALIGNMENT: the angle between the image-loss gradient and the FM gradient. If they
     are near-orthogonal the term is adding a new direction (potentially useful); if they
     are strongly opposed it is fighting FM, which is the mechanism that would degrade
     diversity and stroke sharpness.

Also measured: how all three vary with t, so the window is chosen from data. The prior
`t<=0.3` gate is NOT assumed -- an earlier audit showed the structure signal is weakest
there, and at high t the noise carries almost no GT information, so forcing image L1
there encourages predicting the conditional mean (blur, lost diversity).

This runs on K with a handful of real latents; it does NOT train.

Usage:  python tools/d_20261101/grad_audit_p1.py
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
OUT = Path("assets/d_20261101/grad_audit")
K_CKPT = "assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt"


def load_gt(n=8, dev="cuda"):
    d = next(PANEL.glob("eval_samples_ctrl/*/eval200fix"))
    out = []
    for i in range(100):
        gp = d / f"gt{i}.png"
        if gp.is_file():
            a = np.asarray(Image.open(gp).convert("RGB"), np.float32) / 255.0
            out.append(torch.from_numpy(a).permute(2, 0, 1).to(dev) * 2 - 1)
        if len(out) >= n:
            break
    return torch.stack(out)


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for gi in g:
        if gi is not None:
            total += float(gi.pow(2).sum())
    return total ** 0.5


def grad_vec(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [gi.detach().clone() if gi is not None else torch.zeros_like(p)
            for gi, p in zip(g, params)]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sf = 0.18215

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ---- load K via the repo's own factory so the architecture matches exactly ----
    from tools.style_matrix.runtime.common import construct  # noqa: E402

    # A minimal spec that reproduces K: point `source` at the checkpoint and transfer the
    # style branch, which is what K itself carries. Built with initialize_source=True so
    # the resulting weights are bit-identical to K (verified by check_spec_init.py).
    spec = {
        "source": K_CKPT,
        "transfer_style": True,
        "freeze_backbone": False,
        "skip_fm": False,
        "config": {
            "seed": 0,
            "style_matrix_repr": 0,
            "style_matrix_table": "assets/u100_pretrain_20261024/tables/font_remap.json",
        },
    }
    model, cfg, tr = construct(spec, device=dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable generator tensors: {len(params)}")
    if not params:
        raise SystemExit("no trainable parameters -- the audit would measure nothing")

    gts = load_gt(dev=dev)
    print(f"gt batch: {tuple(gts.shape)}")

    rows = []
    for t_val in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85):
        torch.manual_seed(0)
        x0 = vae.encode(gts).latent_dist.mode() * sf
        eps = torch.randn_like(x0)
        x_t = x0 * (1 - t_val) + eps * t_val
        v_target = eps - x0

        # --- FM gradient ---
        model.zero_grad(set_to_none=True)
        v_pred = model(x_t, torch.full((x_t.shape[0],), t_val * 1000.0, device=dev))
        loss_fm = F.mse_loss(v_pred.float(), v_target)
        g_fm = grad_vec(loss_fm, params)
        n_fm = sum(float(gi.pow(2).sum()) for gi in g_fm) ** 0.5

        # --- image-loss gradient, THROUGH the frozen decoder ---
        model.zero_grad(set_to_none=True)
        v_pred2 = model(x_t, torch.full((x_t.shape[0],), t_val * 1000.0, device=dev))
        z_hat = x_t - t_val * v_pred2.float()
        img = vae.decode(z_hat / sf).sample
        loss_img = F.l1_loss(img, gts)
        g_img = grad_vec(loss_img, params)
        n_img = sum(float(gi.pow(2).sum()) for gi in g_img) ** 0.5

        # cosine between the two gradient directions (flattened)
        flat_fm = torch.cat([gi.reshape(-1) for gi in g_fm])
        flat_img = torch.cat([gi.reshape(-1) for gi in g_img])
        cos = float(F.cosine_similarity(flat_fm[None], flat_img[None]).item())

        ratio = n_img / max(n_fm, 1e-12)
        rows.append(dict(t=t_val, fm_grad=n_fm, img_grad=n_img, ratio=ratio, cos=cos,
                         loss_fm=float(loss_fm), loss_img=float(loss_img)))
        print(f"  t={t_val:4.2f}  |g_FM|={n_fm:9.3e}  |g_img|={n_img:9.3e}  "
              f"ratio={ratio:6.3f}  cos={cos:+.3f}  L_img={float(loss_img):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "grad_audit.json", "w"), indent=1)
    print(f"\nwrote {OUT/'grad_audit.json'}")
    print("\nread: ratio in ~[0.05,0.20] is the review's starting scan window;")
    print("cos near 0 => the image term adds a new direction; cos strongly negative =>")
    print("it fights flow matching, which is how stroke sharpness/diversity would degrade.")


if __name__ == "__main__":
    main()

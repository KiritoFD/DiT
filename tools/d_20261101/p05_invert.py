"""P0.5 — does a topology term add repair gradient that plain reconstruction lacks?

The question P1 depends on. Before spending thousands of U-Net steps on a structure loss,
optimise the LATENT with the decoder frozen, and compare two paired variants on the SAME
characters and the SAME budget:

  A: L1 + LPIPS (+ latent anchor)              -- reconstruction only
  B: A + soft-clDice + ink/width constraints   -- adds topology

If A already repairs most breaks, P1 does not need a topology loss. If B repairs
materially more at equal budget, the topology term carries information that pixel
reconstruction does not -- and only then is a STRUCT arm worth building.

Deliberately NOT interpolating toward the GT latent: that path was already shown to make
fragmentation worse (see docs/1031/P0_LATENT_ORACLE.md), and it is a different question
from whether nonlinear optimisation can restore connectivity.

Reports both error directions separately (over-fragmentation and false merging) because a
single averaged frag lets them cancel -- see `topology_split` in src/eval/metrics.py.

Usage:
  python tools/d_20261101/p05_invert.py --n 20 --steps 200
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from src.eval.metrics import topology_split  # noqa: E402

_LOCAL = Path("assets/d_20261101/reports_assets/gar_crossdecode/K_A")
_REMOTE = Path("assets/f_20261101/eval/K_A")
PANEL = _LOCAL if _LOCAL.is_dir() else _REMOTE


def load_panel():
    d = next(PANEL.glob("eval_samples_ctrl/*/eval200fix"))
    out = []
    for i in range(100):
        gp, pp = d / f"gt{i}.png", d / f"g{i}.png"
        if not (gp.is_file() and pp.is_file()):
            continue
        gt = np.asarray(Image.open(gp).convert("RGB"), np.float32) / 255.0
        pr = np.asarray(Image.open(pp).convert("RGB"), np.float32) / 255.0
        out.append((i, gt, pr))
    return out


def pick(rows, n, seed=0):
    """n worst-fragmented + n random, so the sample has repair headroom and a control."""
    scored = []
    for i, gt, pr in rows:
        st = topology_split(pr[None], gt[None])
        scored.append((st["frag"], i, gt, pr))
    scored.sort(reverse=True)
    worst = scored[:n]
    rng = np.random.default_rng(seed)
    rest = scored[n:]
    idx = rng.choice(len(rest), size=min(n, len(rest)), replace=False)
    rand = [rest[j] for j in sorted(idx)]
    return worst, rand


def soft_clDice(pred_prob, target, iters=6):
    """Centerline Dice: connectivity-aware overlap.

    clDice compares skeletons rather than areas, so a prediction that covers the stroke
    but is split into pieces scores worse than one that is connected -- which is exactly
    the property plain Dice lacks. Differentiable via soft morphological erosion.
    """
    def soft_skel(m, iters):
        s = m
        for _ in range(iters):
            e = -F.max_pool2d(-s, 3, 1, 1)          # soft erosion
            s = F.relu(s - e + s * e)               # soft opening residual ~ centerline
        return s

    sp, st = soft_skel(pred_prob, iters), soft_skel(target, iters)
    num = 2 * (sp * target).sum() + 1e-6
    den = sp.sum() + target.sum() + 1e-6
    tp = num / den
    num2 = 2 * (st * pred_prob).sum() + 1e-6
    den2 = st.sum() + pred_prob.sum() + 1e-6
    ts = num2 / den2
    return 1.0 - 0.5 * (tp + ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mu", type=float, default=0.05, help="latent anchor weight")
    ap.add_argument("--out", default="assets/d_20261101/p05")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    import lpips as _lp
    lp = _lp.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    sf = 0.18215
    print(f"device={dev} steps={a.steps} lr={a.lr} mu={a.mu}")

    rows = load_panel()
    worst, rand = pick(rows, a.n)
    groups = [("worst20", worst), ("random20", rand)]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    for gname, items in groups:
        for variant in ("A", "B"):
            recs = []
            for frag0, idx, gt, pr in items:
                gt_t = torch.from_numpy(gt).permute(2, 0, 1)[None].to(dev) * 2 - 1
                pr_t = torch.from_numpy(pr).permute(2, 0, 1)[None].to(dev) * 2 - 1
                with torch.no_grad():
                    z0 = (vae.encode(pr_t).latent_dist.mode()) * sf
                z = z0.clone().detach().requires_grad_(True)
                opt = torch.optim.Adam([z], lr=a.lr)
                for step in range(a.steps):
                    opt.zero_grad(set_to_none=True)
                    img = vae.decode(z / sf).sample
                    l1 = F.l1_loss(img, gt_t)
                    p_lp = lp(img, gt_t).mean()
                    loss = l1 + 0.5 * p_lp
                    if variant == "B":
                        # Ink is dark. `img` is in [-1,1] here, so a stroke pixel is
                        # negative: the ink probability is sigmoid(+k*(0-mean)), NOT
                        # sigmoid(-k*mean) -- the latter inverts the mask and makes the
                        # "topology" variant optimise background against the stroke
                        # target, which is not a topology loss at all.
                        pw = torch.sigmoid((0.0 - img.mean(1, keepdim=True)) * 3.0)
                        tw = (gt_t.mean(1, keepdim=True) < 0).float()
                        loss = loss + 1.0 * soft_clDice(pw, tw)
                        loss = loss + 2.0 * (pw.mean() - tw.mean()).abs()
                    loss = loss + a.mu * ((z - z0) ** 2).mean()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    img = vae.decode(z / sf).sample
                    img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
                s0 = topology_split(pr[None], gt[None])
                s1 = topology_split(img[None], gt[None])
                recs.append(dict(idx=idx, frag_before=s0["frag"], frag_after=s1["frag"],
                                 over_before=s0["over_frac"], over_after=s1["over_frac"],
                                 merged_before=s0["merged_frac"], merged_after=s1["merged_frac"],
                                 latent_shift=float((z - z0).abs().mean().item()),
                                 l1_to_gt=float(np.abs(img - gt).mean())))
                del z, z0
            agg = {k: float(np.mean([r[k] for r in recs])) for k in recs[0] if k != "idx"}
            results[f"{gname}_{variant}"] = dict(per_char=recs, mean=agg)
            print(f"  {gname:9s} {variant}: frag {agg['frag_before']:.3f} -> "
                  f"{agg['frag_after']:.3f} | over {agg['over_before']:.2f}->{agg['over_after']:.2f}"
                  f" | merged {agg['merged_before']:.2f}->{agg['merged_after']:.2f}"
                  f" | |dz|={agg['latent_shift']:.3f}")

    json.dump(results, open(out / "p05.json", "w"), indent=1)
    print(f"\nwrote {out/'p05.json'}")
    print("\nA vs B on the same characters and budget:")
    for gname, _ in groups:
        A, B = results[f"{gname}_A"]["mean"], results[f"{gname}_B"]["mean"]
        print(f"  {gname:9s} dFrag(A)={A['frag_after']-A['frag_before']:+.3f} "
              f"dFrag(B)={B['frag_after']-B['frag_before']:+.3f} "
              f"-> B extra {(B['frag_after']-B['frag_before'])-(A['frag_after']-A['frag_before']):+.3f}")
        print(f"            dOver(A)={A['over_after']-A['over_before']:+.3f} dOver(B)={B['over_after']-B['over_before']:+.3f}")
        print(f"            dMerged(A)={A['merged_after']-A['merged_before']:+.3f} dMerged(B)={B['merged_after']-B['merged_before']:+.3f}")


if __name__ == "__main__":
    main()

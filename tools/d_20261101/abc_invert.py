"""A/C/B latent-inversion comparison with a convergence criterion.

Supersedes p05_invert.py, which had two defects:
  * an inverted ink polarity (fixed) that faked the earlier result, and
  * no convergence criterion, so A and B were compared at an arbitrary equal step count
    rather than at comparable convergence.

Arms, on IDENTICAL samples and optimisation settings:
  A: L1 + LPIPS + latent anchor                     -- plain reconstruction
  C: A + ink/width correction                        -- is the gain just ink level?
  B: C + soft-clDice                                 -- does topology add anything?

Four readouts per arm, because frag alone cannot tell a repair from an erasure
(see docs/1031/P05_TOPOLOGY_LOSS.md):
  1. per-character frag delta (+ component_excess / component_deficit)
  2. ink retention    -- the erasure guard
  3. latent drift and its VAE round-trip stability -- rules out gains from leaving the
     normal latent manifold
  4. (OCR / E-G are run separately; they need the OCR stack, not just the VAE)

The convergence criterion is on the TOTAL objective (Adam, fixed lr): stop when the
running median of |loss| improvement over the last `patience` steps falls below
`tol` *relative* to the mean loss, capped at --steps. All arms share it.
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

from src.eval.metrics import ink_retention, topology_split  # noqa: E402

_LOCAL = Path("assets/d_20261101/reports_assets/gar_crossdecode/K_A")
_REMOTE = Path("assets/f_20261101/eval/K_A")
PANEL = _LOCAL if _LOCAL.is_dir() else _REMOTE
ARMS = ("A", "C", "B")


def ink_prob(img_m11, k=3.0):
    """Ink probability for an image in [-1,1]. Ink is DARK.

    Direction is asserted in tests/eval/test_ink_polarity.py; do not re-derive it here.
    """
    return torch.sigmoid((0.0 - img_m11.mean(1, keepdim=True)) * k)


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


def load_rows():
    d = next(PANEL.glob("eval_samples_ctrl/*/eval200fix"))
    out = []
    for i in range(100):
        gp, pp = d / f"gt{i}.png", d / f"g{i}.png"
        if gp.is_file() and pp.is_file():
            gt = np.asarray(Image.open(gp).convert("RGB"), np.float32) / 255.0
            pr = np.asarray(Image.open(pp).convert("RGB"), np.float32) / 255.0
            out.append((i, gt, pr))
    return out


def select(rows, n, seed=0):
    scored = sorted(((topology_split(pr[None], gt[None])["frag"], i, gt, pr)
                     for i, gt, pr in rows), reverse=True)
    worst = scored[:n]
    rest = scored[n:]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(rest), size=min(n, len(rest)), replace=False))
    return worst, [rest[j] for j in pick]


def optimize(z0, gt_t, vae, lp, arm, args, sf):
    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=args.lr)
    hist, converged_at = [], None
    tw = (gt_t.mean(1, keepdim=True) < 0).float()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        img = vae.decode(z / sf).sample
        l1 = F.l1_loss(img, gt_t)
        loss = l1 + 0.5 * lp(img, gt_t).mean()
        if arm in ("C", "B"):
            pw = ink_prob(img)
            loss = loss + args.w_ink * (pw.mean() - tw.mean()).abs()
        if arm == "B":
            pw = ink_prob(img)
            loss = loss + args.w_cldice * soft_clDice(pw, tw)
        loss = loss + args.mu * ((z - z0) ** 2).mean()
        loss.backward()
        opt.step()
        v = float(loss.detach())
        hist.append(v)
        # relative-improvement convergence test on the total objective
        if step >= args.patience:
            ref = float(np.mean(hist[-args.patience - 1:-args.patience]))
            if abs(ref) > 0 and abs(ref - v) / abs(ref) < args.tol:
                converged_at = step
                break
    return z, (converged_at if converged_at is not None else args.steps), hist[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--w-ink", type=float, default=2.0)
    ap.add_argument("--w-cldice", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", default="assets/d_20261101/p1abc")
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
    print(f"device={dev} arms={ARMS} n={a.n} max_steps={a.steps} "
          f"patience={a.patience} tol={a.tol} lr={a.lr}")

    rows = load_rows()
    worst, rand = select(rows, a.n)
    results = {}

    for gname, items in (("worst20", worst), ("random20", rand)):
        for arm in ARMS:
            recs = []
            for frag0, idx, gt, pr in items:
                gt_t = torch.from_numpy(gt).permute(2, 0, 1)[None].to(dev) * 2 - 1
                pr_t = torch.from_numpy(pr).permute(2, 0, 1)[None].to(dev) * 2 - 1
                with torch.no_grad():
                    z0 = vae.encode(pr_t).latent_dist.mode() * sf
                z, csteps, last = optimize(z0, gt_t, vae, lp, arm, a, sf)
                with torch.no_grad():
                    img = vae.decode(z / sf).sample
                    out = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
                    # VAE round-trip stability of the OPTIMISED latent: re-encode its
                    # render and decode again; a large change means the gain came from
                    # leaving the normal latent manifold rather than a real fix.
                    rt_t = torch.from_numpy(out).permute(2, 0, 1)[None].to(dev) * 2 - 1
                    ze = vae.encode(rt_t).latent_dist.mode() * sf
                    rt = (vae.decode(ze / sf).sample.clamp(-1, 1) + 1) / 2
                    rt = rt[0].permute(1, 2, 0).cpu().numpy()
                s = topology_split(out[None], gt[None])
                recs.append(dict(
                    idx=idx, steps=csteps, final_loss=last,
                    frag_before=frag0, frag_after=s["frag"],
                    excess=s["component_excess"], deficit=s["component_deficit"],
                    ink_ret=ink_retention(out[None], gt[None]),
                    latent_drift=float((z - z0).abs().mean().item()),
                    rt_change=float(np.abs(rt - out).mean())))
                del z, z0, img
            agg = {k: float(np.mean([r[k] for r in recs])) for k in recs[0] if k != "idx"}
            results[f"{gname}_{arm}"] = dict(per_char=recs, mean=agg)
            print(f"  {gname:9s} {arm}: frag {agg['frag_before']:6.3f}->{agg['frag_after']:6.3f}"
                  f" | ink_ret {agg['ink_ret']:.3f} | excess {agg['excess']:.2f}"
                  f" deficit {agg['deficit']:.2f} | |dz| {agg['latent_drift']:.3f}"
                  f" | rt {agg['rt_change']:.4f} | steps~{agg['steps']:.0f}")

    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outd / "abc.json", "w"), indent=1)
    print(f"\nwrote {outd/'abc.json'}")
    for gname, _ in (("worst20", worst), ("random20", rand)):
        A = results[f"{gname}_A"]["mean"]; C = results[f"{gname}_C"]["mean"]
        B = results[f"{gname}_B"]["mean"]
        print(f"\n{gname}: dFrag A={A['frag_after']-A['frag_before']:+.3f} "
              f"C={C['frag_after']-C['frag_before']:+.3f} B={B['frag_after']-B['frag_before']:+.3f}")
        pA = [r["frag_after"] for r in results[f"{gname}_A"]["per_char"]]
        pC = [r["frag_after"] for r in results[f"{gname}_C"]["per_char"]]
        pB = [r["frag_after"] for r in results[f"{gname}_B"]["per_char"]]
        d = np.array(pB) - np.array(pC)
        print(f"  B vs C per-char: mean {d.mean():+.3f} sd {d.std():.3f} "
              f"| B better {int((d < 0).sum())}/{len(d)}")
        d2 = np.array(pC) - np.array(pA)
        print(f"  C vs A per-char: mean {d2.mean():+.3f} sd {d2.std():.3f} "
              f"| C better {int((d2 < 0).sum())}/{len(d2)}")
        print(f"  ink retention: A {A['ink_ret']:.3f} C {C['ink_ret']:.3f} B {B['ink_ret']:.3f}")


if __name__ == "__main__":
    main()

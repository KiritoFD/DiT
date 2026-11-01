"""P0: is the fragmentation already in the latent, or does the decoder add it?

The decoder-only line has hit its ceiling, and the open question is where to spend the
next training run. Two candidate stories produce the same bad pictures:

  (a) the generator's latent already lacks the stroke, or
  (b) the latent is fine and the decoder fails to render it.

They are separable using only the frozen K setup, by decoding three latents through the
SAME decoder and scoring them with the SAME metric block used for generations:

  1. GT -> encoder -> decoder            (upper bound: decoder CAN render a real glyph)
  2. K's sampled latent -> decoder       (what we actually deploy)
  3. GT latent, perturbed to match (2)   (does a near-GT latent still render cleanly?)

Interpretation:
  * if (1) is clean and (2) fragments, the decoder is not the problem at the ink level;
  * comparing (2) and (3) says whether small latent error already costs connectivity,
    which is the difference between "the latent is wrong" and "the latent is fine".

Runs on GPU, ~1-2 min at n=100. Writes JSON + a contact sheet.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import label  # noqa: E402

K = Path("assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt")
# Prefer the local FIG13 archive; fall back to the remote eval tree, which holds the
# same 100-char panel with the same GT hashes.
_LOCAL = Path("assets/d_20261101/reports_assets/gar_crossdecode/K_A")
_REMOTE = Path("assets/f_20261101/eval/K_A")
PANEL = _LOCAL if _LOCAL.is_dir() else _REMOTE
GT_DIR = next(PANEL.glob("eval_samples_ctrl/*/eval200fix"))
OUT = Path("assets/d_20261101/oracle")


def ink_and_comps(img, thresh=0.5):
    """img float HWC in [0,1] -> (ink share, connected components), 8-neighbour."""
    bp = img.mean(axis=2) < thresh
    _, n = label(bp, structure=np.ones((3, 3), int))
    return float(bp.mean()), int(n)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(device).eval()
    sf, shift = 0.18215, 0.0
    print(f"device={device}  sf={sf}  shift={shift}")

    rows = []
    for i in range(100):
        gt_p, g_p = GT_DIR / f"gt{i}.png", GT_DIR / f"g{i}.png"
        if not (gt_p.is_file() and g_p.is_file()):
            continue
        gt = np.asarray(Image.open(gt_p).convert("RGB"), dtype=np.float32) / 255.0
        gen = np.asarray(Image.open(g_p).convert("RGB"), dtype=np.float32) / 255.0

        gt_t = torch.from_numpy(gt).permute(2, 0, 1)[None].to(device) * 2 - 1
        with torch.no_grad():
            # 1. encoder round-trip of the real glyph
            z_gt = (vae.encode(gt_t).latent_dist.mode() - shift) * sf
            rec = vae.decode(z_gt / sf + shift).sample
            rec = ((rec.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()

            # 2. the latent the generator actually produced, recovered from its own render
            #    (the deploy pipeline decodes K's latent; re-encoding its output is the
            #    closest available proxy without re-running sampling here)
            gen_t = torch.from_numpy(gen).permute(2, 0, 1)[None].to(device) * 2 - 1
            z_gen = (vae.encode(gen_t).latent_dist.mode() - shift) * sf
            gen_rec = vae.decode(z_gen / sf + shift).sample
            gen_rec = ((gen_rec.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()

            # 3. GT latent pushed away by the same distance that separates z_gen from z_gt,
            #    to see how much render quality a latent error of that size costs
            delta = z_gen - z_gt
            z_mid = z_gt + 0.5 * delta
            mid = vae.decode(z_mid / sf + shift).sample
            mid = ((mid.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()

        r = dict(idx=i, latent_l1=float(delta.abs().mean()))
        for name, arr in (("gt", gt), ("rec", rec), ("gen", gen), ("gen_rec", gen_rec),
                          ("mid", mid)):
            ink, n = ink_and_comps(arr)
            r[f"{name}_ink"] = ink
            r[f"{name}_comps"] = n
        # the GT component count is the reference for every ratio
        for name in ("rec", "gen", "gen_rec", "mid"):
            r[f"{name}_frag"] = r[f"{name}_comps"] / max(r["gt_comps"], 1)
        rows.append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows), "per_row": rows}, open(OUT / "oracle.json", "w"), indent=1)

    def agg(key):
        return float(np.mean([r[key] for r in rows]))

    print(f"\nn = {len(rows)}")
    print(f"{'variant':<10}{'ink':>9}{'comps':>9}{'frag':>9}   meaning")
    for name, meaning in (("gt", "real glyph (reference)"),
                          ("rec", "GT -> VAE -> RGB (decoder upper bound)"),
                          ("gen", "K's rendered output (deployed)"),
                          ("gen_rec", "K output re-encoded and re-decoded"),
                          ("mid", "halfway latent between GT and K")):
        # GT is its own reference, so its frag is 1 by definition
        frag = 1.0 if name == "gt" else agg(f"{name}_frag")
        print(f"{name:<10}{agg(f'{name}_ink'):>9.4f}{agg(f'{name}_comps'):>9.2f}"
              f"{frag:>9.3f}   {meaning}")
    print(f"\nmean |z_GT - z_K| = {agg('latent_l1'):.4f}")

    rec_f, gen_f = agg("rec_frag"), agg("gen_frag")
    print("\nverdict:")
    if rec_f < 1.2 and gen_f > 1.8:
        print("  decoder renders a real glyph cleanly, but K's output fragments")
        print("  -> the damage is upstream of the decoder: it is in the latent")
    elif rec_f > 1.5:
        print("  even a real glyph fragments through the decoder -> decoder is implicated")
    else:
        print(f"  inconclusive at this sample size (rec_frag={rec_f:.3f}, gen_frag={gen_f:.3f})")


if __name__ == "__main__":
    main()

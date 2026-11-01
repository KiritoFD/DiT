"""Decoder-only poster: what swapping the decoder actually looks like.

The honest problem this poster has to solve: the effect is tiny. Mean absolute pixel
difference between K and any retrained decoder is at most 0.008 on a 0..1 scale, so a
plain side-by-side reads as "identical" and proves nothing. The poster therefore shows
three things at once:

  1. the images themselves, at readable size, over several characters;
  2. amplified DIFFERENCE maps (x8) so the structure of the change is visible -- where
     the decoder edits ink, not that it edits a lot;
  3. numbers on every panel, so nobody has to trust their eyes about magnitude.

Everything comes from the archived same-latent cross-decode set, where the generator is
verified identical across arms. No new inference.
"""
import glob
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path("assets/d_20261101/reports_assets/gar_crossdecode")
OUT = Path("assets/metrics_index/poster_decoder.png")

# kept in training order so the eye can follow the recipe
ARMS = [("K_A", "K\nstock sd-vae\n(0 steps)", 0),
        ("GAR_s000100", "GAR @100", 100),
        ("GAR_s000336", "GAR @336", 336),
        ("GAR_s000672", "GAR @672", 672),
        ("K_plus_D1dec", "D1 @2500\n(different recipe)", 2500)]

# Characters chosen by decoder disagreement, so the poster shows the *most* editable
# cases rather than a lucky or unlucky one; 98 and 89 also have high K-vs-GT error,
# so the reader sees a hard character, not the easiest one.
CHARS = [98, 89, 79, 13, 56, 21]
ZOOM = 0.30          # fraction of the image the zoom inset covers
AMP = 8.0            # difference-map amplification


def panel_dir(a):
    return Path(glob.glob(str(ROOT / a / "eval_samples_ctrl" / "*" / "eval200fix"))[0])


def img(a, stem, i):
    f = panel_dir(a) / f"{stem}{i}.png"
    return None if not f.is_file() else \
        np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0


def zoom_box(a, i):
    """Centre the zoom on where K and the strongest decoder differ most."""
    d = np.abs(a - i)
    k = 64
    # integral-image argmax of the k x k window with the most change
    c = np.cumsum(np.cumsum(d, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = d.shape
    best, by, bx = -1, 0, 0
    for y in range(0, H - k, 8):
        for x in range(0, W - k, 8):
            s = c[y + k, x + k] - c[y, x + k] - c[y + k, x] + c[y, x]
            if s > best:
                best, by, bx = s, y, x
    return by, bx, k


def main():
    dirs = {a: panel_dir(a) for a, _, _ in ARMS}
    # verify the shared-generator premise before drawing anything
    import hashlib
    import json
    sigs = set()
    for a in dirs:
        p = json.loads((dirs[a] / "protocol.json").read_text(encoding="utf-8"))
        sigs.add((p["checkpoint_sha256"], p["seed"], p["sampler"], p["num_steps"], p["cfg"]))
    if len(sigs) != 1:
        raise SystemExit(f"REFUSING: arms do not share a generator/protocol: {sigs}")
    print("premise ok: one generator, one protocol across all arms")

    kd = dirs["K_A"]
    n_rows = len(CHARS)
    # columns: GT, then (image, diff) per arm -> 1 + 2*len(ARMS)
    n_cols = 1 + 2 * len(ARMS)
    fig = plt.figure(figsize=(1.95 * n_cols, 2.35 * n_rows + 1.7))

    gs = fig.add_gridspec(n_rows, n_cols, hspace=.06, wspace=.06,
                          left=.035, right=.995, top=.86, bottom=.055)

    for r, ci in enumerate(CHARS):
        gt = img("K_A", "gt", ci)
        k = img("K_A", "g", ci)
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#ddd"); s.set_linewidth(.6)

            if c == 0:
                ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
                if r == 0:
                    ax.set_title("GT", fontsize=9.5, pad=4)
                ax.set_ylabel(f"#{ci}", fontsize=9, rotation=0, ha="right", va="center")
                continue

            a_idx = (c - 1) // 2
            is_diff = (c - 1) % 2 == 1
            arm = ARMS[a_idx][0]
            if not is_diff:
                im = img(arm, "g", ci)
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
                if r == 0:
                    ax.set_title(ARMS[a_idx][1], fontsize=8.4, linespacing=1.2, pad=4)
                if arm != "K_A" and k is not None and im is not None:
                    ax.text(.5, -.03, f"|Δ|={np.abs(k - im).mean():.4f}", transform=ax.transAxes,
                            ha="center", va="top", fontsize=6.6, color="#c0392b")
            else:
                if arm == "K_A":
                    # K vs K is the zero reference: show it so the reader can calibrate
                    # what "no change at all" looks like under the same 8x amplification.
                    ax.imshow(np.zeros_like(k), cmap="inferno", vmin=0, vmax=1)
                    if r == 0:
                        ax.set_title("K − K\nzero reference\n(same ×8 scale)", fontsize=7.6,
                                     linespacing=1.2, pad=4, color="#7f8c8d")
                    if r == 0:
                        ax.text(.5, -.03, "|Δ|=0.0000", transform=ax.transAxes, ha="center",
                                va="top", fontsize=6.6, color="#7f8c8d")
                else:
                    other = img(arm, "g", ci)
                    d = np.abs(k - other) * AMP if (k is not None and other is not None) else None
                    if d is not None:
                        # zoom the inset on the busiest region so the change is legible
                        y, x, kk = zoom_box(k, other)
                        ax.imshow(d[y:y + kk, x:x + kk], cmap="inferno", vmin=0, vmax=1)
                    if r == 0:
                        ax.set_title(f"|K − {arm.replace('K_plus_','').replace('_s','@')}|\n×{AMP:.0f} zoom",
                                     fontsize=8.0, linespacing=1.2, pad=4, color="#7d3c98")

    fig.suptitle(
        "Decoder-only poster · one latent, several decoders, and what actually changes\n"
        "generator is K and verified bit-identical across every column; only the decoder differs",
        fontsize=13, y=.975)
    fig.text(.5, .925,
             "left of each pair = the rendered character · right of each pair = |K − that decoder| "
             "amplified 8× and zoomed on its busiest 64×64 window\n"
             "the whole effect is small by construction: no decoder moves more than "
             "0.008 mean absolute pixel value, so the difference maps show WHERE ink is "
             "edited, not how much",
             ha="center", fontsize=9.2, linespacing=1.5)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"wrote {OUT}")
    print(f"rows={n_rows} cols={n_cols}")


if __name__ == "__main__":
    main()

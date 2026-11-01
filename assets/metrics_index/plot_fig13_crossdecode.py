"""FIG13 left panel: the same latent decoded three ways, next to GT.

The report asks for "K original decoder / K+D1dec / K+GAR / GT" on the same latents,
zoomed on thin-stroke terminals and adjacent ink lines.

This comparison is only meaningful because the four runs genuinely share everything
except the decoder. Verified from the eval protocol.json of each run before drawing:
  checkpoint_sha256  60a21973...  (identical -> the generator really is K)
  seed 0, sampler heun, num_steps 50, cfg 1.0, gamma 0.0, vae_sha256 b29b4139...
and the 100 GT images hash identically across all four, so row i is the same character
in every column.

Note D1b_2500 carries the same generator hash as K_A -- that is the point of that arm
(it is the clean control), so K+D1dec and D1b are two decoders over one generator, and
GAR is a third. The differences in the figure are therefore decoder-only, by construction.
"""
import glob
import hashlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path("assets/d_20261101/reports_assets/gar_crossdecode")
OUT = Path("assets/metrics_index/fig13_crossdecode.png")
# The STD-soft conditioning (`eval200fix_input_g`) is byte-identical across all arms --
# it is the shared input, not a decoder output -- so only K_A keeps a copy in the archive.
COLS = [("K_A", "K\noriginal sd-vae decoder"),
        ("K_plus_D1dec", "K + D1 decoder\n(D1 @2500, actually applied)"),
        ("D1b_2500", "D1b @2500 eval\nNO-OP: identical to K"),
        ("GAR_s000672", "K + GAR decoder\n(@672)")]


def panel_dir(name):
    hits = glob.glob(str(ROOT / name / "eval_samples_ctrl" / "*" / "eval200fix"))
    if not hits:
        raise SystemExit(f"no eval200fix panel for {name}")
    return Path(hits[0])


def verify():
    """Refuse to draw unless the runs really do share generator, seed and GT."""
    sigs, gts = {}, {}
    for name, _ in COLS:
        p = panel_dir(name)
        proto = json.loads((p / "protocol.json").read_text(encoding="utf-8"))
        sigs[name] = (proto["checkpoint_sha256"], proto["seed"], proto["sampler"],
                      proto["num_steps"], proto["cfg"], proto["vae_sha256"])
        h = hashlib.md5()
        for f in sorted(p.glob("gt*.png")):
            h.update(open(f, "rb").read())
        gts[name] = h.hexdigest()
    gens = {s[0] for s in sigs.values()}
    seeds = {s[1:] for s in sigs.values()}
    if len(gens) != 1:
        raise SystemExit(f"REFUSING: generators differ across runs: {gens}")
    if len(seeds) != 1:
        raise SystemExit(f"REFUSING: sampling protocols differ: {seeds}")
    if len(set(gts.values())) != 1:
        raise SystemExit(f"REFUSING: GT sets differ: {gts}")
    print(f"verified: one generator {list(gens)[0][:16]}..., "
          f"one protocol, one GT set ({len(set(gts.values()))} hash)")
    return {n: panel_dir(n) for n, _ in COLS}


def load(p, stem, i):
    f = p / f"{stem}{i}.png"
    if not f.is_file():
        return None
    a = np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0
    return a


def crop(a, box):
    y0, y1, x0, x1 = box
    return a[y0:y1, x0:x1]


def main():
    dirs = verify()
    n_chars = min(len(list(dirs[n].glob("g*.png"))) for n, _ in COLS)
    print(f"chars per panel: {n_chars}")

    # How different is each decoder from K, per character? Rank by disagreement so the
    # figure shows the cases where the decoder actually changes something, not 100 ties.
    kd = dirs["K_A"]
    diff = {}
    for i in range(n_chars):
        a = load(kd, "g", i)
        if a is None:
            continue
        ds = []
        for n, _ in COLS:
            if n == "K_A":
                continue
            b = load(dirs[n], "g", i)
            if b is not None:
                ds.append(float(np.abs(a - b).mean()))
        if ds:
            diff[i] = float(np.mean(ds))
    ranked = sorted(diff, key=diff.get, reverse=True)
    picks = ranked[:4] + ranked[len(ranked) // 2: len(ranked) // 2 + 1] + ranked[-2:]
    print("picked rows (idx, mean|K-decoder|):",
          [(i, round(diff[i], 4)) for i in picks])

    rows = len(picks)
    cols = len(COLS) + 1
    fig, axs = plt.subplots(rows, cols, figsize=(2.05 * cols, 2.25 * rows))
    if rows == 1:
        axs = axs[None, :]

    for r, i in enumerate(picks):
        gt = load(kd, "gt", i)
        cells = [("GT", gt)] + [(lab, load(dirs[n], "g", i)) for n, lab in COLS]
        for c, (lab, img) in enumerate(cells):
            ax = axs[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if img is None:
                ax.text(.5, .5, "n/a", ha="center", va="center", fontsize=8)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(lab, fontsize=8.6, linespacing=1.25)
            if c == 0:
                ax.set_ylabel(f"idx {i}", fontsize=8)
            for s in ax.spines.values():
                s.set_color("#ccc"); s.set_linewidth(.6)

    fig.suptitle("FIG13 · One latent, decoded several ways, next to GT\n"
                 "shared generator 60a21973…, seed 0, heun/50/CFG1.0, identical GT set "
                 "(verified before drawing); rows ordered by decoder disagreement with K.\n"
                 "The D1b column is byte-identical to K -- its eval ran with "
                 "decoder_override=null, so it never applied its trained decoder",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

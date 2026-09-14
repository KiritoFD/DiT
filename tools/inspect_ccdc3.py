# -*- coding: utf-8 -*-
"""inspect ccdc gif props: size, frames, filename patterns."""
import collections
import io
import zipfile

from PIL import Image, ImageSequence
import numpy as np

Z = r"G:\GitHub\DiT\chinese-calligraphy-dataset-with-calligrapher-221030.zip"
z = zipfile.ZipFile(Z)


def dec(n):
    if "__MACOSX" in n:
        return None
    try:
        return n.encode("cp437").decode("utf-8")
    except Exception:
        return n


gifs = []
for zi in z.infolist():
    d = dec(zi.filename)
    if d and d.lower().endswith(".gif"):
        gifs.append(zi)
print(f"gifs: {len(gifs)}")
print("name samples:", [dec(zi.filename) for zi in gifs[:8]])
props = collections.Counter()
frames = collections.Counter()
shown = 0
for zi in gifs[:400]:
    with z.open(zi) as f:
        im = Image.open(io.BytesIO(f.read()))
    n_frames = getattr(im, "n_frames", 1)
    frames[n_frames] += 1
    props[(im.mode, im.size)] += 1
    if shown < 3:
        a = np.asarray(im.convert("L"), dtype=np.float32)
        ink = a < 128
        print(f"{dec(zi.filename)}: mode={im.mode} size={im.size} frames={n_frames} "
              f"mean={a.mean():.1f} ink={ink.mean():.3f}")
        shown += 1
print("sizes:", props.most_common(6))
print("frame counts:", frames.most_common(5))

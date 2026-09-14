# -*- coding: utf-8 -*-
"""inspect ccdc zip v2: decode UTF-8 names, find files."""
import collections
import io
import zipfile

import numpy as np
from PIL import Image

Z = r"G:\GitHub\DiT\chinese-calligraphy-dataset-with-calligrapher-221030.zip"
z = zipfile.ZipFile(Z)


def dec(n):
    if "__MACOSX" in n:
        return None
    try:
        return n.encode("cp437").decode("utf-8")
    except Exception:
        return n


names = [dec(n) for n in z.namelist()]
names = [n for n in names if n]
files = [n for n in names if "." in n.split("/")[-1]]
exts = collections.Counter(n.rsplit(".", 1)[-1].lower() for n in files)
print(f"entries(non-macosx): {len(names)}, files: {len(files)}, exts: {exts.most_common(5)}")
dirs = collections.Counter()
for n in files:
    parts = n.split("/")
    dirs[parts[1] if len(parts) > 2 else parts[0]] += 1
print(f"top dirs: {len(dirs)}")
for d, c in dirs.most_common(20):
    print(f"  {d}: {c}")
# image props of a few
img_files = [n for n in files if n.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"image files: {len(img_files)}")
props = collections.Counter()
shown = 0
orig_names = z.namelist()
for zi in z.infolist():
    if zi.is_dir():
        continue
    d = dec(zi.filename)
    if not d or not d.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    with z.open(zi) as f:
        img = Image.open(io.BytesIO(f.read()))
    props[(img.mode, img.size)] += 1
    if shown < 3:
        a = np.asarray(img.convert("L"), dtype=np.float32)
        ink = a < 128
        print(f"sample {d}: mode={img.mode} size={img.size} mean={a.mean():.1f} "
              f"ink={ink.mean():.3f} unique={len(np.unique(a))}")
        shown += 1
    if shown >= 3 and len(props) > 8:
        break
print("mode/size:", props.most_common(8))

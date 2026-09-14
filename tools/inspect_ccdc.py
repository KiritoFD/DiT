# -*- coding: utf-8 -*-
"""inspect chinese-calligraphy-dataset-with-calligrapher zip."""
import collections
import io
import zipfile

import numpy as np
from PIL import Image

Z = r"G:\GitHub\DiT\chinese-calligraphy-dataset-with-calligrapher-221030.zip"
z = zipfile.ZipFile(Z)
names = z.namelist()
print("entries:", len(names))
print("first 10:", names[:10])
pngs = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
print("images:", len(pngs))
props = collections.Counter()
shown = 0
for zi in z.infolist()[:3000]:
    if zi.is_dir() or not zi.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    with z.open(zi) as f:
        img = Image.open(io.BytesIO(f.read()))
    props[(img.mode, img.size)] += 1
    if shown < 3:
        a = np.asarray(img.convert("L"), dtype=np.float32)
        ink = a < 128
        print(f"sample {zi.filename}: mode={img.mode} size={img.size} mean={a.mean():.1f} "
              f"ink={ink.mean():.3f} unique={len(np.unique(a))}")
        shown += 1
print("mode/size:", props.most_common(5))
# 目录结构: 书家?
dirs = collections.Counter()
for n in pngs:
    parts = n.split("/")
    if len(parts) >= 2:
        dirs[parts[1] if len(parts) > 2 else parts[0]] += 1
print("top dirs:", len(dirs), list(dirs.most_common(15)))

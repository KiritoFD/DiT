# -*- coding: utf-8 -*-
"""inspect Calli-Tongji.zip structure + image properties."""
import collections
import io
import zipfile

import numpy as np
from PIL import Image

z = zipfile.ZipFile(r"E:\Calli-Tongji.zip")
names = []
for n in z.namelist():
    try:
        nn = n.encode("cp437").decode("gbk")
    except Exception:
        nn = n
    names.append(nn)
dirs = collections.Counter()
for n in names:
    parts = n.split("/")
    if len(parts) >= 3 and parts[2]:
        dirs[parts[1]] += 1
print("entries:", len(names))
print("script-calligrapher dirs:", len(dirs))
for d, c in sorted(dirs.items(), key=lambda x: -x[1])[:50]:
    print(f"  {d}: {c}")

# image properties of a few samples
props = collections.Counter()
sizes = collections.Counter()
shown = 0
for zi in z.infolist():
    if zi.is_dir() or not zi.filename.lower().endswith(".png"):
        continue
    with z.open(zi) as f:
        img = Image.open(io.BytesIO(f.read()))
    props[(img.mode, img.size)] += 1
    if shown < 3:
        a = np.asarray(img.convert("L"), dtype=np.float32)
        print(f"sample: mode={img.mode} size={img.size} mean={a.mean():.1f} "
              f"ink_ratio={(a < 128).mean():.3f} min={a.min()} max={a.max()}")
        shown += 1
print("mode/size distribution:")
for k, c in props.most_common(10):
    print(f"  {k}: {c}")

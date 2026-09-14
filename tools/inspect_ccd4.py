# -*- coding: utf-8 -*-
"""inspect chinese-calligraphy-dataset-221030 (1).zip"""
import collections
import io
import zipfile

import numpy as np
from PIL import Image

Z = r"E:\chinese-calligraphy-dataset-221030 (1).zip"
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
print(f"entries: {len(names)}, files: {len(files)}, exts: {exts.most_common(5)}")
dirs = collections.Counter()
for n in files:
    parts = n.split("/")
    dirs[parts[1] if len(parts) > 2 else parts[0]] += 1
print(f"top dirs: {len(dirs)}, {dirs.most_common(10)}")
# 字符标签: 目录结构 {字}/{...}?
subdirs = collections.Counter()
for n in files:
    parts = n.split("/")
    if len(parts) >= 4:
        subdirs[parts[2]] += 1
print(f"level-2 dirs (char labels?): {len(subdirs)}, samples: {list(subdirs.most_common(10))}")
# image props
img_files = [zi for zi in z.infolist()
             if not zi.is_dir() and dec(zi.filename)
             and dec(zi.filename).lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
props = collections.Counter()
shown = 0
for zi in img_files[:300]:
    with z.open(zi) as f:
        try:
            im = Image.open(io.BytesIO(f.read()))
        except Exception:
            continue
    props[(im.mode, im.size)] += 1
    if shown < 3:
        a = np.asarray(im.convert("L"), dtype=np.float32)
        print(f"{dec(zi.filename)}: mode={im.mode} size={im.size} mean={a.mean():.1f} "
              f"ink={(a<128).mean():.3f}")
        shown += 1
print("mode/size:", props.most_common(6))

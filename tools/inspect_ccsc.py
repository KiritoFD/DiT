# -*- coding: utf-8 -*-
"""inspect CCSC train zip: structure + image sizes + quality."""
import collections
import io
import zipfile

import numpy as np
from PIL import Image

BASE = r"Chinese Calligraphy Styles by Calligraphers_datasets(1)\Chinese Calligraphy Styles by Calligraphers_datasets\Chinese Calligraphy Styles by Calligraphers_data_datasets\data"
for which in ("train", "test"):
    z = zipfile.ZipFile(
        f"{BASE}\\Chinese Calligraphy Styles by Calligraphers_{which}_datasets.zip")
    names = z.namelist()
    pngs = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"[{which}] entries={len(names)} images={len(pngs)}")
    print("  first 6:", names[:6])
    props = collections.Counter()
    shown = 0
    sharp = []
    for zi in z.infolist():
        if zi.is_dir() or not zi.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        with z.open(zi) as f:
            img = Image.open(io.BytesIO(f.read()))
        props[(img.mode, img.size)] += 1
        if shown < 2:
            a = np.asarray(img.convert("L"), dtype=np.float32)
            ink = a < 128
            # sharpness proxy: ink boundary pixel ratio
            print(f"  sample {zi.filename}: mode={img.mode} size={img.size} "
                  f"mean={a.mean():.1f} ink={ink.mean():.3f} "
                  f"unique_vals={len(np.unique(a))}")
            shown += 1
        if len(sharp) < 200:
            sharp.append(zi.filename)
    print("  mode/size:", props.most_common(5))

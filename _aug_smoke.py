import os, sys, csv, re
sys.path.insert(0, "tools")
import numpy as np
from PIL import Image
from aug6 import _augment_one

# point aug6's output dir somewhere safe for the smoke test
import aug6
aug6.OUT_IMGS = "/tmp/aug_test/aug"
os.makedirs(aug6.OUT_IMGS, exist_ok=True)

# take one source image from common csv
with open("5script/train_3top30_common.csv", encoding="utf-8") as f:
    r = next(csv.DictReader(f))
iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
src = f"final_imgs_256/{iid}.png"
print("source:", src, "combo:", r["script"], r["character"], r["calligrapher"])

os.makedirs("/tmp/aug_test", exist_ok=True)
# one original copy + 6 variants
img = Image.open(src).convert("L")
img.save("/tmp/aug_test/orig.png")
for vi in range(6):
    p = _augment_one((src, 1000000 + vi, 1000 + vi))
    # copy
    Image.open(p).save(f"/tmp/aug_test/v{vi}.png", )
print("saved 6 variants + orig to /tmp/aug_test")
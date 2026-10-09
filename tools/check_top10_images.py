import os
from PIL import Image

models = [
    "02_v13", "03_v21", "04_v23", "v54", "05_v66", "06_v68", "07_v70",
    "48_moyi_12ch", "48_moyi_4ch", "48_dit_b_aug_v66route", "gt"
]

print("=== 检查每个模型目录下的 00..09 图片的尺寸和来源 ===")
for m in models:
    mdir = os.path.join("assets/aligned_eval200_top10", m)
    files = sorted([f for f in os.listdir(mdir) if f.endswith(".png")])
    print(f"[{m}] ({len(files)} files):")
    # 打印 00.png 的尺寸
    if "00.png" in files:
        im = Image.open(os.path.join(mdir, "00.png"))
        print(f"   00.png size={im.size}, mode={im.mode}")

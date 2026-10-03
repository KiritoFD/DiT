"""对比: 我们现有 std_path  vs  我新渲染的 std —— 墨点率/是否骨架/是否二值。"""
import csv, os, random
import numpy as np
from PIL import Image, ImageDraw

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
random.seed(9)

old = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
new = list(csv.DictReader(open("assets/train_hcsu_kxl_final.csv", encoding="utf-8")))


def ink(p):
    a = np.asarray(Image.open(p).convert("L"))
    return float((a < 128).mean()), len(np.unique(a)), a.shape


print("=== 我们现有 std_path ===")
o = random.sample(old, 200)
f = [ink(r["std_path"]) for r in o]
print(f"  墨点率 med={np.median([x[0] for x in f]):.4f}  灰阶 med={int(np.median([x[1] for x in f]))}")
print("=== 我们现有 image_path ===")
f = [ink(r["image_path"]) for r in o]
print(f"  墨点率 med={np.median([x[0] for x in f]):.4f}  灰阶 med={int(np.median([x[1] for x in f]))}")

print("=== 我新渲染的 std_path ===")
n = random.sample(new, 200)
f = [ink(r["std_path"]) for r in n]
print(f"  墨点率 med={np.median([x[0] for x in f]):.4f}  灰阶 med={int(np.median([x[1] for x in f]))}")
print("=== 我新归一化的 image_path ===")
f = [ink(r["image_path"]) for r in n]
print(f"  墨点率 med={np.median([x[0] for x in f]):.4f}  灰阶 med={int(np.median([x[1] for x in f]))}")

# 接触表: 上=我们的 (image, std), 下=新的 (image, std)
TH = 128
pairs = 5
sheet = Image.new("RGB", (pairs * 2 * TH, 4 * TH + 30), "white")
d = ImageDraw.Draw(sheet)
d.text((4, 3), "row1: OURS image | row2: OURS std | row3: NEW image | row4: NEW std", fill=(200, 0, 0))
for i, r in enumerate(random.sample(old, pairs)):
    sheet.paste(Image.open(r["image_path"]).convert("L").resize((TH, TH)).convert("RGB"),
                (i * 2 * TH, 20))
    sheet.paste(Image.open(r["std_path"]).convert("L").resize((TH, TH)).convert("RGB"),
                ((i * 2 + 1) * TH, 20))
for i, r in enumerate(random.sample(new, pairs)):
    sheet.paste(Image.open(r["image_path"]).convert("L").resize((TH, TH)).convert("RGB"),
                (i * 2 * TH, 20 + 2 * TH))
    sheet.paste(Image.open(r["std_path"]).convert("L").resize((TH, TH)).convert("RGB"),
                ((i * 2 + 1) * TH, 20 + 2 * TH))
sheet.save("/root/Workspace/xy/DiT/_review/std_compare.png")
print("\nsaved _review/std_compare.png")

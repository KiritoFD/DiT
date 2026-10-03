"""复核 wild 的极性判定。

我的旧判据: 边框均值 < 0.5 -> 判为反相。这只是"背景亮不亮"的代理。
更硬的判据: **笔画相对背景是深还是浅** —— 墨水是少数派,
   n_dark  = #{a < B - 0.25}     (比背景深很多的像素)
   n_light = #{a > B + 0.25}     (比背景浅很多的像素)
   n_light > n_dark  -> 反相 (白字黑底)
两者是否一致? 不一致的话我原来的 34.4% 就是错的。
"""
import os, random, collections
import numpy as np
from PIL import Image, ImageDraw

ROOT = "/root/Workspace/xy/HCSU/wild_extract"
OUT = "/root/Workspace/xy/HCSU/_polarity.png"
random.seed(11)

files = []
for d in sorted(os.listdir(ROOT)):
    dp = os.path.join(ROOT, d)
    if os.path.isdir(dp):
        for f in os.listdir(dp):
            if f.lower().endswith(".png"):
                files.append((d, f))
smp = random.sample(files, 2500)

tab = collections.Counter()
rows = []
for d, f in smp:
    try:
        a = np.asarray(Image.open(os.path.join(ROOT, d, f)).convert("L"), dtype=np.float32) / 255.0
    except Exception:
        continue
    g = a
    B = float(np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]]).mean())
    n_dark = int((g < B - 0.25).sum())
    n_light = int((g > B + 0.25).sum())
    inv_old = B < 0.5                 # 旧判据
    inv_new = n_light > n_dark        # 新判据(笔画是浅色 -> 反相)
    tab[(inv_old, inv_new)] += 1
    rows.append((d, f, B, n_dark, n_light, inv_old, inv_new, float((g < B - 0.25).mean())))

print(f"样本 {len(rows)}")
print("\n=== 旧判据(边框均值<0.5) x 新判据(笔画更浅) ===")
print(f"{'':>22}{'新:正相':>10}{'新:反相':>10}")
for o in [False, True]:
    print(f"{'旧:反相' if o else '旧:正相':>22}{tab[(o,False)]:>10}{tab[(o,True)]:>10}")
agree = tab[(False,False)] + tab[(True,True)]
print(f"\n一致率 {100*agree/len(rows):.1f}%")
print(f"旧判据判为反相: {sum(1 for r in rows if r[5])} ({100*sum(1 for r in rows if r[5])/len(rows):.1f}%)")
print(f"新判据判为反相: {sum(1 for r in rows if r[6])} ({100*sum(1 for r in rows if r[6])/len(rows):.1f}%)")

# 墨覆盖率(用各自判据归一化后)
inks = []
for d, f, B, nd, nl, o, n, _ in rows:
    a = np.asarray(Image.open(os.path.join(ROOT, d, f)).convert("L"), dtype=np.float32) / 255.0
    if n:
        a = 1.0 - a
        B = 1.0 - B
    inks.append(float((np.abs(a - B) > 0.25).mean()))
inks = np.array(inks)
print(f"\n归一化后墨覆盖: med={np.median(inks):.4f}  近空白<0.01 {100*(inks<0.01).mean():.1f}%  "
      f"过密>0.45 {100*(inks>0.45).mean():.1f}%")
print(f"可用(墨覆盖 0.03~0.45): {100*((inks>=0.03)&(inks<=0.45)).mean():.1f}%")

# 接触表: 旧判据 vs 新判据 不一致的样本 + 各自的反相
diff = [r for r in rows if r[5] != r[6]]
print(f"\n两判据不一致的样本: {len(diff)}  例: {[(r[0], r[1], round(r[2],2)) for r in diff[:5]]}")

TH = 150
pick = (diff[:6] if diff else []) + [r for r in rows if r[6]][:6]
sheet = Image.new("RGB", (12 * TH, 2 * TH + 20), "white")
ImageDraw.Draw(sheet).text((4, 4), "top: original (per NEW-criterion inverted flag) | bottom: NEW-criterion corrected", fill="red")
for i, r in enumerate(pick[:12]):
    d, f = r[0], r[1]
    im = Image.open(os.path.join(ROOT, d, f)).convert("L").resize((TH, TH), Image.LANCZOS)
    sheet.paste(im.convert("RGB"), (i * TH, 20))
    if r[6]:
        im2 = Image.fromarray(255 - np.asarray(im))
    else:
        im2 = im
    sheet.paste(im2.convert("RGB"), (i * TH, 20 + TH))
sheet.save(OUT)
print("saved", OUT)

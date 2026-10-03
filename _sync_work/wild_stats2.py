"""尺寸真实分布 + 极性归一化后的可用率。"""
import os, random, collections
import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/HCSU/wild_extract"
random.seed(2)

files = []
for d in sorted(os.listdir(ROOT)):
    dp = os.path.join(ROOT, d)
    if os.path.isdir(dp):
        for f in os.listdir(dp):
            if f.lower().endswith(".png"):
                files.append((d, f))

N = 6000
sample = random.sample(files, min(N, len(files)))
sizes = collections.Counter()
hs, ws = [], []
rows = []
for d, f in sample:
    try:
        im = Image.open(os.path.join(ROOT, d, f))
        w, h = im.size
        sizes[(w, h)] += 1
        hs.append(h); ws.append(w)
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        continue
    g = a.mean(axis=2)
    b = float(np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]]).mean())
    if b < 0.5:                      # 反极性 -> 反转
        g = 1.0 - g
        b = 1.0 - b
    ink = float((np.abs(g - b) > 0.25).mean())
    rows.append((b, ink, b >= 0.70, 0.03 <= ink <= 0.45))

print(f"抽样 {len(sizes)} 张")
print(f"\n=== 尺寸 ===")
print(f"  宽 min={min(ws)} p10={int(np.percentile(ws,10))} med={int(np.median(ws))} "
      f"p90={int(np.percentile(ws,90))} max={max(ws)}")
print(f"  高 min={min(hs)} p10={int(np.percentile(hs,10))} med={int(np.median(hs))} "
      f"p90={int(np.percentile(hs,90))} max={max(hs)}")
print(f"  正方形比例: {100*sum(1 for h,w in zip(hs,ws) if h==w)/len(hs):.1f}%")
print(f"  top8 尺寸: {sizes.most_common(8)}")

bs = np.array([r[0] for r in rows])
inks = np.array([r[1] for r in rows])
white_ok = np.array([r[2] for r in rows])
ink_ok = np.array([r[3] for r in rows])

print(f"\n=== 极性归一化后 (b<0.5 则反转) ===")
print(f"  底色 med={np.median(bs):.3f}  p10={np.percentile(bs,10):.3f} p90={np.percentile(bs,90):.3f}")
print(f"  白底 >0.70: {100*white_ok.mean():.1f}%")
print(f"  墨覆盖率 med={np.median(inks):.4f}  "
      f"近空白<0.01: {100*(inks<0.01).mean():.1f}%  "
      f"过密>0.45: {100*(inks>0.45).mean():.1f}%")
both = (white_ok & ink_ok).mean()
print(f"\n  **白底 + 墨覆盖正常**: {100*both:.1f}%  -> 全量估计 {int(both*len(files))} 张")
print(f"  (对比: 不做极性反转只有 40.9%)")

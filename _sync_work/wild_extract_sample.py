"""抽取 wild.zip 的抽样子集并做质量体检(尺寸 / 底色极性 / 墨覆盖率)。"""
import zipfile, os, sys, io, re
import numpy as np
from PIL import Image

ZIP = "/root/Workspace/xy/HCSU/wild.zip"
PWD = b"fBsJvQ1vQe0q2UhjFxztQ1q3C"
OUT = "/root/Workspace/xy/HCSU/_sample/extracted"
U = re.compile(r"#U([0-9a-fA-F]{4})")
dec = lambda s: U.sub(lambda m: chr(int(m.group(1), 16)), s)

names = [l.strip() for l in open("/root/Workspace/xy/HCSU/_sample/files.txt", encoding="utf-8")
         if l.strip() and not l.strip().endswith("/")]
print("待抽取:", len(names))

zf = zipfile.ZipFile(ZIP)
zf.setpassword(PWD)
os.makedirs(OUT, exist_ok=True)

rows = []
for n in names:
    try:
        data = zf.read(n)
        im = Image.open(io.BytesIO(data))
        a = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
        h, w = a.shape[:2]
        g = a.mean(axis=2)
        border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
        b_mean = float(border.mean())
        # 墨覆盖: 与底色的差异
        ink = float((np.abs(g - b_mean) > 0.25).mean())
        rows.append((n, h, w, b_mean, ink, len(data)))
    except Exception as e:
        rows.append((n, -1, -1, -1.0, -1.0, -1))
        print("ERR", n, e)

# 落盘以便后续目视
for n, h, w, b, ink, sz in rows:
    if sz > 0:
        d, _, fn = n.rpartition("/")
        od = os.path.join(OUT, dec(d))
        os.makedirs(od, exist_ok=True)
        open(os.path.join(od, dec(fn)), "wb").write(zf.read(n))

sizes = {}
for _, h, w, *_ in rows:
    sizes[(h, w)] = sizes.get((h, w), 0) + 1
print("\n=== 尺寸分布 ===")
for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v}")

bm = [r[3] for r in rows if r[3] >= 0]
ink = [r[4] for r in rows if r[4] >= 0]
print(f"\n=== 底色(边框均值) ===  n={len(bm)}")
print(f"  min={min(bm):.3f}  median={np.median(bm):.3f}  max={max(bm):.3f}")
print(f"  **白底(>0.7)**: {sum(1 for x in bm if x > 0.7)}   "
      f"**黑底(<0.3)**: {sum(1 for x in bm if x < 0.3)}   "
      f"中间: {sum(1 for x in bm if 0.3 <= x <= 0.7)}")
print(f"\n=== 墨覆盖率 ===")
print(f"  min={min(ink):.4f}  median={np.median(ink):.4f}  max={max(ink):.4f}")
print(f"  **近乎空白(<0.01)**: {sum(1 for x in ink if x < 0.01)}")
print(f"  **过密(>0.5)**    : {sum(1 for x in ink if x > 0.5)}")

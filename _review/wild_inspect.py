"""用 Python namelist(真中文) 抽样 + 质量体检 + 字符 Unicode 分区统计。"""
import zipfile, os, io, collections, sys
import numpy as np
from PIL import Image

ZIP = "/root/Workspace/xy/HCSU/wild.zip"
PWD = b"fBsJvQ1vQe0q2UhjFxztQ1q3C"
OUT = "/root/Workspace/xy/HCSU/_sample/imgs"

z = zipfile.ZipFile(ZIP)
z.setpassword(PWD)
names = [n for n in z.namelist() if not n.endswith("/")]
print("文件数:", len(names))

# ---------- 全量结构统计 (不读数据) ----------
per_callig, per_script, chars = collections.Counter(), collections.Counter(), collections.Counter()
sizes = collections.Counter()
for n in names:
    d, _, fn = n.rpartition("/")
    if "-" in d:
        c, _, s = d.rpartition("-")
    else:
        c, s = d, "?"
    per_callig[c] += 1
    per_script[s] += 1
    ch = os.path.splitext(fn)[0]
    chars[ch] += 1
    sizes[n] = z.getinfo(n).file_size

print(f"书家 {len(per_callig)} / 书体 {len(per_script)} / 字 {len(chars)}")
print("书体:", dict(per_script))

# ---------- 字符 Unicode 分区 ----------
def zone(cp):
    if 0x4E00 <= cp <= 0x9FFF: return "CJK基本区"
    if 0x3400 <= cp <= 0x4DBF: return "CJK扩展A"
    if 0x20000 <= cp <= 0x2A6DF: return "CJK扩展B"
    if 0x2A700 <= cp <= 0x2EBEF: return "CJK扩展C-F"
    if 0xF900 <= cp <= 0xFAFF: return "兼容汉字"
    if 0x3000 <= cp <= 0x303F: return "CJK标点"
    return f"其他 U+{cp:04X}"

zc = collections.Counter()
for ch in chars:
    zc[zone(ord(ch[0]) if ch else 0)] += 1
print("\n=== 字符 Unicode 分区 (决定能不能渲染标准字形) ===")
for k, v in zc.most_common():
    print(f"  {k}: {v}")

# ---------- 小文件比例 (全量, 只看 file_size) ----------
tiny = sum(1 for v in sizes.values() if v < 5000)
print(f"\n可疑小文件 <5KB: {tiny} ({100*tiny/len(sizes):.1f}%)")

# ---------- 抽样本 ----------
N = 120
step = max(1, len(names) // N)
sample = names[::step][:N]
os.makedirs(OUT, exist_ok=True)
rows = []
for n in sample:
    try:
        a = np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"), dtype=np.float32) / 255.0
        h, w = a.shape[:2]
        g = a.mean(axis=2)
        border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
        b = float(border.mean())
        ink = float((np.abs(g - b) > 0.25).mean())
        rows.append((n, h, w, b, ink))
    except Exception as e:
        print("ERR", n, e)

sz = collections.Counter((r[1], r[2]) for r in rows)
print("\n=== 样本尺寸分布 ===")
for k, v in sz.most_common():
    print(f"  {k}: {v}")

bm = [r[3] for r in rows]
ik = [r[4] for r in rows]
print(f"\n=== 底色(边框均值) n={len(bm)} ===")
print(f"  min={min(bm):.3f} med={np.median(bm):.3f} max={max(bm):.3f}")
print(f"  白底>0.7: {sum(1 for x in bm if x>0.7)}   黑底<0.3: {sum(1 for x in bm if x<0.3)}   "
      f"中间: {sum(1 for x in bm if 0.3<=x<=0.7)}")
print(f"\n=== 墨覆盖率 ===")
print(f"  min={min(ik):.4f} med={np.median(ik):.4f} max={max(ik):.4f}")
print(f"  近空白<0.01: {sum(1 for x in ik if x<0.01)}   过密>0.5: {sum(1 for x in ik if x>0.5)}")

# 落盘前 24 张用于目视
for n in sample[:24]:
    d, _, fn = n.rpartition("/")
    od = os.path.join(OUT, d)
    os.makedirs(od, exist_ok=True)
    open(os.path.join(od, fn), "wb").write(z.read(n))
print("\n已落盘 24 张到", OUT)

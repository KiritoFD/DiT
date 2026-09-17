"""HCSU wild 全量统计 + 与我们数据集的对比 + 可用性判定。

输出:
  1. 结构统计 (书家/书体/字)
  2. 图像体检 (尺寸/底色极性/墨覆盖率/空白率) —— 分层抽样
  3. 与我们 28,569 行数据的对比 (书家/字/书体 增量)
  4. 可用子集规模估计
"""
import os, re, csv, glob, random, collections, json
import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/HCSU/wild_extract"
OUR_CSV = "/root/Workspace/xy/DiT/assets/train_fame-kxl-tj-px60.csv"
random.seed(0)

# ---------- 1. 结构 ----------
per_callig = collections.Counter()
per_script = collections.Counter()
per_pair = collections.Counter()
chars = collections.Counter()
files = []
for d in sorted(os.listdir(ROOT)):
    dp = os.path.join(ROOT, d)
    if not os.path.isdir(dp):
        continue
    c, _, s = d.rpartition("-")
    fs = [f for f in os.listdir(dp) if f.lower().endswith(".png")]
    per_pair[d] = len(fs)
    per_callig[c] += len(fs)
    per_script[s] += len(fs)
    for f in fs:
        ch = os.path.splitext(f)[0]
        chars[ch] += 1
        files.append((d, f))

print("=" * 72)
print("1. 结构统计")
print(f"   文件总数 {len(files)}   目录(书家-书体) {len(per_pair)}")
print(f"   书家 {len(per_callig)}   书体 {len(per_script)}   不重复字 {len(chars)}")
print("\n   书体分布:")
for s, n in per_script.most_common():
    print(f"     {s}: {n:>6}  ({100*n/len(files):5.1f}%)")
print("\n   书家 top15:")
for c, n in per_callig.most_common(15):
    print(f"     {c}: {n}")
print("\n   每个 书家-书体 的样本数: "
      f"min={min(per_pair.values())} med={int(np.median(list(per_pair.values())))} "
      f"max={max(per_pair.values())}")

# ---------- 2. 图像体检 (分层抽样) ----------
print("\n" + "=" * 72)
print("2. 图像体检 (分层抽样)")
strata = collections.defaultdict(list)
for d, f in files:
    strata[d].append(f)
NS = 4000
sample = []
for d, fs in strata.items():
    k = max(1, int(round(NS * len(fs) / len(files))))
    sample += [(d, f) for f in random.sample(fs, min(k, len(fs)))]
print(f"   抽样 {len(sample)} 张")

sizes, bgs, inks, bad = collections.Counter(), [], [], []
for d, f in sample:
    p = os.path.join(ROOT, d, f)
    try:
        im = Image.open(p)
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as e:
        bad.append((p, str(e)[:40])); continue
    h, w = a.shape[:2]
    sizes[(h, w)] += 1
    g = a.mean(axis=2)
    b = float(np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]]).mean())
    bgs.append(b)
    inks.append(float((np.abs(g - b) > 0.25).mean()))

print("   尺寸分布:", dict(sizes.most_common(5)))
if bad:
    print(f"   读取失败 {len(bad)}: {bad[:3]}")
bgs, inks = np.array(bgs), np.array(inks)
print(f"\n   底色(边框均值)  med={np.median(bgs):.3f}  "
      f"p10={np.percentile(bgs,10):.3f} p90={np.percentile(bgs,90):.3f}")
print(f"     白底 >0.70 : {int((bgs>0.70).sum()):>5} ({100*(bgs>0.70).mean():5.1f}%)")
print(f"     灰底 0.3~0.7: {int(((bgs>=0.3)&(bgs<=0.7)).sum()):>5} ({100*((bgs>=0.3)&(bgs<=0.7)).mean():5.1f}%)")
print(f"     黑底 <0.30 : {int((bgs<0.30).sum()):>5} ({100*(bgs<0.30).mean():5.1f}%)")
print(f"\n   墨覆盖率  med={np.median(inks):.4f}  "
      f"p10={np.percentile(inks,10):.4f} p90={np.percentile(inks,90):.4f}")
print(f"     近空白 <0.010: {int((inks<0.01).sum()):>5} ({100*(inks<0.01).mean():5.1f}%)")
print(f"     偏稀 0.01~0.03:{int(((inks>=0.01)&(inks<0.03)).sum()):>5} ({100*((inks>=0.01)&(inks<0.03)).mean():5.1f}%)")
print(f"     正常 0.03~0.45:{int(((inks>=0.03)&(inks<=0.45)).sum()):>5} ({100*((inks>=0.03)&(inks<=0.45)).mean():5.1f}%)")
print(f"     过密 >0.450  : {int((inks>0.45).sum()):>5} ({100*(inks>0.45).mean():5.1f}%)")

# ---------- 3. 与我们对比 ----------
print("\n" + "=" * 72)
print("3. 与我们现有数据对比")
with open(OUR_CSV, encoding="utf-8") as fh:
    our = list(csv.DictReader(fh))
our_c = {r["calligrapher"] for r in our}
our_ch = {r["character"] for r in our}
our_s = {r["script"] for r in our}
print(f"   我们: {len(our)} 行, {len(our_c)} 书家, {len(our_ch)} 字, 书体={sorted(our_s)}")
print(f"   wild: {len(files)} 张, {len(per_callig)} 书家, {len(chars)} 字, 书体={sorted(per_script)}")

new_c = set(per_callig) - our_c
new_ch = set(chars) - our_ch
print(f"\n   书家交集 {len(set(per_callig)&our_c)}   新增书家 {len(new_c)}")
print(f"   字交集   {len(set(chars)&our_ch)}   新增字   {len(new_ch)}")
print(f"\n   新增书家清单 ({len(new_c)}):")
for c in sorted(new_c):
    print(f"     {c}: {per_callig[c]} 张  ({[k for k in per_pair if k.startswith(c+'-')]})")
print(f"\n   书体增量: {sorted(set(per_script)-our_s)}")
for s in sorted(set(per_script) - our_s):
    print(f"     {s}: {per_script[s]} 张")

# ---------- 4. 可用性 ----------
print("\n" + "=" * 72)
print("4. 可用性判定")
ok_size = sum(v for k, v in sizes.items() if k == (256, 256))
print(f"   256x256 占比: {100*ok_size/max(1,sum(sizes.values())):.1f}%")
usable = ((bgs > 0.70) & (inks >= 0.03) & (inks <= 0.45)).mean()
print(f"   **白底 + 墨覆盖率正常** 的比例: {100*usable:.1f}%  "
      f"-> 全量估计 {int(usable*len(files))} 张")
# 新增书家的可用量
new_c_files = sum(per_callig[c] for c in new_c)
print(f"   新增书家样本量: {new_c_files} 张 (占 {100*new_c_files/len(files):.1f}%)")

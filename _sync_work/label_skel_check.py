"""1) 每张图的 书家/glyph 标签完整性  2) std skel 是否建全 + 是不是 3px
   3) top10 书家 x 书体 的字数/张数
"""
import csv, os, collections, random
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label as cc_label
from skimage.morphology import skeletonize

os.chdir("/root/Workspace/xy/DiT")
random.seed(0)

o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))
m = o + w

print("=" * 80)
print("1. 标签完整性 (每一张图是否都有 书家 / glyph / 书体 标签)")
cols = ["calligrapher", "calligrapher_id", "character", "character_id",
        "glyph_id", "script", "script_id", "image_path", "std_path"]
for tag, rows in [("老", o), ("HCSU", w), ("合并", m)]:
    bad = collections.Counter()
    for r in rows:
        for c in cols:
            v = r.get(c)
            if v is None or str(v).strip() == "":
                bad[c] += 1
        # glyph_id 必须是合法整数
        try:
            int(r["glyph_id"])
        except Exception:
            bad["glyph_id(非整数)"] += 1
    if bad:
        print(f"   {tag:4s} {len(rows):>6} 行: **缺字段** {dict(bad)}")
    else:
        print(f"   {tag:4s} {len(rows):>6} 行: 全部 9 个字段齐全 ✓")
# 书家名 -> id 是否一对一
c2i = collections.defaultdict(set)
i2c = collections.defaultdict(set)
for r in m:
    c2i[r["calligrapher"]].add(r["calligrapher_id"])
    i2c[r["calligrapher_id"]].add(r["calligrapher"])
print(f"   书家: {len(c2i)} 个, 一名字多 id 的 {sum(1 for v in c2i.values() if len(v)>1)}"
      f", 一 id 多名的 {sum(1 for v in i2c.values() if len(v)>1)}")

print("=" * 80)
print("2. std skel: 覆盖 + 是不是 3px")
for tag, rows in [("老", o), ("HCSU", w)]:
    miss = sum(1 for r in rows if not os.path.exists(r["std_path"]))
    print(f"   {tag:4s} std_path 缺失 {miss}/{len(rows)}")

# 3px 验证: std = skeletonize(a<127) 后 dilate(1)。
# 再骨架化一次得到 1px 中线, 面积比 = 平均线宽。
print("   3px 验证 (对 std 再骨架化, 面积比 = 平均线宽):")
for tag, rows in [("老", o), ("HCSU", w)]:
    widths, ncomp, empty = [], [], 0
    for r in random.sample(rows, 200):
        a = np.asarray(Image.open(r["std_path"]).convert("L"))
        ink = a < 128
        if not ink.any():
            empty += 1
            continue
        sk = skeletonize(ink)
        if not sk.any():
            continue
        widths.append(ink.sum() / sk.sum())          # 平均线宽(像素)
        _, nc = cc_label(ink)                        # 连通块数
        ncomp.append(nc)
    widths = np.array(widths)
    print(f"     {tag:4s} 抽 {len(widths)}:  线宽 mean={widths.mean():.2f} "
          f"med={np.median(widths):.2f}  p10={np.percentile(widths,10):.2f} "
          f"p90={np.percentile(widths,90):.2f}   全空图 {empty}")
    print(f"          连通块数 med={int(np.median(ncomp))} "
          f"(= 笔画段数, 太少说明骨架糊成一团)")

print("=" * 80)
print("3. Top10 书家 x 书体")
cnt = collections.Counter(r["calligrapher"] for r in m)
top = [c for c, _ in cnt.most_common(10)]
ch = collections.defaultdict(set)     # (书家,书体) -> 字数
img = collections.Counter()            # (书家,书体) -> 张数
for r in m:
    ch[(r["calligrapher"], r["script"])].add(r["character"])
    img[(r["calligrapher"], r["script"])] += 1
scripts = ["楷", "行", "隶"]
hdr = f"{'书家':<10}{'总张':>6}{'总字':>6}  " + "".join(f"{s+'字':>7}{s+'张':>7}" for s in scripts)
print(hdr)
print("-" * len(hdr))
for c in top:
    tot_ch = len({r["character"] for r in m if r["calligrapher"] == c})
    line = f"{c:<10}{cnt[c]:>6}{tot_ch:>6}  "
    for s in scripts:
        line += f"{len(ch[(c,s)]):>7}{img[(c,s)]:>7}"
    print(line)
print("-" * len(hdr))
tot_line = f"{'合计':<10}{len(m):>6}{len({r['character'] for r in m}):>6}  "
for s in scripts:
    tot_line += (f"{len({r['character'] for r in m if r['script']==s}):>7}"
                 f"{sum(1 for r in m if r['script']==s):>7}")
print(tot_line)
print()
print("  全部 48 书家的书体分布 (前 20):")
print(f"  {'书家':<10}{'楷':>8}{'行':>8}{'隶':>8}")
for c in [x for x, _ in cnt.most_common(20)]:
    print(f"  {c:<10}{len(ch[(c,'楷')]):>8}{len(ch[(c,'行')]):>8}{len(ch[(c,'隶')]):>8}")

"""22 位新增书家逐个明细: 张数 / 不重复字数 / 书体 / 其中"我们已有字"占比
   / 且能否被对应书体字体渲染出标准骨架。"""
import os, csv, collections
from PIL import Image, ImageFont

ROOT = "/root/Workspace/xy/HCSU/wild_extract"
FDIR = "/root/Workspace/xy/DiT/tools/fonts"
OUR_CSV = "/root/Workspace/xy/DiT/assets/train_fame-kxl-tj-px60.csv"

SCRIPT_FONT = {
    "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
    "行": ["STXINGKA.TTF", "FZSTK.TTF"],
    "隶": ["SIMLI.TTF", "STLITI.TTF"],
}
KEEP = {"楷", "行", "隶"}          # 用户决定: 只留 楷/行/隶

# ---- 我们的数据 ----
with open(OUR_CSV, encoding="utf-8") as f:
    our = list(csv.DictReader(f))
our_c = {r["calligrapher"] for r in our}
our_ch = {r["character"] for r in our}
# 我们每个书家已有的字数
our_c_ch = collections.defaultdict(set)
for r in our:
    our_c_ch[r["calligrapher"]].add(r["character"])

# ---- wild 结构 ----
data = collections.defaultdict(lambda: collections.defaultdict(set))  # 书家 -> 书体 -> 字集合
for d in os.listdir(ROOT):
    dp = os.path.join(ROOT, d)
    if not os.path.isdir(dp):
        continue
    c, _, s = d.rpartition("-")
    for f in os.listdir(dp):
        if f.lower().endswith(".png"):
            data[c][s].add(os.path.splitext(f)[0])

_fc = {}
def can_render(ch, script):
    for fn in SCRIPT_FONT.get(script, []):
        fp = os.path.join(FDIR, fn)
        if not os.path.exists(fp):
            continue
        if fn not in _fc:
            try:
                _fc[fn] = ImageFont.truetype(fp, 200)
            except Exception:
                _fc[fn] = None
        font = _fc[fn]
        if font is None:
            continue
        try:
            m = font.getmask(ch, mode="L")
            bb = m.getbbox()
            if bb and (bb[2] - bb[0]) > 8 and (bb[3] - bb[1]) > 8:
                return True
        except Exception:
            pass
    return False


new_c = sorted(set(data) - our_c)
print(f"新增书家 {len(new_c)} 位（我们已有 {len(our_c)} 位）\n")
hdr = (f"{'书家':<10}{'书体':<10}{'张数':>6}{'字数':>6}{'已有字':>7}"
       f"{'新字':>6}{'可渲染':>7}{'可用张数':>8}")
print(hdr)
print("-" * 72)

tot_ok_img = 0
tot_img = 0
rows = []
for c in new_c:
    scr = sorted(data[c])
    keep_scr = [s for s in scr if s in KEEP]
    if not keep_scr:
        print(f"{c:<10}{'/'.join(scr):<10}{0:>6}{0:>6}{0:>7}{0:>6}{0:>7}{0:>8}   (书体全部不在保留集)")
        continue
    chs = set().union(*[data[c][s] for s in keep_scr])
    imgs = sum(len(data[c][s]) for s in keep_scr)
    have = chs & our_ch
    newch = chs - our_ch
    # 逐 (书体,字) 判定能否渲染
    ok_img = 0
    for s in keep_scr:
        for ch in data[c][s]:
            if can_render(ch, s):
                ok_img += 1
    tot_ok_img += ok_img
    tot_img += imgs
    rows.append((c, scr, keep_scr, imgs, len(chs), len(have), len(newch), ok_img))
    print(f"{c:<10}{'/'.join(keep_scr):<10}{imgs:>6}{len(chs):>6}{len(have):>7}"
          f"{len(newch):>6}{ok_img:>7}{ok_img:>8}")

print("-" * 70)
print(f"新增书家合计: {tot_img} 张, 其中可渲染 {tot_ok_img} "
      f"({100*tot_ok_img/max(1,tot_img):.1f}%)")

# ---- 与"已有字"的交集: 决定新字规模 ----
all_new_keep = set()
for c in new_c:
    for s in data[c]:
        if s in KEEP:
            all_new_keep |= data[c][s]
print(f"\n新增书家覆盖的不重复字: {len(all_new_keep)}")
print(f"  其中我们已有: {len(all_new_keep & our_ch)}")
print(f"  **全新字**  : {len(all_new_keep - our_ch)}")

# ---- 全部 wild(保留书体) 的新字 ----
allw = set()
for c in data:
    for s in data[c]:
        if s in KEEP:
            allw |= data[c][s]
print(f"\n全 wild(楷/行/隶) 不重复字: {len(allw)}")
print(f"  我们已有 {len(allw & our_ch)}, 全新 {len(allw - our_ch)}")

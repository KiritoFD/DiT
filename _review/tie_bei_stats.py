"""tie / bei 的结构 + 图像体检 + 字体可渲染性 + 与 wild/我们的关系。"""
import os, csv, random, collections
import numpy as np
from PIL import Image, ImageFont

HCSU = "/root/Workspace/xy/HCSU"
FDIR = "/root/Workspace/xy/DiT/tools/fonts"
OUR_CSV = "/root/Workspace/xy/DiT/assets/train_fame-kxl-tj-px60.csv"

SCRIPT_FONT = {
    "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
    "行": ["STXINGKA.TTF", "FZSTK.TTF"],
    "隶": ["SIMLI.TTF", "STLITI.TTF"],
}
KEEP = {"楷", "行", "隶"}

with open(OUR_CSV, encoding="utf-8") as f:
    our = list(csv.DictReader(f))
our_c = {r["calligrapher"] for r in our}
our_ch = {r["character"] for r in our}


def scan(root):
    d = collections.defaultdict(lambda: collections.defaultdict(set))
    for dn in os.listdir(root):
        dp = os.path.join(root, dn)
        if not os.path.isdir(dp):
            continue
        c, _, s = dn.rpartition("-")
        for fn in os.listdir(dp):
            if fn.lower().endswith(".png"):
                d[c][s].add(os.path.splitext(fn)[0])
    return d


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


random.seed(7)
for tag, root in [("tie", f"{HCSU}/tie_extract"), ("bei", f"{HCSU}/bei_extract")]:
    d = scan(root)
    per_script = collections.Counter()
    per_callig = collections.Counter()
    allch = set()
    for c in d:
        for s in d[c]:
            per_script[s] += len(d[c][s])
            per_callig[c] += len(d[c][s])
            allch |= d[c][s]
    n = sum(per_script.values())
    print("=" * 72)
    print(f"{tag}: {n} 张 / {len(per_callig)} 书家 / {len(per_script)} 书体 / {len(allch)} 字")
    print(f"  书体: {dict(per_script)}")
    print(f"  书家数 {len(per_callig)}  新增(我们没有) {len(set(per_callig)-our_c)}")
    print(f"  字: 我们已有 {len(allch & our_ch)}, 全新 {len(allch - our_ch)}")

    # 只留 楷/行/隶 的可渲染统计
    ok = tot = 0
    okc = set()
    for c in d:
        for s in d[c]:
            if s not in KEEP:
                continue
            for ch in d[c][s]:
                tot += 1
                if can_render(ch, s):
                    ok += 1
                    okc.add(c)
    print(f"  楷/行/隶 口径: {tot} 张, 可渲染 {ok} ({100*ok/max(1,tot):.1f}%)")

    # 图像体检
    files = []
    for c in d:
        for s in d[c]:
            for ch in d[c][s]:
                files.append((c, s, ch))
    smp = random.sample(files, min(1500, len(files)))
    sizes, bgs, inks, lv = collections.Counter(), [], [], []
    for c, s, ch in smp:
        p = os.path.join(root, f"{c}-{s}", f"{ch}.png")
        if not os.path.exists(p):
            continue
        try:
            im = Image.open(p)
            sizes[im.size] += 1
            a = np.asarray(im.convert("L"))
        except Exception:
            continue
        lv.append(len(np.unique(a)))
        g = a.astype(np.float32) / 255.0
        b = float(np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]]).mean())
        bgs.append(b); inks.append(float((np.abs(g - b) > 0.25).mean()))
    lv = np.array(lv); bgs = np.array(bgs); inks = np.array(inks)
    print(f"  尺寸 top3: {sizes.most_common(3)}   正方形 {100*sum(1 for k,v in sizes.items() if k[0]==k[1] and v>0)/max(1,len(lv)):.0f}%")
    print(f"  灰阶数 med={int(np.median(lv))}  二值(<=2阶) {100*(lv<=2).mean():.1f}%")
    print(f"  底色: 白>0.7 {100*(bgs>0.7).mean():.1f}%  灰 {100*((bgs>=0.3)&(bgs<=0.7)).mean():.1f}%  黑<0.3 {100*(bgs<0.3).mean():.1f}%")
    print(f"  墨覆盖 med={np.median(inks):.3f}  近空白<0.01 {100*(inks<0.01).mean():.1f}%  过密>0.45 {100*(inks>0.45).mean():.1f}%")

# 三源之间的书家/字关系
print("=" * 72)
wild = scan(f"{HCSU}/wild_extract")
tie = scan(f"{HCSU}/tie_extract")
bei = scan(f"{HCSU}/bei_extract")
for a, b, na, nb in [(wild, tie, "wild", "tie"), (wild, bei, "wild", "bei"), (tie, bei, "tie", "bei")]:
    ca, cb = set(a), set(b)
    print(f"{na} vs {nb}: 书家交集 {len(ca&cb)}  {na}独有 {len(ca-cb)}  {nb}独有 {len(cb-ca)}")

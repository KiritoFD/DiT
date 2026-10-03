"""强力清洗: 宋高宗整批丢弃 + keep<0.60 一律丢弃。

策略 (用户裁定: 干净的优先级远大于数据多):
  丢 D1 宋高宗 全部          -> 该批扫描纹理严重, 且书家本身不重要
  丢 D2 keep < 0.60          -> 开运算保留率, 有可见散点/纹理
  丢 D3 墨点 < 10px          -> 空白
保留 keep >= 0.60 的干净图。
同时渲一张"被判丢的 light 档"海报, 让人确认丢得对不对。
"""
import csv, os, collections, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_opening
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
FONT = "tools/fonts/simhei.ttf"
ST = np.ones((3, 3), bool)
DROP_CALLIG = {"宋高宗"}
KEEP_THR = 0.60


def keep_ratio(path):
    try:
        g = np.asarray(Image.open(path).convert("L")
                       .resize((256, 256), Image.LANCZOS)) < 128
    except Exception:
        return -1.0, 0
    ink = int(g.sum())
    if ink < 10:
        return 0.0, ink
    return float(binary_opening(g, structure=ST).sum() / ink), ink


if __name__ == "__main__":
    o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
    w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))
    for r in o:
        r["_src"] = "old"
    for r in w:
        r["_src"] = "hcsu"
    m = o + w
    print(f"输入 {len(m)} 张", flush=True)
    with Pool(48) as p:
        res = p.map(keep_ratio, [r["image_path"] for r in m], chunksize=200)

    keep_rows, drop_rows, drop_reason = [], [], collections.Counter()
    band_light = []
    for r, (k, ink) in zip(m, res):
        reason = None
        if r["calligrapher"] in DROP_CALLIG:
            reason = "D1_书家整批丢"
        elif k < 0:
            reason = "D3_读取失败"
        elif ink < 10:
            reason = "D3_空白"
        elif k < KEEP_THR:
            reason = "D2_散点纹理"
        if reason:
            drop_rows.append(r)
            drop_reason[reason] += 1
            if reason == "D2_散点纹理" and 0.30 <= k < 0.60:
                band_light.append((r, k))
        else:
            keep_rows.append(r)

    print()
    print("=== 丢弃统计 ===")
    for k2, v in drop_reason.most_common():
        print(f"   {k2:<16} {v:>6}  ({100*v/len(m):.2f}%)")
    print(f"   {'丢弃合计':<16} {len(drop_rows):>6}  ({100*len(drop_rows)/len(m):.2f}%)")
    print(f"   {'保留':<16} {len(keep_rows):>6}  ({100*len(keep_rows)/len(m):.2f}%)")
    print()
    print("=== 保留后规模 ===")
    print(f"   张数 {len(keep_rows)}   书家 {len({r['calligrapher'] for r in keep_rows})}"
          f"   字 {len({r['character'] for r in keep_rows})}"
          f"   (书体,字) {len({(r['script'], r['character']) for r in keep_rows})}")
    print(f"   各来源: " + str(collections.Counter(r["_src"] for r in keep_rows)))
    bc = collections.Counter(r["calligrapher"] for r in keep_rows)
    print(f"   最少张数的书家 top5: {bc.most_common()[-5:]}")

    # ---- 写过滤后 CSV ----
    cols = [c for c in m[0].keys() if c != "_src"]
    out = "assets/train_merged_clean.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in keep_rows:
            wr.writerow({c: r[c] for c in cols})
    print(f"\n-> {out}")
    with open("_sync_work/drop_images_final.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(r["image_path"] for r in drop_rows))
    print(f"-> _sync_work/drop_images_final.txt ({len(drop_rows)} 条)")

    # ---- 海报: 被判丢的 light 档 ----
    random.seed(1)
    samp = random.sample(band_light, min(12, len(band_light)))
    CELL, LBL = 256, 42
    nc = 4
    nr = (len(samp) + nc - 1) // nc
    cv = Image.new("RGB", (nc * CELL + (nc + 1) * 10,
                           nr * (CELL + LBL) + (nr + 1) * 10), (24, 24, 28))
    dr = ImageDraw.Draw(cv)
    fb = ImageFont.truetype(FONT, 15)
    for i, (r, k) in enumerate(samp):
        rr, cc = i // nc, i % nc
        x0 = 10 + cc * (CELL + 10)
        y0 = 10 + rr * (CELL + LBL + 10)
        g = np.asarray(Image.open(r["image_path"]).convert("L")
                       .resize((CELL, CELL), Image.LANCZOS)) < 128
        arr = np.where(g, 0, 255).astype(np.uint8)
        cv.paste(Image.fromarray(np.stack([arr] * 3, -1)), (x0, y0 + LBL))
        dr.text((x0 + 4, y0 + 3), f"{r['calligrapher']}/{r['character']} keep={k:.2f}",
                font=fb, fill=(255, 225, 140))
    cv.save("assets/dropped_light_band.png")
    print(f"-> assets/dropped_light_band.png ({len(samp)} 张被判丢的 light 档)")

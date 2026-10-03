"""v15 few-shot 数据的 resize（统一 256）+ img_id 同步（用 basename）。"""
import csv
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None

CALS = ("伊秉绶", "沈周", "徐渭", "宋高宗")
SIZE = 256


def rz(a):
    s, d = a
    if os.path.exists(d):
        return 1
    try:
        im = Image.open(s).convert("RGB")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.LANCZOS)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        im.save(d)
        return 1
    except Exception:
        return 0


for c in CALS:
    od = f"data/50k/fs50_imgs_{c}"
    jobs = []
    for nm in ("train", "eval"):
        for r in csv.DictReader(open(f"assets/fs50_{c}_{nm}.csv", encoding="utf-8")):
            src = r["image_path"]
            if not os.path.isabs(src):
                src = os.path.join("/root/Workspace/xy/DiT", src)
            jobs.append((src, os.path.join("/root/Workspace/xy/DiT", od,
                                           os.path.basename(src))))
    with ThreadPoolExecutor(16) as ex:
        ok = sum(ex.map(rz, jobs))
    for nm in ("train", "eval"):
        p = f"assets/fs50_{c}_{nm}.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        cols = list(rows[0].keys())
        for r in rows:
            b = os.path.basename(r["image_path"])
            r["image_path"] = os.path.join(od, b)
            r["src_image_path"] = os.path.join(od, b)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print(f"  {c}: {ok}/{len(jobs)} resize -> {od}")

    # img_id 用 basename 匹配（resize 改过路径，完整路径对不上）
    allr = list(csv.DictReader(open(f"assets/fs50_{c}_all.csv", encoding="utf-8")))
    idmap = {os.path.basename(r["image_path"]): r["img_id"] for r in allr}
    for nm in ("train", "eval"):
        p = f"assets/fs50_{c}_{nm}.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        cols = list(rows[0].keys())
        if "img_id" not in cols:
            cols.append("img_id")
        n = 0
        for r in rows:
            r["img_id"] = idmap.get(os.path.basename(r["image_path"]), "")
            if r["img_id"]:
                n += 1
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"    {c}/{nm}: {n}/{len(rows)} 有 img_id")

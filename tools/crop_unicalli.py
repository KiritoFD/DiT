# -*- coding: utf-8 -*-
"""
crop_unicalli.py — UniCalli 列级 bbox 裁切为单字 256x256 (fame-tj-uc base 素材).

输入: data/unicalli/data.csv + data/unicalli/images/
输出: data/imgs/unicalli_chars/{img_id}.png (img_id 从 960000+, 确定性顺序)
      assets/unicalli_rows.csv (中间表: image_path/calligrapher/script/character;
                                calligrapher_id 由合并脚本统一分配)

处理: 只留 楷/行/隶; 书家名归一 (宋徽宗/文徵明); 排除非个人书家(佚名/刻石类);
      非汉字剔除; bbox 白底 pad 成正方形 -> resize 256; 幂等(已存在跳过).
"""
import ast
import csv
import multiprocessing as mp
import os
import re
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir("/root/Workspace/xy/DiT")

OUT_DIR = "data/imgs/unicalli_chars"
OUT_CSV = "assets/unicalli_rows.csv"
IMG_ROOT = "data/unicalli/images"
IMG_ID_BASE = 960000
NAME_MAP = {"赵佶\\宋徽宗": "宋徽宗", "文征明": "文徵明"}
EXCLUDE_AUTHORS = {"佚名", "摩崖刻石", "墓志", "造像记", "金文刻石", "碑刻", "墨迹"}
KEEP = {"楷": "0", "行": "3", "隶": "4"}
HAN_RE = re.compile(r"^[\u4e00-\u9fff]$")
fields = ["image_path", "calligrapher", "script", "character", "calligrapher_id",
          "script_id", "character_id", "glyph_id", "aug"]


def load_works():
    rows = list(csv.DictReader(open("data/unicalli/data.csv", encoding="utf-8")))
    out = []
    stats = {"kept_works": 0, "excl_auth": 0, "excl_style": 0, "nonhan": 0}
    for r in rows:
        boxes = ast.literal_eval(r["location"])
        author = NAME_MAP.get(r["author"], r["author"].strip())
        chiro = r["chirography"].strip()
        if author in EXCLUDE_AUTHORS:
            stats["excl_auth"] += len(boxes)
            continue
        if chiro not in KEEP:
            stats["excl_style"] += len(boxes)
            continue
        chars = []
        for b in boxes:
            ch = b["c"]
            if not HAN_RE.match(ch):
                stats["nonhan"] += 1
                continue
            chars.append((ch, b["p"]))
        if chars:
            out.append({"img_path": r["img_path"], "author": author,
                        "chiro": chiro, "chars": chars})
            stats["kept_works"] += 1
    return out, stats


def work(task):
    """task = (out_path, img_path, (ch, (x1,y1,x2,y2))) -> (img_id, err)."""
    out_path, img_path, (ch, p) = task
    iid = int(os.path.basename(out_path)[:-4])
    if os.path.exists(out_path):
        return iid, None
    try:
        im = Image.open(img_path).convert("L")
        x1, y1, x2, y2 = p
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 8:
            return iid, "tiny"
        side = int(max(w, h) * 1.15)
        canvas = Image.new("L", (side, side), 255)
        canvas.paste(im.crop((x1, y1, x2, y2)), ((side - w) // 2, (side - h) // 2))
        canvas.resize((256, 256), Image.LANCZOS).save(out_path)
        return iid, None
    except Exception as e:
        return int(os.path.basename(out_path)[:-4]), f"{img_path}: {e}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    works, stats = load_works()
    print(f"kept works: {stats}", flush=True)
    tasks = []
    metas = []
    k = 0
    for w in works:
        for ch, p in w["chars"]:
            tasks.append((os.path.join(OUT_DIR, f"{IMG_ID_BASE + k}.png"),
                          os.path.join(IMG_ROOT, w["img_path"]), (ch, p)))
            metas.append({"character": ch, "calligrapher": w["author"],
                          "chiro": w["chiro"], "img_id": IMG_ID_BASE + k})
            k += 1
    print(f"total char crops: {len(tasks)}", flush=True)
    t0 = time.time()
    n_fail = 0
    with mp.Pool(32) as pool:
        for c, (iid, err) in enumerate(pool.imap_unordered(work, tasks, chunksize=64), 1):
            if err:
                n_fail += 1
                print(f"FAIL: {err}", flush=True)
            if c % 2000 == 0:
                print(f"  {c}/{len(tasks)} ({c/(time.time()-t0):.0f}/s)", flush=True)
    print(f"cropped {len(tasks)-n_fail} ok, {n_fail} fail, {time.time()-t0:.0f}s", flush=True)

    n_written = 0
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in metas:
            img_path = f"{OUT_DIR}/{m['img_id']}.png"
            if not os.path.exists(img_path):
                continue
            w.writerow({
                "image_path": img_path, "calligrapher": m["calligrapher"],
                "script": m["chiro"], "character": m["character"],
                "calligrapher_id": "", "script_id": KEEP[m["chiro"]],
                "character_id": "", "glyph_id": "", "aug": ""})
            n_written += 1
    print(f"written {OUT_CSV}: {n_written} rows", flush=True)


if __name__ == "__main__":
    main()

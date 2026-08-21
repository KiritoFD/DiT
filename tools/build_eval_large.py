#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建本地大测试集 eval500_top6.csv + final_images/ 本地镜像。

- 源池: 5script/train.csv 过滤 top6 书家(楷6+隶6, 11人) 的 script_id∈{0,4}
- 排除 eval100_top6.csv 已用 id, 分层抽样 楷250 + 隶250 (seed=0)
- image_path 保持 final_images/<id>.png 格式(与远程一致), 本地从
  dataset/images/<a>/<b>/<c>/<id>.png 索引后拷贝到 final_images/
"""
import csv
import os
import random

random.seed(0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAILI_TOP6 = ["颜真卿", "赵孟𫖯", "褚遂良", "智永", "柳公权", "欧阳询"]
LI_TOP6 = ["赵孟𫖯", "王澍", "吴叡", "金农", "陈鸿寿", "邓石如"]
ALLOWED = {(0, n) for n in KAILI_TOP6} | {(4, n) for n in LI_TOP6}
N_PER_SCRIPT = 250


def main():
    # 1) 读源池与已用 id
    with open(os.path.join(ROOT, "5script", "train.csv"), encoding="utf-8") as f:
        pool = [r for r in csv.DictReader(f)
                if (int(r["script_id"]), r["calligrapher"]) in ALLOWED]
    with open(os.path.join(ROOT, "docs", "s6_report", "csv", "eval100_top6.csv"),
              encoding="utf-8") as f:
        used_ids = {os.path.basename(r["image_path"])[:-4]
                    for r in csv.DictReader(f)}
    print(f"top6 池: {len(pool)} 行, eval100 已用 {len(used_ids)} id")

    # 2) 排除已用, 分层抽样
    cand = {0: [], 4: []}
    for r in pool:
        if os.path.basename(r["image_path"])[:-4] in used_ids:
            continue
        cand[int(r["script_id"])].append(r)
    picked = []
    for sid in (0, 4):
        random.shuffle(cand[sid])
        take = cand[sid][:N_PER_SCRIPT]
        print(f"script {sid}: 候选 {len(cand[sid])}, 取 {len(take)}")
        picked.extend(take)

    # 3) 索引本地数据集并拷贝到 final_images/
    need = {os.path.basename(r["image_path"])[:-4] for r in picked}
    found = {}
    img_root = os.path.join(ROOT, "dataset", "images")
    for dirpath, _dirnames, filenames in os.walk(img_root):
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            if ext.lower() == ".png" and stem in need and stem not in found:
                found[stem] = os.path.join(dirpath, fn)
    print(f"本地索引命中: {len(found)}/{len(need)}")

    out_img = os.path.join(ROOT, "final_images")
    os.makedirs(out_img, exist_ok=True)
    miss = 0
    for iid, src in found.items():
        dst = os.path.join(out_img, f"{iid}.png")
        if not os.path.exists(dst):
            with open(src, "rb") as a, open(dst, "wb") as b:
                b.write(a.read())
    miss = len(need) - len(found)

    # 4) 写 CSV (补 glyph_id 列, 与 eval100 格式一致)
    hdr = ["image_path", "calligrapher", "script", "character",
           "calligrapher_id", "script_id", "character_id", "glyph_id"]
    out_csv = os.path.join(ROOT, "5script", "eval500_top6.csv")
    rows = []
    for r in sorted(picked, key=lambda r: (int(r["script_id"]), r["calligrapher"])):
        row = dict(r)
        row["glyph_id"] = r["character_id"]
        rows.append(row)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {out_csv}: {len(rows)} 行 (缺失图片 {miss})")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""远程: 为楷/隶书体每个可用字采样一张 GT skeleton 图(降采样到256)，
输出到本地可拉回的目录 std_gt/<script>/<char_hex>.png。
用法: python _sample_gt_skel.py [n]  (n=每书体采样字数量,默认 30)
"""
import csv, os, sys, json
from collections import defaultdict
import numpy as np, cv2
from PIL import Image

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rows = list(csv.DictReader(open("5script/train.csv", encoding="utf-8")))
# script_id -> char -> 样本图 id (image_path)  取每个 glyph 第一张
glyph = defaultdict(dict)
for r in rows:
    sid = int(r["script_id"])
    ch = r["character"]
    if ch not in glyph[sid]:
        glyph[sid][ch] = os.path.basename(r["image_path"])[:-4]

# script_id: 0=楷,4=隶 (从之前 stats 知道)
OUT = "std_gt"
os.makedirs(OUT, exist_ok=True)
names = {0: "kai", 4: "li"}
report = {}
for sid, key in names.items():
    chars = sorted(glyph[sid].keys())[:N]
    dout = os.path.join(OUT, key)
    os.makedirs(dout, exist_ok=True)
    got = 0
    for ch in chars:
        imgid = glyph[sid][ch]
        skf = f"final_skeleton/{imgid}.png"
        if not os.path.exists(skf):
            continue
        im = Image.open(skf).convert("L").resize((256,256), Image.NEAREST)
        im.save(os.path.join(dout, f"U+{ord(ch):05X}.png"))
        got += 1
    print(f"[{key}] 采样 {got}/{N} 字 GT skeleton -> {dout}")
    report[key] = got
with open("std_gt_report.json","w",encoding="utf-8") as f:
    json.dump(report,f)

# -*- coding: utf-8 -*-
"""验证膨胀骨架相似度真值是否反映"结构相似"直觉。"""
import sys, os, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cv2

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

def load():
    d = "data/skel/std_skeleton_d3/kai"
    out = {}
    for p in glob.glob(os.path.join(d, "U+*.png")):
        cp = int(os.path.basename(p)[2:].replace(".png", ""), 16)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        out[cp] = img
    return out

skel = load()

def sim(a, b, k=4):
    # a,b 灰度图 (0-255), 笔画=暗. 转成 亮=笔画
    am = (a < 128).astype(np.uint8)
    bm = (b < 128).astype(np.uint8)
    ak = cv2.dilate(am, np.ones((k, k), np.uint8))
    bk = cv2.dilate(bm, np.ones((k, k), np.uint8))
    inter = (ak & bk).sum()
    un = (ak | bk).sum()
    return inter / (un + 1e-8), am.sum(), bm.sum()

pairs = [("土","士"),("大","太"),("人","入"),("日","曰"),("天","夫"),("刀","力"),
         ("申","由"),("王","玉"),("牛","午"),("己","已"),("海","深"),("木","林"),
         ("水","火"),("山","日"),("好","对"),("明","朋"),("休","体")]
import random
random.seed(0)
for a, b in pairs:
    if ord(a) in skel and ord(b) in skel:
        iou, na, nb = sim(skel[ord(a)], skel[ord(b)])
        print(f"  {a}/{b}: dil-IoU={iou:.3f}  笔画px: {a}={na} {b}={nb}")

# 随机不同字基线
chars = [c for c in skel.keys() if 0x4e00 <= c < 0x9fff]
rng = random.Random(1)
random_sims = []
for _ in range(2000):
    ca, cb = rng.sample(chars, 2)
    iou, _, _ = sim(skel[ca], skel[cb])
    random_sims.append(iou)
print(f"\n随机对 dil-IoU: mean={np.mean(random_sims):.3f} std={np.std(random_sims):.3f}")

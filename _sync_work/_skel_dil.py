# -*- coding: utf-8 -*-
import sys, os, glob, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cv2
os.chdir("/root/Workspace/xy/DiT")
skel = {}
for p in glob.glob("data/skel/std_skeleton_d3/kai/U+*.png"):
    cp = int(os.path.basename(p)[2:].replace(".png", ""), 16)
    skel[cp] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

def sk(a): return (a > 128).astype(np.uint8)

def sim(a, b, k):
    ak = cv2.dilate(sk(a), np.ones((k, k), np.uint8))
    bk = cv2.dilate(sk(b), np.ones((k, k), np.uint8))
    return (ak & bk).sum() / (ak | bk).sum() + 1e-8

pairs = [("土","士"),("大","太"),("人","入"),("日","曰"),("天","夫"),("刀","力"),
         ("申","由"),("王","玉"),("牛","午"),("己","已"),("海","深"),("木","林"),
         ("水","火"),("山","日"),("好","对"),("明","朋"),("休","体"),("休","林")]
chars = [c for c in skel if 0x4e00 <= c < 0x9fff]
rng = random.Random(1)
for k in [2, 3, 5]:
    print(f"=== dilation k={k} ===")
    rs = [sim(skel[c1], skel[c2], k) for c1, c2 in [rng.sample(chars, 2) for _ in range(3000)]]
    print(f"  随机对 mean={np.mean(rs):.3f}  top10%={np.percentile(rs,90):.3f}")
    for a, b in pairs:
        if ord(a) in skel and ord(b) in skel:
            print(f"    {a}/{b}: {sim(skel[ord(a)], skel[ord(b)], k):.3f}")

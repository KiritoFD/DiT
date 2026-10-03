# -*- coding: utf-8 -*-
import sys, os, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cv2
os.chdir("/root/Workspace/xy/DiT")
skel = {}
for p in glob.glob("data/skel/std_skeleton_d3/kai/U+*.png"):
    cp = int(os.path.basename(p)[2:].replace(".png", ""), 16)
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    skel[cp] = img
print("size", list(skel.values())[0].shape)
for ch in "土士大太日曰木林水":
    img = skel[ord(ch)]
    print(f"{ch}: mean={img.mean():.1f} 白线(>128)={(img>128).sum()}")
# 笔画 = img > 128
def iou0(a, b):
    am = (a > 128).astype(bool); bm = (b > 128).astype(bool)
    return (am & bm).sum()/(am | bm).sum() + 1e-8
def cos0(a, b):
    am = (a > 128).astype(bool); bm = (b > 128).astype(bool)
    return (am & bm).sum()/(np.sqrt(am.sum()*bm.sum()) + 1e-8)
for a, b in [("土","士"),("大","太"),("木","林"),("水","火"),("海","深"),("好","对")]:
    if ord(a) in skel and ord(b) in skel:
        print(f"  {a}/{b}: 像素IoU={iou0(skel[ord(a)],skel[ord(b)]):.3f}  余弦={cos0(skel[ord(a)],skel[ord(b)]):.3f}")

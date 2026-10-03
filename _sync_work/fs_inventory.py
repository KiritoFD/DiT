#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""few-shot 选题清单：把 wild_extract 的 78 个文件夹过四道闸，输出可用主题排名。

四道闸（缺一都会让结论作废，全部实测踩过）:
  1) 书家**未训过** —— 87-pair 表里 45 位书家；同名(含异体字)即算已训，
     否则"全新书家"的前提不成立（文档 74 的 B 组 24 个候选里 22 个死在这条）。
  2) 书体在分布内 —— v15a 只有 楷(0)/行(3)/隶(4)；草/篆 是分布外（怀素·草=混淆项）。
  3) 背景极性=白底墨字 —— 模型是白底训练的，黑底(拓片)会把 baseline 排序
     变成"脏污程度排序"（文档 74 §2.1）。黑底可用 255-a 矫正后复用。
  4) std 骨架够用 —— few-shot 的 g 条件借用 50k 里**同一个字**的标准骨架，
     需要 >= 150 个"训练50+评测100"互不重叠、且有 std 骨架的字。
"""
import collections
import csv
import json
import os
import re
import sys

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
WILD = "/root/Workspace/xy/HCSU/wild_extract"
NEED = 150                      # 50 train + 100 eval，按字互斥

# 简化/繁体、异体字归一：同一书家的不同写法视作同一人
VARIANT = {"徵": "征", "曲": "曲"}


def norm_name(n):
    return "".join(VARIANT.get(c, c) for c in n)


SCRIPT_ID = {"楷": "0", "行": "3", "隶": "4"}

# ── 1) 已训书家名单 ──────────────────────────────────────────────────────────
m = json.load(open("assets/callig_script_id_map.json", encoding="utf-8"))
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
id2name = {}
for r in rows:
    id2name.setdefault(r["calligrapher_id"], r["calligrapher"])
trained = {}
for key in m["pair_map"]:
    cid, sid = key.split(":")
    trained.setdefault(norm_name(id2name.get(cid, "?" + cid)), set()).add(sid)

# ── 2) 每个字可用的 std 骨架 ─────────────────────────────────────────────────
char2std = collections.defaultdict(list)
for r in rows:
    if r.get("std_path"):
        char2std[r["character"]].append(r["std_path"])

# ── 3) wild 文件夹已被 50k 用掉多少 ──────────────────────────────────────────
used = collections.Counter()
for r in rows:
    mm = re.match(r".*wild/([^/]+)/[^/]+$", r.get("src_image_path", "") or "")
    if mm:
        used[mm.group(1)] += 1

out = []
for folder in sorted(os.listdir(WILD)):
    p = os.path.join(WILD, folder)
    if not os.path.isdir(p) or "-" not in folder:
        continue
    cal, _, sc = folder.partition("-")
    files = sorted(f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        continue
    # 极性抽样（最多 120 张，够判背景底色）
    samp = files[::max(1, len(files) // 120)]
    ink, near_white = [], []
    for f in samp:
        try:
            a = np.asarray(Image.open(os.path.join(p, f)).convert("L"), dtype=np.uint8)
        except Exception:
            continue
        ink.append((a < 128).mean())
        near_white.append((a > 245).mean())
    ink_mean = float(np.mean(ink)) if ink else 1.0
    white_mean = float(np.mean(near_white)) if near_white else 0.0
    # 白底墨字: 暗像素少(ink<0.35) 且 近白像素多; 否则判黑底
    polarity = "white" if ink_mean < 0.35 and white_mean > 0.25 else "BLACK"

    chars = {os.path.splitext(f)[0] for f in files}
    with_std = {c for c in chars if c in char2std}
    ok_script = sc in SCRIPT_ID
    ok_cal = norm_name(cal) not in trained
    gate1 = "NEW" if ok_cal else "trained"
    gate2 = sc if ok_script else f"{sc}!OOD"
    verdict = ("OK" if polarity == "white" else "FIX(反相)") if (ok_cal and ok_script) else "SKIP"
    if verdict.startswith(("OK", "FIX")) and len(with_std) < NEED:
        verdict = f"SKIP(字不足{len(with_std)})"
    out.append((verdict != "SKIP", folder, cal, gate2, polarity, len(files),
                len(with_std), used.get(folder, 0), gate1, verdict, ink_mean))

print(f"{'folder':<14}{'书体':<8}{'极性':<7}{'总张数':>7}{'可用字':>7}{'已用50k':>8}  判定")
for _, folder, cal, sc, pol, n, ws, u, g1, verdict, ink in sorted(
        out, key=lambda z: (-z[0], z[1])):
    print(f"{folder:<14}{sc:<8}{pol:<7}{n:>7}{ws:>7}{u:>8}  {verdict}  ink={ink:.3f}")

avail = [z for z in out if z[0]]
print(f"\n可用主题 {len(avail)} 个: " + ", ".join(z[1] for z in avail))
print("其中白底免矫正: " + ", ".join(z[1] for z in avail if z[4] == "white"))
print("需反相矫正:     " + ", ".join(z[1] for z in avail if z[4] == "BLACK"))

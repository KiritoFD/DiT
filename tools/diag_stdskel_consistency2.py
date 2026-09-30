#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_stdskel_consistency2.py — 决定性检查: 解码成墨迹图比 IoU。

为什么需要第二个脚本: 只比 latent cosine 有两个歧义
  (a) 骨架 latent 稀疏, 任意两字 cosine 都偏高 -> 判别力弱 (实测异字 0.787);
  (b) 若 fame 的 std_skel1 其实是"按(书家,字)"而非"按字", 比较就不成立。
本脚本:
  [1] 先验证两边各自"同字是否恒定" (判定是否 character-level);
  [2] 再把共同字的骨架 latent 解码成 256x256 墨迹图, 算 IoU / 前景比;
  [3] 用 IoU 直接判读"两套 std 骨架是否同源"。
"""
import csv
import glob
import os
import sys

import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

TOP10_SH = "data/top10_style23/shards_std"
TOP10_CSV = "assets/train_top10_style23.csv"
FAME_SH = "data/archive/legacy_skeletons/std_skel1_latents_fame"
FAME_CSV = "data/archive/legacy_csv/train_fame.csv"
VAE_PATH = "pretrained_models/sd-vae-ft-ema"
SCALING = 0.18215


def load_shards(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                m[int(i)] = z["latents"][j].astype(np.float32)
    return m


def rows_of(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main():
    print("=" * 78)
    print("[1] 两边各自: 同字是否恒定? (判定是否 character-level)")
    print("=" * 78)
    for tag, shd, csvp in (("top10", TOP10_SH, TOP10_CSV), ("fame", FAME_SH, FAME_CSV)):
        sh = load_shards(shd)
        rs = rows_of(csvp)
        by = {}
        for r in rs:
            iid = int(r["img_id"]) if "img_id" in r else int(
                os.path.basename(r["image_path"])[:-4])
            by.setdefault(r["character"], []).append(iid)
        multi = {c: v for c, v in by.items() if len(v) > 1}
        same, diff = 0, 0
        for c, v in list(multi.items())[:400]:
            ref = sh.get(v[0])
            if ref is None:
                continue
            for i in v[1:]:
                cur = sh.get(i)
                if cur is None:
                    continue
                if np.array_equal(ref, cur):
                    same += 1
                else:
                    diff += 1
        print("  %-6s 多成员字组(抽样400): 组内 bit-exact 相同=%d 不同=%d"
              % (tag, same, diff))
        print("         => %s" % ("character-level (同字恒定)" if diff == 0
                                  else "**不是** character-level (同字会变)"))

    print()
    print("=" * 78)
    print("[2] 解码共同字的骨架 -> 256x256 墨迹图")
    print("=" * 78)
    t10s = load_shards(TOP10_SH)
    fms = load_shards(FAME_SH)

    def cmap(sh, csvp):
        out = {}
        for r in rows_of(csvp):
            c = r["character"]
            if c in out:
                continue
            iid = int(r["img_id"]) if "img_id" in r else int(
                os.path.basename(r["image_path"])[:-4])
            if iid in sh:
                out[c] = sh[iid]
        return out

    t10, fm = cmap(t10s, TOP10_CSV), cmap(fms, FAME_CSV)
    common = sorted(set(t10) & set(fm))
    print("  共同字 = %d" % len(common))

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_PATH).eval()

    def decode_to_ink(lat):
        with torch.no_grad():
            z = torch.from_numpy(lat)[None].float() / SCALING
            img = vae.decode(z).sample[0]          # (3,256,256) in [-1,1]
        g = img.mean(0).numpy()                    # 灰度
        return (g < 0.0)                           # 墨迹 bool (白底黑字)

    N = 24
    rng = np.random.default_rng(0)
    pick = [common[i] for i in rng.choice(len(common), min(N, len(common)), replace=False)]
    ious, fg_a, fg_b = [], [], []
    for c in pick:
        a = decode_to_ink(t10[c])
        b = decode_to_ink(fm[c])
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        ious.append(inter / max(union, 1))
        fg_a.append(a.mean())
        fg_b.append(b.mean())
    ious = np.array(ious)
    print("  n=%d  IoU: mean=%.4f median=%.4f min=%.4f max=%.4f"
          % (len(ious), ious.mean(), np.median(ious), ious.min(), ious.max()))
    print("  前景占比: top10=%.4f  fame=%.4f" % (np.mean(fg_a), np.mean(fg_b)))

    print()
    print("=" * 78)
    print("[3] 判读")
    print("=" * 78)
    if ious.mean() > 0.85:
        print("  => ✅ 两套 std 骨架基本同源 (IoU %.3f), top10 SkelNet 可直接复用" % ious.mean())
    elif ious.mean() > 0.6:
        print("  => ⚠️ 结构相近但有可见差异 (IoU %.3f), 属轻度分布外" % ious.mean())
    else:
        print("  => ❌ 明显不同源 (IoU %.3f), 直接复用不可信" % ious.mean())
    print("  抽样字 IoU: %s" % [(c, round(float(v), 3)) for c, v in zip(pick, ious)])


if __name__ == "__main__":
    main()

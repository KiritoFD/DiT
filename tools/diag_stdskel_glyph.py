#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_stdskel_glyph.py — 修正版: 按 **glyph_id** 对齐比较两套 std 骨架。

修正了什么: 上一版按 `character` 字符串分组, 但 top10 的 csv 同时有 `character_id`
与 `glyph_id` —— **同一个字可以有多个字形变体**, 变体的 std 骨架不同。
按字符串分组会把变体混在一起, 于是"同字"比较其实在比不同的字形 -> 假阴性。

本脚本:
  [A] 按 glyph_id 分组, 验证各自"同 glyph 是否恒定";
  [B] 共同 glyph_id -> 解码比 IoU;
  [C] 判读。
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
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TOP10_SH = "data/top10_style23/shards_std"
TOP10_CSV = "assets/train_top10_style23.csv"
FAME_SH = "data/archive/legacy_skeletons/std_skel1_latents_fame"
FAME_CSV = "data/archive/legacy_csv/train_fame.csv"
SCALING = 0.18215
N_DEC = 12


def load_shards(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                m[int(i)] = z["latents"][j]
    return m


def key_map(csvp, key):
    """key(glyph_id) -> img_id (首次出现)"""
    out = {}
    for r in csv.DictReader(open(csvp, encoding="utf-8")):
        k = int(r[key])
        if k in out:
            continue
        iid = int(r["img_id"]) if "img_id" in r else int(
            os.path.basename(r["image_path"])[:-4])
        out[k] = iid
    return out


def main():
    print("[load] shards ...", flush=True)
    t10s = load_shards(TOP10_SH)
    fms = load_shards(FAME_SH)
    t10g = key_map(TOP10_CSV, "glyph_id")
    fmg = key_map(FAME_CSV, "glyph_id")
    print("[load] top10 glyphs=%d | fame glyphs=%d" % (len(t10g), len(fmg)), flush=True)

    # ---- [A] 同 glyph 是否恒定 ----
    print("\n=== [A] 同 glyph_id 是否恒定 ===", flush=True)
    for tag, sh, csvp in (("top10", t10s, TOP10_CSV), ("fame", fms, FAME_CSV)):
        by = {}
        for r in csv.DictReader(open(csvp, encoding="utf-8")):
            iid = int(r["img_id"]) if "img_id" in r else int(
                os.path.basename(r["image_path"])[:-4])
            by.setdefault(int(r["glyph_id"]), []).append(iid)
        multi = [(g, v) for g, v in by.items() if len(v) > 1][:300]
        same = diff = 0
        for g, v in multi:
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
        print("  %-6s 同 glyph 组内 bit-exact 相同=%d 不同=%d -> %s"
              % (tag, same, diff, "glyph-level (恒定)" if diff == 0 else "仍会变"),
              flush=True)

    # ---- [B] 共同 glyph -> IoU ----
    print("\n=== [B] 共同 glyph_id -> 墨迹 IoU ===", flush=True)
    common = sorted(set(t10g) & set(fmg))
    print("  共同 glyph_id=%d" % len(common), flush=True)
    rng = np.random.default_rng(0)
    pick = [common[i] for i in rng.choice(len(common), min(N_DEC, len(common)), replace=False)]

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").eval()

    def ink(iid, sh):
        with torch.no_grad():
            z = torch.from_numpy(sh[iid][None].astype(np.float32)) / SCALING
            g = vae.decode(z).sample[0].mean(0).numpy()
        return g < 0.0

    ious, fa, fb = [], [], []
    for k, gid in enumerate(pick):
        a, b = ink(t10g[gid], t10s), ink(fmg[gid], fms)
        u = np.logical_or(a, b).sum()
        ious.append(np.logical_and(a, b).sum() / max(u, 1))
        fa.append(a.mean())
        fb.append(b.mean())
        print("    [%2d/%d] glyph=%-6d IoU=%.3f (top10 fg=%.4f, fame fg=%.4f)"
              % (k + 1, len(pick), gid, ious[-1], fa[-1], fb[-1]), flush=True)
    ious = np.array(ious)
    print("\n  IoU: mean=%.4f median=%.4f min=%.4f max=%.4f"
          % (ious.mean(), np.median(ious), ious.min(), ious.max()), flush=True)
    print("  前景比: top10=%.4f fame=%.4f" % (np.mean(fa), np.mean(fb)), flush=True)

    print("\n=== [C] 判读 ===", flush=True)
    if ious.mean() > 0.85:
        print("  => 同源 (IoU %.3f)" % ious.mean())
    elif ious.mean() > 0.6:
        print("  => 结构相近但有差异 (IoU %.3f)" % ious.mean())
    else:
        print("  => 仍不同源 (IoU %.3f)" % ious.mean())


if __name__ == "__main__":
    main()

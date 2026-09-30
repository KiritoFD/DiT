#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_stdskel_fast.py — 精简版: 只回答"两套 std 骨架是否同源"。

与 consistency2 的区别: shard 只加载一次、字组抽样小、逐行 flush 输出到日志,
避免之前 17 分钟不吐字的问题。

判定:
  [A] 各自"同字是否恒定" -> 确认是 character-level;
  [B] 解码 12 个共同字 -> 墨迹 IoU (直接可比);
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
            ids, lat = z["img_ids"], z["latents"]
            for j, i in enumerate(ids):
                m[int(i)] = lat[j]
    return m


def char_to_id(csvp):
    out = {}
    for r in csv.DictReader(open(csvp, encoding="utf-8")):
        c = r["character"]
        if c in out:
            continue
        iid = int(r["img_id"]) if "img_id" in r else int(
            os.path.basename(r["image_path"])[:-4])
        out[c] = iid
    return out


def main():
    print("[load] top10 shards ...", flush=True)
    t10s = load_shards(TOP10_SH)
    print("[load] fame shards ...", flush=True)
    fms = load_shards(FAME_SH)
    t10 = char_to_id(TOP10_CSV)
    fm = char_to_id(FAME_CSV)
    print("[load] top10 ids=%d chars=%d | fame ids=%d chars=%d"
          % (len(t10s), len(t10), len(fms), len(fm)), flush=True)

    # ---- [A] 同字是否恒定 ----
    print("\n=== [A] 同字是否恒定 (character-level?) ===", flush=True)
    for tag, sh, csvp in (("top10", t10s, TOP10_CSV), ("fame", fms, FAME_CSV)):
        by = {}
        for r in csv.DictReader(open(csvp, encoding="utf-8")):
            iid = int(r["img_id"]) if "img_id" in r else int(
                os.path.basename(r["image_path"])[:-4])
            by.setdefault(r["character"], []).append(iid)
        multi = [(c, v) for c, v in by.items() if len(v) > 1][:300]
        same = diff = 0
        for c, v in multi:
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
        print("  %-6s 组内 bit-exact 相同=%d 不同=%d -> %s"
              % (tag, same, diff, "character-level" if diff == 0 else "非 character-level"),
              flush=True)

    # ---- [B] 解码比 IoU ----
    print("\n=== [B] 解码共同字 -> 墨迹 IoU ===", flush=True)
    common = sorted(set(t10) & set(fm))
    print("  共同字=%d" % len(common), flush=True)
    rng = np.random.default_rng(0)
    pick = [common[i] for i in rng.choice(len(common), N_DEC, replace=False)]

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").eval()
    print("  VAE 就绪", flush=True)

    def ink(iid, sh):
        with torch.no_grad():
            z = torch.from_numpy(sh[iid][None].astype(np.float32)) / SCALING
            g = vae.decode(z).sample[0].mean(0).numpy()
        return g < 0.0

    ious, fa, fb = [], [], []
    for k, c in enumerate(pick):
        a = ink(t10[c], t10s)
        b = ink(fm[c], fms)
        u = np.logical_or(a, b).sum()
        ious.append(np.logical_and(a, b).sum() / max(u, 1))
        fa.append(a.mean())
        fb.append(b.mean())
        print("    [%2d/%d] %s IoU=%.3f (top10 fg=%.4f, fame fg=%.4f)"
              % (k + 1, len(pick), c, ious[-1], fa[-1], fb[-1]), flush=True)
    ious = np.array(ious)
    print("\n  IoU: mean=%.4f median=%.4f min=%.4f max=%.4f"
          % (ious.mean(), np.median(ious), ious.min(), ious.max()), flush=True)
    print("  前景比: top10=%.4f fame=%.4f" % (np.mean(fa), np.mean(fb)), flush=True)

    print("\n=== [C] 判读 ===", flush=True)
    if ious.mean() > 0.85:
        print("  => 同源 (IoU %.3f): top10 SkelNet 可直接用于 fame std 骨架" % ious.mean())
    elif ious.mean() > 0.6:
        print("  => 结构相近但有差异 (IoU %.3f): 轻度分布外" % ious.mean())
    else:
        print("  => 不同源 (IoU %.3f): 直接复用不可信" % ious.mean())


if __name__ == "__main__":
    main()

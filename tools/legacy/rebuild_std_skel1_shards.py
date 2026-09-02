# -*- coding: utf-8 -*-
"""rebuild_std_skel1_shards.py — 从 skel_bank_std1.npz 重建训练 shards.

修复: MCCDLatentDataset 的 img_id 是 image_path 里的图像编号 (非 csv 行号).
输入: skel_bank_std1.npz (keys="script|char", latents) + train_fame.csv
输出: std_skel1_latents_fame/shard_*.npz (latents + 真实 img_ids)
"""
import os
import sys
import csv
import argparse
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_fame.csv")
    ap.add_argument("--bank", default="skel_bank_std1.npz")
    ap.add_argument("--out-dir", default="std_skel1_latents_fame")
    ap.add_argument("--shard-size", type=int, default=2592)
    args = ap.parse_args()

    bank = np.load(args.bank, allow_pickle=True)
    keys = list(bank["keys"])
    latents = bank["latents"]
    key2i = {k: i for i, k in enumerate(keys)}
    # fallback: char -> 任意书体条目 (楷优先, 字形最规范)
    char2i = {}
    for prio in ("楷", "隶", "行", "草", "篆"):
        for k, i in key2i.items():
            sc, ch = k.split("|", 1)
            if sc == prio:
                char2i.setdefault(ch, i)
    print(f"[bank] {len(keys)} entries, latents {latents.shape}, "
          f"char-fallback covers {len(char2i)} chars")

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    print(f"[csv] {len(rows)} rows")

    os.makedirs(args.out_dir, exist_ok=True)
    # 清掉旧的错误 shards
    for f in os.listdir(args.out_dir):
        if f.startswith("shard_"):
            os.remove(os.path.join(args.out_dir, f))

    shard_l, shard_i, n_shard, miss, fallback = [], [], 0, 0, 0

    def flush():
        nonlocal shard_l, shard_i, n_shard
        if not shard_l:
            return
        np.savez(os.path.join(args.out_dir, f"shard_{n_shard:05d}.npz"),
                 latents=np.stack(shard_l).astype(np.float16),
                 img_ids=np.array(shard_i, dtype=np.int64))
        n_shard += 1
        shard_l, shard_i = [], []

    for r in rows:
        img_id = int(os.path.basename(r["image_path"]).split(".")[0])
        k = f'{r["script"]}|{r["character"]}'
        i = key2i.get(k)
        if i is None:
            i = char2i.get(r["character"])
            if i is None:
                miss += 1
                continue
            fallback += 1
        shard_l.append(latents[i])
        shard_i.append(img_id)
        if len(shard_l) >= args.shard_size:
            flush()
    flush()
    print(f"[shards] {n_shard} shards, miss={miss}, cross-script-fallback={fallback} "
          f"-> {args.out_dir}/")
    print("done.")


if __name__ == "__main__":
    main()

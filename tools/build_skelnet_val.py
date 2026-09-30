#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_skelnet_val.py — 建 SkelNet standalone 的**早停验证集**（按字符留出）。

为什么按字符留出:
  最终评测口径是 strict(未见过该字) 与 seen(见过该字)。若验证集只是随机抽样本,
  同一个字仍在训练集里 -> 指标会高估泛化能力, 早停会停在一个"记住字形"的点上。
  按**字符**留出后, 验证指标 ≈ strict 口径, 早停才有意义。

产物: assets/val_skelnet.csv   (与原训练 csv 同列, 只有被留出的行)
      assets/train_top10_style23_minusval.csv  (剔除留出字后的训练集)
      assets/val_skelnet_ids.npz  (img_ids, 方便训练脚本直接过滤)
"""
import os
import sys
import csv
import re
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

SRC = 'assets/train_top10_style23.csv'
VAL_CSV = 'assets/val_skelnet.csv'
TR_CSV = 'assets/train_top10_style23_minusval.csv'
IDS_NPZ = 'assets/val_skelnet_ids.npz'
FRAC = 0.10
SEED = 0


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    print(f"[src] {SRC}: {len(rows)} 行")
    chars = sorted({r['character'] for r in rows})
    rng = np.random.RandomState(SEED)
    n_val = max(1, int(len(chars) * FRAC))
    val_chars = set(np.array(chars)[rng.choice(len(chars), n_val, replace=False)].tolist())
    print(f"[split] 字 {len(chars)} 个 -> 留出 {len(val_chars)} 个 ({FRAC:.0%}), 种子 {SEED}")

    val = [r for r in rows if r['character'] in val_chars]
    tr = [r for r in rows if r['character'] not in val_chars]
    print(f"[split] 验证 {len(val)} 行 / 训练 {len(tr)} 行 "
          f"(留出 {len(val)/len(rows):.2%})")

    # 书家/书体覆盖
    for nm, rs in (('验证', val), ('训练', tr)):
        slots = sorted({r['slot_name'] for r in rs})
        print(f"  {nm}: 字 {len({r['character'] for r in rs})}  槽位 {len(slots)}  "
              f"书家 {len({r['calligrapher'] for r in rs})}")

    # 与最终评测集(strict84)的字是否重叠 -> 决定 val 指标能否代理 strict
    for ev in ('assets/eval_top10_strict_subset84.csv', 'assets/eval_top10_seen_20.csv'):
        if not os.path.exists(ev):
            print(f"  [{ev}] 不存在, 跳过")
            continue
        er = list(csv.DictReader(open(ev, encoding='utf-8')))
        ec = {r['character'] for r in er if 'character' in r}
        if not ec:
            ec = {re.sub(r'.*/(.).*', r'\1', r.get('image_path', '')) for r in er}
        inter = ec & val_chars
        print(f"  [{os.path.basename(ev)}] {len(er)} 行, 字 {len(ec)} 个; "
              f"与留出字重叠 {len(inter)} -> "
              f"{'⚠ 有重叠, val 指标会偏高' if inter else '✓ 无重叠, val 指标可代理 strict'}")

    for p, rs in ((VAL_CSV, val), (TR_CSV, tr)):
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rs)
        print(f"[out] {p}: {len(rs)} 行")

    ids = np.array(sorted({int(r['img_id']) for r in val}), dtype=np.int64)
    np.savez_compressed(IDS_NPZ, img_ids=ids, chars=np.array(sorted(val_chars)))
    print(f"[out] {IDS_NPZ}: {len(ids)} 个 img_id")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_callig_map.py — 生成干净书家词表映射 json (raw calligrapher_id -> 0..N-1).

用法: python tools/build_callig_map.py --csv a.csv,b.csv --out 5script/callig_id_map.json
"""
import argparse
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from src.utils.callig_map import build_callig_id_map, save_callig_id_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="逗号分隔的 csv 路径 (train,eval)")
    ap.add_argument("--out", default="5script/callig_id_map.json")
    args = ap.parse_args()

    paths = [p for p in args.csv.split(",") if p.strip()]
    mapping = build_callig_id_map(paths)
    save_callig_id_map(mapping, args.out)
    print(f"书家词表: {len(mapping)} 个 -> {args.out}")
    for raw, idx in sorted(mapping.items()):
        print(f"  raw_id={raw:>4} -> idx={idx}")


if __name__ == "__main__":
    main()
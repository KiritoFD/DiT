# -*- coding: utf-8 -*-
"""audit_strict84.py — 项目自带的 assets/eval_top10_strict_subset84.csv 到底"严格"在哪?

已知: 84 行 / 84 个不同字, 其中 82 个字的**字**在训练清单里出现过 (字级未见率仅 2/84)。
所以"严格未见字"这个说法需要重新核对 —— 本脚本把三个层级都算出来。
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL = "exp-std/csv/eval200_fixed.csv"
STRICT = "assets/eval_top10_strict_subset84.csv"
TRAIN = "exp-std/csv/train.csv"


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    st, ev, tr = rows(STRICT), rows(EVAL), rows(TRAIN)
    t_ip = {str(r["image_path"]).strip() for r in tr}
    t_id = {str(r.get("img_id", "")).strip() for r in tr}
    t_src = {str(r.get("src_image_path", "")).strip() for r in tr}
    t3 = {tuple(str(r.get(k, "")).strip() for k in ("calligrapher", "script", "character")) for r in tr}
    t2 = {tuple(str(r.get(k, "")).strip() for k in ("calligrapher", "character")) for r in tr}
    t1 = {str(r["character"]).strip() for r in tr}

    ev_ip = {str(r["image_path"]).strip() for r in ev}
    print(f"[strict84] {len(st)} 行; 是否 eval200_fixed 的子集: "
          f"{sum(1 for r in st if str(r['image_path']).strip() in ev_ip)}/{len(st)}")
    print()
    print(f"  {'层级':34s} {'与训练重叠':>10s} / 84")
    for nm, f in (("① img_id", lambda r: str(r.get('img_id', '')).strip() in t_id),
                  ("② image_path", lambda r: str(r['image_path']).strip() in t_ip),
                  ("③ src_image_path", lambda r: str(r.get('src_image_path', '')).strip() in t_src),
                  ("④ (书家,书体,字)", lambda r: tuple(str(r.get(k, '')).strip() for k in
                                                      ("calligrapher", "script", "character")) in t3),
                  ("⑤ (书家,字)", lambda r: tuple(str(r.get(k, '')).strip() for k in
                                                  ("calligrapher", "character")) in t2),
                  ("   仅字", lambda r: str(r["character"]).strip() in t1)):
        n = sum(1 for r in st if f(r))
        print(f"  {nm:34s} {n:10d}")

    print()
    print("[顺带] eval200_fixed 187 的三个层级 (与 train.csv 比):")
    for nm, f in (("② image_path", lambda r: str(r['image_path']).strip() in t_ip),
                  ("③ src_image_path", lambda r: str(r.get('src_image_path', '')).strip() in t_src),
                  ("④ (书家,书体,字)", lambda r: tuple(str(r.get(k, '')).strip() for k in
                                                      ("calligrapher", "script", "character")) in t3),
                  ("   仅字", lambda r: str(r["character"]).strip() in t1)):
        n = sum(1 for r in ev if f(r))
        print(f"  {nm:34s} {n:3d}/187")
    return 0


if __name__ == "__main__":
    sys.exit(main())

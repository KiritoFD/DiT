# -*- coding: utf-8 -*-
"""
基于官方 MCCD_Character label 构建 汉字标签 -> 连续数字 id 的映射表。

字段解析（关键）：
  - 5 段：字-字体-朝代-书家/出处-id
  - 6 段：字-字体-朝代-书家-碑帖(忽略)-id
书家取值：5段[3] / 6段[3]。碑帖名忽略。
书家清洗：null/空/纯数字(六体编号) → 统一归入 "others" 一个 id。

输出 _id_maps.json：
  {
    "character":    {"字": id, ...},   # 7765
    "script":       {"字体": id, ...}, # 12
    "calligrapher": {"书家": id, ...}, # 真实书家 + "others"
    "meta": {
      "calligrapher_dirty_merged": {"null":0, "":0, "纯数字数":N},
      "raw_calligrapher_count": ...,
      ...
    }
  }
id 从 0 开始连续分配。
"""
import os, json, argparse
from collections import Counter

ROOT = "MCCD/MCCD/MCCD_Character/trainset_dataset"
OTHERS = "others"


def load_rows(root):
    rows = []
    for split in ("train", "test"):
        p = os.path.join(root, f"{split}_label.txt")
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    rows.append(line)
    return rows


def clean_calligrapher(cal):
    """返回 (清洗后书家 or OTHERS, 是否脏)"""
    s = cal.strip()
    if s in ("", "null", "None", "nan"):
        return OTHERS, True
    if s.isdigit():
        return OTHERS, True
    return s, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="_id_maps.json")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.root)
    chars, scripts, callis = {}, {}, {}
    seg_ct = Counter()
    dirty_reasons = Counter()
    per_split = Counter()

    for line in rows:
        p = line.split("-")
        seg_ct[len(p)] += 1
        if len(p) == 5:
            c, s, cal = p[0], p[1], p[3]
            per_split["5seg"] += 1
        elif len(p) == 6:
            c, s, cal = p[0], p[1], p[3]
            per_split["6seg"] += 1
        else:
            per_split["other"] += 1
            continue
        for d, key in ((chars, c), (scripts, s)):
            if key not in d:
                d[key] = len(d) + args.start
        cal, is_dirty = clean_calligrapher(cal)
        if is_dirty:
            if p[3].strip() == "":
                dirty_reasons["empty"] += 1
            elif p[3].strip() == "null":
                dirty_reasons["null"] += 1
            elif p[3].strip().isdigit():
                dirty_reasons["digit(六体编号)"] += 1
            else:
                dirty_reasons[f"other:{p[3]!r}"] += 1
        if cal not in callis:
            callis[cal] = len(callis) + args.start

    print("rows:", len(rows), "seg_ct:", dict(seg_ct), "per_split:", dict(per_split))
    print("character:", len(chars), "script:", len(scripts), "calligrapher(含others):", len(callis))
    print("dirty merged reasons:", dict(dirty_reasons))
    print("calligrapher real(非others):", len(callis) - 1)

    out = {
        "character": chars,
        "script": scripts,
        "calligrapher": callis,
        "meta": {
            "rows": len(rows),
            "seg5": per_split.get("5seg", 0),
            "seg6": per_split.get("6seg", 0),
            "dirty_merged": dict(dirty_reasons),
            "counts": {
                "character": len(chars),
                "script": len(scripts),
                "calligrapher": len(callis),
                "calligrapher_real": len(callis) - 1,
            },
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()

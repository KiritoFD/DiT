# -*- coding: utf-8 -*-
"""
build_eval_strict_midclean.py — 对 mid_clean 训练集的组合泛化 eval 集.

标准 (用户定义的"凸集"口径, 比旧 eval_strict_top6 更严):
  - (script, character)   在 train csv 中出现过   -> 字要素有覆盖
  - (script, calligrapher) 在 train csv 中出现过   -> 书家要素有覆盖
  - (script, calligrapher, character) 三元组未出现过 -> 组合没覆盖
  - img_id 不在 train csv 中                        -> 图片本身未训练
  - script 由 --scripts 指定 (mid_clean: 楷0/行3/隶4)

每个 (script, character) 只选一张 (unique glyph), 按 script 均衡,
总计上限 --total (默认 600)。用 archive/final_manifest.json 的**全量**
候选池 (含未进入任何 train csv 的图)。

用法 (远程):
  /opt/conda/bin/python tools/build_eval_strict_midclean.py \
      --train-csv 5script/train_mid_clean.csv \
      --manifest archive/final_manifest.json \
      --out 5script/eval_strict_midclean.csv --total 600
"""
import os
import csv
import json
import random
import argparse
from collections import Counter, defaultdict

import sys
sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_NAME_TO_ID = {"楷": 0, "行": 3, "隶": 4}
NUM_CHARACTERS = 7026
SEED = 42


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="5script/train_mid_clean.csv")
    ap.add_argument("--manifest", default="archive/final_manifest.json")
    ap.add_argument("--out", default="5script/eval_strict_midclean.csv")
    ap.add_argument("--total", type=int, default=600)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def img_id_of(path):
    return int(os.path.basename(path).replace(".png", ""))


def main():
    args = parse_args()
    print("Loading manifest...")
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    id2entry = {e["img_id"]: e for e in manifest}
    print(f"  manifest: {len(id2entry)} entries")

    print(f"Loading {args.train_csv} (authoritative name->id maps)...")
    train_ids = set()
    train_char_map = {}    # (script, character) -> character_id
    train_calli_map = {}   # (script, calligrapher) -> calligrapher_id
    train_triples = set()  # (script, calligrapher, character)
    for r in csv.DictReader(open(args.train_csv, encoding="utf-8")):
        train_ids.add(img_id_of(r["image_path"]))
        skey = (r["script"], r["character"])
        train_char_map.setdefault(skey, int(r["character_id"]))
        ckey = (r["script"], r["calligrapher"])
        train_calli_map.setdefault(ckey, int(r["calligrapher_id"]))
        train_triples.add((r["script"], r["calligrapher"], r["character"]))
    print(f"  train imgs: {len(train_ids)}, (script,char): {len(train_char_map)}, "
          f"(script,calli): {len(train_calli_map)}, triples: {len(train_triples)}")

    cand = []
    for iid, e in id2entry.items():
        if iid in train_ids:
            continue
        sname = e.get("orig_script", "")
        if sname not in SCRIPT_NAME_TO_ID:
            continue
        char, calli = e.get("orig_char", ""), e.get("orig_calli", "")
        if (sname, char) not in train_char_map:
            continue
        if (sname, calli) not in train_calli_map:
            continue
        if (sname, calli, char) in train_triples:
            continue
        cand.append((iid, e, sname, char, calli))
    print(f"  strict candidates: {len(cand)} "
          f"{dict(Counter(c[2] for c in cand))}")

    # unique glyph per (script, char), balanced across scripts, up to --total
    random.seed(args.seed)
    by_script_glyph = defaultdict(lambda: defaultdict(list))
    for iid, e, sname, char, calli in cand:
        by_script_glyph[sname][(sname, char)].append((iid, e, char, calli))
    n_scripts = len(by_script_glyph)
    per_script = args.total // max(n_scripts, 1)
    selected = []
    for sname in SCRIPT_NAME_TO_ID:
        glyphs = list(by_script_glyph.get(sname, {}).keys())
        random.shuffle(glyphs)
        take = glyphs[:per_script]
        for g in take:
            selected.append((sname, random.choice(by_script_glyph[sname][g])))
        print(f"  {sname}: {len(glyphs)} unique glyphs -> take {len(take)}")

    out_rows = []
    for sname, (iid, e, char, calli) in selected:
        sid = SCRIPT_NAME_TO_ID[sname]
        char_id = train_char_map[(sname, char)]
        calli_id = train_calli_map[(sname, calli)]
        out_rows.append({
            "image_path": f"final_imgs_256/{iid}.png",
            "calligrapher": calli,
            "script": sname,
            "character": char,
            "calligrapher_id": calli_id,
            "script_id": sid,
            "character_id": char_id,
            "glyph_id": sid * NUM_CHARACTERS + char_id,
        })
    random.shuffle(out_rows)

    # validation
    out_ids = {img_id_of(r["image_path"]) for r in out_rows}
    assert not (out_ids & train_ids), f"eval overlaps train! {len(out_ids & train_ids)}"
    assert len(out_ids) == len(out_rows), "duplicate img_ids"
    eval_chars = {r["character"] for r in out_rows}
    unseen = [c for c in eval_chars
              if not any((s, c) in train_char_map for s in SCRIPT_NAME_TO_ID)]
    assert not unseen, f"zero-shot chars leaked: {unseen[:10]}"
    for r in out_rows:
        assert (r["script"], r["calligrapher"], r["character"]) not in train_triples

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nDone: {len(out_rows)} rows -> {args.out}")
    print(f"scripts: {dict(Counter(r['script'] for r in out_rows))}, "
          f"uniq chars: {len(eval_chars)}, zero-shot: 0 (by construction)")


if __name__ == "__main__":
    main()

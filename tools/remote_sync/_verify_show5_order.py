# -*- coding: utf-8 -*-
"""核对: kailishu_eval.csv 前5行 的 image/char/script vs eval_samples 的 samples.json conds。
判断 poster 的 show5 样本顺序与 eval_samples 生成样本顺序是否一致。"""
import csv, os, json, glob

rows = list(csv.DictReader(open("kailishu_eval.csv", encoding="utf-8")))[:5]
print("=== kailishu_eval.csv 前5行 (show5 候选) ===")
for i, r in enumerate(rows):
    pid = os.path.basename(r["image_path"])[:-4]
    print(f"  [{i}] img={pid} char={r['character']} script={r['script']} "
          f"script_id={r['script_id']} callig={r['calligrapher']} glyph_id={r['glyph_id']}")

# eval_samples 最近 step 的 samples.json
steps = sorted(glob.glob("5script/results/v3b_xl_glyphcond/*/checkpoints/eval_samples/step*/samples.json"))
print(f"\n=== eval_samples 最新 step: {os.path.basename(os.path.dirname(steps[-1])) if steps else 'N/A'} ===")
if steps:
    sj = json.load(open(steps[-1], encoding="utf-8"))
    conds = sj["conds"]
    print("conds (callig, script, glyph_id):")
    for c in conds:
        print("   ", c)
    print("\n比对: conds 里的 (callig,glyph_id) 是否与 kailishu_eval 前5行的 (callig_id,glyph_id) 一致:")
    for i in range(min(5, len(conds), len(rows))):
        c = conds[i]; r = rows[i]
        match = (c[0] == int(r["calligrapher_id"]) and c[2] == int(r["glyph_id"]))
        print(f"  sample{i} cond={c} vs csv[{i}] callig={r['calligrapher_id']} gid={r['glyph_id']} -> {'MATCH' if match else 'MISMATCH'}")

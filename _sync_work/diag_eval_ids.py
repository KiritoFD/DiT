"""定位评测时 embedding 索引越界的来源。

崩溃现象: Indexing.cu:1308 `srcIndex < srcSelectDimSize` failed
位置: in_mem_eval -> sample_latents -> forward_with_cfg

模型表大小（v13 base）:
  y_callig_embedder: num_calligraphers(45) + 1(null) = 46 行
  y_char_embedder  : num_characters = 7765 行

逐个检查评测集的 id 是否越界。
"""
import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")

N_CALLIG = 45
N_CHAR = 7765

cmap_raw = {int(k): int(v) for k, v in
            json.load(open("assets/callig_id_map_50k.json", encoding="utf-8"))["id_map"].items()}
print(f"  callig_id_map_50k: {len(cmap_raw)} 条, raw ids {sorted(cmap_raw)[:6]} ...")
print(f"  模型表: callig={N_CALLIG}(+1 null), char={N_CHAR}")
print()

for name, path in (("seen  ", "assets/eval_seen_v10.csv"),
                   ("strict", "assets/eval_fame3_strict_clean_v9.csv")):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cal_raw = [int(r["calligrapher_id"]) for r in rows]
    cal_mapped = [cmap_raw.get(c, None) for c in cal_raw]
    glyph = [int(r.get("glyph_id", r["character_id"])) for r in rows]

    miss = [c for c, m in zip(cal_raw, cal_mapped) if m is None]
    mapped_ok = [m for m in cal_mapped if m is not None]

    print(f"  === {name} ({len(rows)} 行) ===")
    print(f"    calligrapher_id 原始: {min(cal_raw)}..{max(cal_raw)}")
    print(f"    -> 不在 callig_id_map 里的: {len(miss)} 个 {miss[:6]}")
    if mapped_ok:
        print(f"    -> 映射后: {min(mapped_ok)}..{max(mapped_ok)} "
              f"(表大小 {N_CALLIG}，越界: {sum(1 for m in mapped_ok if m >= N_CALLIG)})")
    print(f"    glyph_id: {min(glyph)}..{max(glyph)}  "
          f"-> 超过 char 表({N_CHAR}) 的: {sum(1 for g in glyph if g >= N_CHAR)} 个")
    print()

# -*- coding: utf-8 -*-
"""构建 楷/隶 子集数据库 + 标准字形latent映射。

过滤 5script/train.csv 只保留 script_id∈{0(楷),4(隶)} 的行;
为每行计算其 glyph 的标准字形 latent 键, 记录到新 csv:
  kailishu_train.csv  columns: 原列 + std_glyph_key (如 kai/U+XXXXX 或 li/U+XXXXX)
同时输出 manifest: glyph_key -> 训练样本数, 供 dataset 按需加载。

注意: 标准字形 latent 按 char(不是 script×char) 提供 —— 楷/隶同字共用同一标准字形图?
  否: 楷书字用楷体标准图, 隶书字用隶书标准图(同 char 但不同书体 → 不同标准图)。
  因此 std_glyph_key = {kai|li}/U+{ord(char):05X}。
"""
import csv, os, json, glob
from collections import defaultdict

SRC = "5script/train.csv"
OUT = "kailishu_train.csv"
KAILI_BOOKS = {0: "kai", 4: "li"}   # script_id -> 书体 key(对应标准字形库子目录)

def main():
    # 可用标准字形 latent(key 存在才保留该样本)
    # latent 目录: std_glyph_latent/{kai,li}/U+XXXXX.npy   (远程路径)
    usable = {}
    for book in ["kai", "li"]:
        for p in glob.glob(f"std_glyph_latent/{book}/U+*.npy"):
            usable[f"{book}/{os.path.splitext(os.path.basename(p))[0]}"] = True

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    out_rows, dropped_no_latent, dropped_non_kaili = [], 0, 0
    per_key_count = defaultdict(int)
    for r in rows:
        sid = int(r["script_id"])
        if sid not in KAILI_BOOKS:
            dropped_non_kaili += 1
            continue
        book = KAILI_BOOKS[sid]
        key = f"{book}/U+{ord(r['character']):05X}"
        if key not in usable:
            dropped_no_latent += 1
            continue
        r["std_glyph_key"] = key
        out_rows.append(r)
        per_key_count[key] += 1

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() | {"std_glyph_key"})
        w.writeheader()
        w.writerows(out_rows)

    print(f"原 {len(rows)} 行 -> 楷隶 {len(out_rows)} 行")
    print(f"  丢弃: 非楷隶 {dropped_non_kaili}, 无标准latent {dropped_no_latent}")
    print(f"  覆盖标准字形 key: {len(per_key_count)}, 总样本 {sum(per_key_count.values())}")
    with open("kailishu_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"rows": len(out_rows), "std_glyph_count": len(per_key_count),
                   "per_key": dict(per_key_count)}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

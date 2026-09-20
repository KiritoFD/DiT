#!/usr/bin/env python
"""从 wild_extract 里挑一个**训练集没见过**的书家，生成 few-shot 的 train/eval CSV。

背景 (2026-09-20):
  实测书家间 strict 差异与训练样本数**负相关**(r=-0.62) —— 样本多的书家风格跨度大，
  而书家条件是一个**预训练且冻结的 128 维向量**，装不下多模态风格。
  few-shot 是检验这件事最干净的实验：拿一个全新书家、只给少量图、冻结主干只训
  该书家的 embedding 新行，看能否泛化到它留出的字。

用法:
  python tools/build_fewshot_csv.py --cal 怀素 --script 草 \
      --k 10 --eval-n 100 --out-prefix assets/fewshot_huaisu

产物:
  <out-prefix>_train.csv    K 张训练图
  <out-prefix>_eval.csv     留出的 N 张（用于测泛化）

⚠ 字符→标准字形(g) 的映射来自 50k 训练集: wild 里有些生僻字在 50k 里没出现过，
  那些字**没有 std_path**，会被跳过（否则 g=ZERO，字条件失效 —— 这个坑踩过）。
"""
import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np

WILD = "/root/Workspace/xy/HCSU/wild_extract"
ROOT = "/root/Workspace/xy/DiT"


def load_50k_glyph_map(path="assets/train_50k_v2.csv"):
    """字符 -> (glyph_id, std_path, character_id)。取每个字的第一条。"""
    m = {}
    for r in csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8")):
        ch = r["character"]
        if ch in m:
            continue
        m[ch] = {
            "glyph_id": r["glyph_id"],
            "std_path": r["std_path"],
            "character_id": r["character_id"],
            "script_id": r.get("script_id", "0"),
        }
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True, help="书家名，如 怀素")
    ap.add_argument("--script", default="", help="书体，如 草；留空=合并该书家所有书体")
    ap.add_argument("--k", type=int, default=10, help="few-shot 训练图数")
    ap.add_argument("--eval-n", type=int, default=100, help="留出的评测图数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--min-per-char", type=int, default=1,
                    help="只保留在训练/eval 里都不重复的字（默认每个字只用一次）")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    # 收集该书家的图。文件名(不含扩展名) = 字符
    dirs = []
    if a.script:
        d = os.path.join(WILD, f"{a.cal}-{a.script}")
        if os.path.isdir(d):
            dirs = [d]
    else:
        for name in sorted(os.listdir(WILD)):
            if name.startswith(a.cal + "-") and os.path.isdir(os.path.join(WILD, name)):
                dirs.append(os.path.join(WILD, name))
    if not dirs:
        raise SystemExit(f"找不到 {a.cal}-{a.script or '*'} 的目录（在 {WILD}）")

    items = []
    for d in dirs:
        script = os.path.basename(d).partition("-")[2]
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            items.append({"file": os.path.join(d, f),
                          "character": os.path.splitext(f)[0],
                          "script": script})
    print(f"[fewshot] {a.cal}: 从 {len(dirs)} 个目录读到 {len(items)} 张")

    # 映射标准字形
    gm = load_50k_glyph_map()
    kept, no_glyph = [], 0
    for it in items:
        g = gm.get(it["character"])
        if not g or not g["std_path"]:
            no_glyph += 1
            continue
        it.update(g)
        kept.append(it)
    print(f"  有标准字形(g) 的: {len(kept)}   跳过(50k 里没这个字): {no_glyph}")

    # 每个字只留一张，避免 train/eval 泄漏同字不同图
    by_char = defaultdict(list)
    for it in kept:
        by_char[it["character"]].append(it)
    uniq = [v[0] for v in by_char.values()]
    print(f"  去重后(每字一张): {len(uniq)}")

    if len(uniq) < a.k + a.eval_n:
        raise SystemExit(f"可用字只有 {len(uniq)}，不够 K={a.k} + eval={a.eval_n}")

    random.shuffle(uniq)
    # ⚠ 先切出**固定**的 eval 块，再从剩下的池里取 K 张训练。
    #   否则不同 K 的留出集不一样，K 扫描之间无法横向比较（实测踩过）。
    ev = uniq[:a.eval_n]
    pool = uniq[a.eval_n:]
    train = pool[:a.k]
    print(f"  eval {len(ev)} 张（固定，与 K 无关）/ train {len(train)} 张")

    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out_prefix)) or ".", exist_ok=True)
    cols = ["image_path", "calligrapher", "script", "character",
            "calligrapher_id", "script_id", "character_id", "glyph_id",
            "aug", "std_path", "source", "src_image_path", "old_50k_id"]
    for name, rows in (("train", train), ("eval", ev)):
        p = os.path.join(ROOT, f"{a.out_prefix}_{name}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for it in rows:
                w.writerow({
                    "image_path": os.path.relpath(it["file"], ROOT),
                    "calligrapher": a.cal,
                    "script": it["script"],
                    "character": it["character"],
                    "calligrapher_id": "45",          # ★ 新增行 = 第 46 个(0-based 45)
                    "script_id": it.get("script_id", "0"),
                    "character_id": it["character_id"],
                    "glyph_id": it["glyph_id"],
                    "aug": "", "std_path": it["std_path"],
                    "source": "fewshot_wild",
                    "src_image_path": it["file"],
                    "old_50k_id": "",
                })
        print(f"  written {p}")

    print("\n[fewshot] 记住: 训练时要用 --num-calligraphers 46 "
          "（45 个原有 + 1 个新书家）")


if __name__ == "__main__":
    main()

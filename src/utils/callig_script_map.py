# -*- coding: utf-8 -*-
"""callig_script_map.py — (书家 × 书体) 联合风格词表.

## 为什么需要 (2026-09-18)
实测 (`docs/system/73`, `logs/_div2.log`):
  - 书家条件目前是**一个预训练且冻结的 128 维单向量**;
  - **32/45 书家跨多书体**(楷/行/隶), 单向量把一个人的多种书体**平均**掉 ->
    风格被冲淡 -> `ratio_style = inter_callig/intra ≈ 1.18`(换书家≈换噪声, "假风格控制");
  - 书家间 strict 差异与样本数**负相关 r=-0.62**(样本多的书家风格跨度大, 单向量装不下)。

把风格类别从 `calligrapher_id`(45) 细化为 `(calligrapher_id, script_id)`(**87 对**),
让同一个人的不同书体各得一个向量 -> 直接解耦"书体模态"。87 对里中位 508 张/对,
足够训一个 128 维向量; 仅 15 对 <50 张(min=1) -> 用**书家级质心回退**(见预训练脚本)。

## 设计要点
- 这是**标签级**改动: 模型仍把 pair 当"书家类"看(`num_calligraphers=87`),
  架构/预训练表加载/冻结逻辑**完全复用**, 是 vs 当前 base 的干净单变量实验。
- 词表按 `(calligrapher_id, script_id)` 排序映射到 `0..N-1`, 保证 train/eval/inference 三端一致。
- 同时保存 `pair -> calligrapher` 与每对的样本数, 供预训练的层级 SupCon / 稀疏回退 / 分析用。

## 产物 JSON 结构
    {
      "num_pairs": 87, "num_calligraphers": 45, "min_samples": 50,
      "pair_map": {"<callig>:<script>": pair_id, ...},          # 87
      "callig_map": {"<callig>": callig_idx, ...},               # 45 (0..44)
      "pair_to_callig": [callig_idx, ...],                       # len 87
      "callig_to_default_pair": {"<callig>": pair_id, ...},      # 未见 pair 的回退
      "pair_counts": {"<callig>:<script>": n, ...},              # 每对样本数
      "sparse_pairs": ["<callig>:<script>", ...]                 # n < min_samples
    }
"""
import csv
import json
from collections import Counter, defaultdict


def _key(callig_id, script_id):
    return f"{int(callig_id)}:{int(script_id)}"


def build_callig_script_map(csv_paths, min_samples=50):
    """扫描若干 csv, 收集所有 (calligrapher_id, script_id) 对, 排序映射到 0..N-1.

    返回上文 JSON 结构的 dict。排序保证三端一致(与 build_callig_id_map 同规则)。
    """
    pair_count = Counter()
    callig_count = Counter()
    for p in csv_paths:
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = int(row["calligrapher_id"])
                s = int(row["script_id"])
                pair_count[_key(c, s)] += 1
                callig_count[c] += 1

    # 连续索引: 书家按 raw id 排序 -> callig_idx; pair 按 (callig_idx, script) 排序 -> pair_id
    callig_map = {c: i for i, c in enumerate(sorted(callig_count))}
    pair_keys = sorted(pair_count.keys(),
                       key=lambda k: (callig_map[int(k.split(":")[0])], int(k.split(":")[1])))
    pair_map = {k: i for i, k in enumerate(pair_keys)}
    pair_to_callig = [callig_map[int(k.split(":")[0])] for k in pair_keys]

    # 每个书家的"默认 pair"(样本最多的那个) —— eval/推理遇到未见 (callig,script) 时回退
    callig_pairs = defaultdict(list)
    for k in pair_keys:
        c = int(k.split(":")[0])
        callig_pairs[c].append((pair_count[k], pair_map[k]))
    callig_to_default_pair = {str(c): max(v)[1] for c, v in callig_pairs.items()}

    sparse_pairs = [k for k in pair_keys if pair_count[k] < min_samples]

    return {
        "num_pairs": len(pair_map),
        "num_calligraphers": len(callig_map),
        "min_samples": int(min_samples),
        "pair_map": pair_map,
        "callig_map": {str(c): i for c, i in callig_map.items()},
        "pair_to_callig": pair_to_callig,
        "callig_to_default_pair": callig_to_default_pair,
        "pair_counts": {k: pair_count[k] for k in pair_keys},
        "sparse_pairs": sparse_pairs,
    }


def save_callig_script_map(mapping, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_callig_script_map(path):
    """读回映射表。返回 dict(与 build 同结构)。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def map_callig_script(callig_id, script_id, mapping):
    """(raw calligrapher_id, script_id) -> pair_id(0..N-1).

    mapping 为 None -> 退回 raw calligrapher_id(零破坏旧行为)。
    未见过的 pair -> 回退到该书家的默认 pair(样本最多的那个); 书家也没见过 -> 0。
    """
    if mapping is None:
        return int(callig_id)
    pm = mapping["pair_map"]
    k = _key(callig_id, script_id)
    if k in pm:
        return int(pm[k])
    # 回退: 该书家的默认 pair
    d = mapping.get("callig_to_default_pair", {}).get(str(int(callig_id)))
    if d is not None:
        return int(d)
    return 0


# ── CLI: 从 csv 构建并保存映射 ────────────────────────────────────────────
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="构建 (书家×书体) 联合风格词表")
    ap.add_argument("--csv", nargs="+", required=True, help="训练集 csv(可多个)")
    ap.add_argument("--out", default="assets/callig_script_id_map.json")
    ap.add_argument("--min-samples", type=int, default=50,
                    help="低于此样本数的 pair 记为 sparse(预训练时质心回退到书家级)")
    a = ap.parse_args()
    m = build_callig_script_map(a.csv, min_samples=a.min_samples)
    save_callig_script_map(m, a.out)
    print(f"[callig-script] {m['num_calligraphers']} 书家 -> {m['num_pairs']} 个 "
          f"(书家,书体) 对; sparse(<{a.min_samples}) {len(m['sparse_pairs'])} 对")
    print(f"[callig-script] 每对样本数: "
          f"min={min(m['pair_counts'].values())} "
          f"med={sorted(m['pair_counts'].values())[len(m['pair_counts'])//2]} "
          f"max={max(m['pair_counts'].values())}")
    print(f"[callig-script] -> {a.out}")
    if m["sparse_pairs"]:
        print(f"[callig-script] sparse pairs(将质心回退): "
              f"{[(k, m['pair_counts'][k]) for k in m['sparse_pairs'][:20]]}")


if __name__ == "__main__":
    _main()

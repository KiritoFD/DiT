"""callig_map.py — 书家词表收紧: 稀疏 calligrapher_id → 连续 0..N-1 的干净词表.

背景 (2026-09-08, sp2 诊断): fame3 训练集只有 41 个唯一书家, 但 embedding 表
num_calligraphers=1013 (为覆盖稀疏 id 最大值 994 而设), 96% 行是死行; 且端到端
训练下 41 书家 embedding 塌缩(pairwise cos 0.323)。这里提供干净词表的基础设施:
把稀疏 raw id 映射到连续索引, 让 num_calligraphers 收紧为真实书家数 N。
"""
import csv
import json


def build_callig_id_map(csv_paths):
    """扫描若干 csv, 收集所有 calligrapher_id, 按数值排序映射到 0..N-1.

    返回 {raw_id(int): idx(int)}。排序保证 train/eval/inference 三端一致。
    """
    ids = set()
    for p in csv_paths:
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids.add(int(row["calligrapher_id"]))
    return {raw: i for i, raw in enumerate(sorted(ids))}


def save_callig_id_map(mapping, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"num_calligraphers": len(mapping),
                   "id_map": {str(k): v for k, v in mapping.items()}},
                  f, ensure_ascii=False, indent=2)


def load_callig_id_map(path):
    """读回映射表, 返回 (mapping{dict[int,int]}, num_calligraphers{int})."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v for k, v in d["id_map"].items()}, int(d["num_calligraphers"])


def map_callig_id(raw_id, mapping):
    """raw calligrapher_id -> 连续索引; mapping 为 None 或未知 id 时回退原值(零破坏旧行为)."""
    if mapping is None:
        return int(raw_id)
    return mapping.get(int(raw_id), int(raw_id))